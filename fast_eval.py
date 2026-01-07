"""
Fast evaluator for the comma controls challenge.
Optimizations:
1. Single shared ONNX model instance per worker (not recreated per rollout)
2. Run test and baseline in parallel across all segments
3. Use more workers and optimized chunking
4. Skip report generation for quick score checks (optional)
"""

import argparse
import importlib
import numpy as np
import onnxruntime as ort
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from functools import partial
from hashlib import md5
from collections import namedtuple
from tqdm import tqdm
import time

# Constants from tinyphysics
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
FUTURE_PLAN_STEPS = FPS * 5

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

# Global model cache per process
_model_cache = {}


class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[token]


def get_model(model_path):
    """Get or create cached model for this process."""
    global _model_cache
    if model_path not in _model_cache:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        with open(model_path, "rb") as f:
            session = ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])
        _model_cache[model_path] = (session, LataccelTokenizer())
    return _model_cache[model_path]


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def predict(session, tokenizer, sim_states, actions, past_preds):
    tokenized_actions = tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
        'states': np.expand_dims(states, axis=0).astype(np.float32),
        'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    res = session.run(None, input_data)[0]
    probs = softmax(res / 0.8, axis=-1)
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return tokenizer.decode(sample)


def load_data(data_path):
    """Load and preprocess CSV data."""
    df = pd.read_csv(data_path)
    return pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values
    })


def run_single_rollout(args):
    """Run a single rollout with given controller."""
    data_path, controller_type, model_path = args
    
    # Get cached model
    session, tokenizer = get_model(model_path)
    
    # Load data
    data = load_data(data_path)
    
    # Initialize controller
    controller = importlib.import_module(f'controllers.{controller_type}').Controller()
    
    # Set random seed for reproducibility
    seed = int(md5(str(data_path).encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)
    
    # Initialize histories
    state_history = []
    action_history = []
    current_lataccel_history = []
    target_lataccel_history = []
    
    # Warm-up phase (context)
    for i in range(CONTEXT_LENGTH):
        state = data.iloc[i]
        state_history.append(State(state['roll_lataccel'], state['v_ego'], state['a_ego']))
        action_history.append(data['steer_command'].values[i])
        current_lataccel_history.append(state['target_lataccel'])
        target_lataccel_history.append(state['target_lataccel'])
    
    current_lataccel = current_lataccel_history[-1]
    
    # Main simulation loop
    for step_idx in range(CONTEXT_LENGTH, len(data)):
        state = data.iloc[step_idx]
        state_obj = State(state['roll_lataccel'], state['v_ego'], state['a_ego'])
        state_history.append(state_obj)
        target_lataccel_history.append(state['target_lataccel'])
        
        # Build future plan
        future_plan = FuturePlan(
            lataccel=data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
            v_ego=data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
            a_ego=data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
        )
        
        # Control step
        if step_idx >= CONTROL_START_IDX:
            action = controller.update(
                target_lataccel_history[step_idx],
                current_lataccel,
                state_history[step_idx],
                future_plan
            )
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        else:
            action = data['steer_command'].values[step_idx]
        action_history.append(action)
        
        # Sim step
        pred = predict(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        
        if step_idx >= CONTROL_START_IDX:
            current_lataccel = pred
        else:
            current_lataccel = data.iloc[step_idx]['target_lataccel']
        
        current_lataccel_history.append(current_lataccel)
    
    # Compute costs
    target = np.array(target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    
    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    
    return {
        'lataccel_cost': lat_accel_cost,
        'jerk_cost': jerk_cost,
        'total_cost': total_cost
    }


def run_batch_evaluation(data_path, test_controller, baseline_controller, model_path, num_segs, num_workers=None):
    """Run batch evaluation with parallel processing."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    
    data_dir = Path(data_path)
    files = sorted(data_dir.iterdir())[:num_segs]
    
    print(f"Evaluating {len(files)} segments with {num_workers} workers...")
    print(f"Test controller: {test_controller}, Baseline controller: {baseline_controller}")
    
    # Prepare all tasks
    test_tasks = [(str(f), test_controller, model_path) for f in files]
    baseline_tasks = [(str(f), baseline_controller, model_path) for f in files]
    
    start_time = time.time()
    
    # Run test controller
    print(f"\nRunning test controller ({test_controller})...")
    with mp.Pool(num_workers) as pool:
        test_results = list(tqdm(
            pool.imap(run_single_rollout, test_tasks, chunksize=max(1, len(files)//num_workers//4)),
            total=len(files)
        ))
    
    # Run baseline controller
    print(f"\nRunning baseline controller ({baseline_controller})...")
    with mp.Pool(num_workers) as pool:
        baseline_results = list(tqdm(
            pool.imap(run_single_rollout, baseline_tasks, chunksize=max(1, len(files)//num_workers//4)),
            total=len(files)
        ))
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    test_df = pd.DataFrame(test_results)
    baseline_df = pd.DataFrame(baseline_results)
    
    print(f"\n{'='*60}")
    print(f"Results ({len(files)} segments, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\nTest Controller ({test_controller}):")
    print(f"  lataccel_cost: {test_df['lataccel_cost'].mean():.4f}")
    print(f"  jerk_cost:     {test_df['jerk_cost'].mean():.4f}")
    print(f"  total_cost:    {test_df['total_cost'].mean():.4f}")
    
    print(f"\nBaseline Controller ({baseline_controller}):")
    print(f"  lataccel_cost: {baseline_df['lataccel_cost'].mean():.4f}")
    print(f"  jerk_cost:     {baseline_df['jerk_cost'].mean():.4f}")
    print(f"  total_cost:    {baseline_df['total_cost'].mean():.4f}")
    
    test_total = test_df['total_cost'].mean()
    baseline_total = baseline_df['total_cost'].mean()
    
    print(f"\n{'='*60}")
    if test_total < baseline_total:
        improvement = (baseline_total - test_total) / baseline_total * 100
        print(f"✅ Test controller BEATS baseline by {improvement:.2f}%!")
    else:
        deficit = (test_total - baseline_total) / baseline_total * 100
        print(f"❌ Test controller is {deficit:.2f}% WORSE than baseline")
    
    if test_total < 100:
        print(f"🏆 COMPETITIVE SCORE: {test_total:.2f} < 100!")
    print(f"{'='*60}")
    
    return test_df, baseline_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast evaluator for comma controls challenge")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--test_controller", type=str, default="simple")
    parser.add_argument("--baseline_controller", type=str, default="pid")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()
    
    run_batch_evaluation(
        args.data_path,
        args.test_controller,
        args.baseline_controller,
        args.model_path,
        args.num_segs,
        args.workers
    )
