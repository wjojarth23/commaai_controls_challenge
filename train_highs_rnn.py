"""
Train an Encoder-Decoder RNN Controller using CMA-ES Optimization

This script:
1. Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to find optimal 
   control sequences. CMA-ES is derivative-free and works well on non-smooth problems.
2. Trains an encoder-decoder RNN on those optimal sequences

The key insight: The ONNX physics model outputs DISCRETE tokens (quantized lataccel).
This makes gradient-based optimization fail. CMA-ES doesn't need gradients.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse
import onnxruntime as ort
from collections import namedtuple
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# Try to import CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("CMA-ES not found. Install with: pip install cma")

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

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins_np = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins_np, right=True)

    def decode(self, token):
        return self.bins_np[token]


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


def get_onnx_session(model_path, use_cuda=True):
    """Load the ONNX physics model with optional CUDA support."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    
    # Try CUDA first if requested
    providers = []
    if use_cuda:
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
            print("Using CUDA for ONNX inference")
        elif 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
            print("Using DirectML for ONNX inference")
    
    providers.append('CPUExecutionProvider')
    
    with open(model_path, "rb") as f:
        session = ort.InferenceSession(f.read(), options, providers)
    
    # Print which provider is being used
    actual_provider = session.get_providers()[0]
    if actual_provider == 'CPUExecutionProvider' and use_cuda:
        print(f"Note: Running on CPU (CUDA not available). Install onnxruntime-gpu for GPU support.")
    
    return session


def predict_lataccel(session, tokenizer, states, actions, past_lataccels):
    """Run the ONNX physics model."""
    tokenized = tokenizer.encode(np.array(past_lataccels))
    states_np = np.column_stack([actions, [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states]])
    input_data = {
        'states': np.expand_dims(states_np, axis=0).astype(np.float32),
        'tokens': np.expand_dims(tokenized, axis=0).astype(np.int64)
    }
    res = session.run(None, input_data)[0]
    token = np.argmax(res[0, -1])
    return tokenizer.decode(token)


def simulate_trajectory(session, tokenizer, data, actions):
    """
    Simulate the full trajectory with given actions using the ONNX physics model.
    Returns the predicted lateral accelerations for the CONTROL period only.
    """
    states_data = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    targets = data['target_lataccel'].values
    
    # Initialize
    state_history = list(states_data[:CONTEXT_LENGTH])
    action_history = list(actions[:CONTEXT_LENGTH])
    lataccel_history = list(targets[:CONTEXT_LENGTH])
    current_lataccel = lataccel_history[-1]
    
    # Run pre-control period (use ground truth lataccel)
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states_data[step])
        action_history.append(actions[step])
        current_lataccel = targets[step]
        lataccel_history.append(current_lataccel)
    
    # Run control period (use physics model)
    control_lataccels = []
    for step in range(CONTROL_START_IDX, min(COST_END_IDX, len(data))):
        state_history.append(states_data[step])
        action_history.append(actions[step])
        
        pred = predict_lataccel(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = pred
        lataccel_history.append(current_lataccel)
        control_lataccels.append(pred)
    
    return np.array(control_lataccels)


def compute_cost(predictions, targets):
    """Compute the total cost matching tinyphysics.py exactly."""
    lat_accel_cost = np.mean((targets - predictions)**2) * 100
    jerk = np.diff(predictions) / DEL_T
    jerk_cost = np.mean(jerk**2) * 100
    total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    return total_cost, lat_accel_cost, jerk_cost


def simulate_with_pid(session, tokenizer, data, kp=0.17, ki=0.1, kd=-0.06, kff=0.13):
    """
    Run simulation with PID controller matching simple.py exactly.
    """
    states_data = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    targets = data['target_lataccel'].values
    initial_actions = data['steer_command'].values
    n_data = len(data)
    
    actions = np.zeros(n_data)
    
    # Initialize
    state_history = list(states_data[:CONTEXT_LENGTH])
    action_history = list(initial_actions[:CONTEXT_LENGTH])
    lataccel_history = list(targets[:CONTEXT_LENGTH])
    current_lataccel = lataccel_history[-1]
    
    actions[:CONTEXT_LENGTH] = initial_actions[:CONTEXT_LENGTH]
    
    # PID state (matching simple.py exactly)
    error_integral = 0.0
    prev_error = 0.0
    
    # Run pre-control period
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states_data[step])
        action_history.append(initial_actions[step])
        actions[step] = initial_actions[step]
        current_lataccel = targets[step]
        lataccel_history.append(current_lataccel)
    
    # Run control period with PID
    control_lataccels = []
    for step in range(CONTROL_START_IDX, min(COST_END_IDX, n_data)):
        target = targets[step]
        error = target - current_lataccel
        error_integral += error  # NO multiplication by DEL_T (matches simple.py)
        error_diff = error - prev_error
        prev_error = error
        
        # PID + feedforward (matching simple.py)
        fb = kp * error + ki * error_integral + kd * error_diff
        ff = kff * target
        action = fb + ff
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        actions[step] = action
        
        state_history.append(states_data[step])
        action_history.append(action)
        
        pred = predict_lataccel(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = pred
        lataccel_history.append(current_lataccel)
        control_lataccels.append(pred)
    
    # Fill remaining
    for step in range(COST_END_IDX, n_data):
        actions[step] = actions[COST_END_IDX - 1]
    
    return actions, np.array(control_lataccels)


def optimize_pid_gains_cmaes(session, tokenizer, data, verbose=True):
    """
    Optimize PID gains using CMA-ES. This is a 4-dimensional problem
    (kp, ki, kd, kff) instead of 400-dimensional, so it's MUCH faster.
    """
    if not HAS_CMA:
        return None, None, float('inf')
    
    targets = data['target_lataccel'].values
    target_slice = targets[CONTROL_START_IDX:COST_END_IDX]
    
    eval_count = [0]
    
    def evaluate_pid(params):
        eval_count[0] += 1
        kp, ki, kd, kff = params
        actions, preds = simulate_with_pid(session, tokenizer, data, kp, ki, kd, kff)
        cost, _, _ = compute_cost(preds, target_slice[:len(preds)])
        return cost
    
    # Starting point
    x0 = [0.17, 0.10, -0.06, 0.13]
    initial_cost = evaluate_pid(x0)
    
    if verbose:
        print(f"    Tuning PID gains with CMA-ES (4D optimization)...")
        print(f"    Initial PID cost: {initial_cost:.2f}")
    
    # CMA-ES options - much faster for 4D
    opts = {
        'maxfevals': 500,
        'bounds': [[0.05, 0.01, -0.2, 0.0], [0.5, 0.3, 0.0, 0.3]],
        'tolfun': 0.01,
        'verbose': -9,
        'popsize': 12,
    }
    
    best_cost = initial_cost
    best_params = x0
    
    es = cma.CMAEvolutionStrategy(x0, 0.05, opts)
    
    start_time = time.time()
    last_print = time.time()
    
    while not es.stop():
        solutions = es.ask()
        fitness = [evaluate_pid(x) for x in solutions]
        es.tell(solutions, fitness)
        
        if es.result.fbest < best_cost:
            best_cost = es.result.fbest
            best_params = es.result.xbest.copy()
        
        if verbose and time.time() - last_print > 2:
            print(f"    PID CMA-ES: gen {es.countiter}, {eval_count[0]} evals, best={best_cost:.2f}")
            last_print = time.time()
    
    kp, ki, kd, kff = best_params
    actions, preds = simulate_with_pid(session, tokenizer, data, kp, ki, kd, kff)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"    PID optimization done in {elapsed:.1f}s")
        print(f"    Best gains: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}, kff={kff:.4f}")
        print(f"    Cost: {initial_cost:.2f} -> {best_cost:.2f} ({(initial_cost-best_cost)/initial_cost*100:.1f}% improvement)")
    
    return actions, best_params, best_cost


class CMAESOptimizer:
    """
    CMA-ES optimizer for finding optimal control sequences.
    
    Why CMA-ES?
    - The ONNX physics model outputs DISCRETE tokens (quantized lataccel)
    - Small perturbations often don't change the output at all
    - Gradient-based methods fail because the gradient is essentially zero
    - CMA-ES is derivative-free and handles non-smooth objectives well
    """
    
    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer
        self.eval_count = 0
        self.best_cost_history = []
    
    def create_cost_function(self, data):
        """Create a cost function for the given trajectory."""
        targets = data['target_lataccel'].values
        target_slice = targets[CONTROL_START_IDX:COST_END_IDX]
        n_control = len(target_slice)
        initial_actions = data['steer_command'].values
        
        def evaluate_cost(control_actions):
            """Evaluate cost for given control actions."""
            self.eval_count += 1
            
            # Build full action array
            actions = np.zeros(len(data))
            actions[:CONTROL_START_IDX] = initial_actions[:CONTROL_START_IDX]
            actions[CONTROL_START_IDX:COST_END_IDX] = np.clip(control_actions, STEER_RANGE[0], STEER_RANGE[1])
            actions[COST_END_IDX:] = control_actions[-1] if len(control_actions) > 0 else 0
            
            # Simulate
            preds = simulate_trajectory(self.session, self.tokenizer, data, actions)
            
            # Compute cost
            total_cost, lat_cost, jerk_cost = compute_cost(preds, target_slice[:len(preds)])
            return total_cost
        
        return evaluate_cost, n_control, initial_actions
    
    def optimize_coordinate_descent(self, data, initial_control_actions, max_iter=3, verbose=True):
        """
        Coordinate descent optimizer - optimizes one action at a time.
        Much faster than CMA-ES for sequential problems.
        """
        evaluate_cost, n_control, _ = self.create_cost_function(data)
        
        x = initial_control_actions[:n_control].copy()
        best_cost = evaluate_cost(x)
        
        if verbose:
            print(f"    Initial cost: {best_cost:.2f}")
        
        # Step sizes to try
        step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01]
        
        for iteration in range(max_iter):
            improved = False
            start_time = time.time()
            
            # Sweep through each control variable
            for i in tqdm(range(n_control), desc=f"    Iter {iteration+1}/{max_iter}", 
                         disable=not verbose, leave=False):
                original_val = x[i]
                
                for step in step_sizes:
                    # Try +step
                    x[i] = np.clip(original_val + step, STEER_RANGE[0], STEER_RANGE[1])
                    cost_plus = evaluate_cost(x)
                    
                    if cost_plus < best_cost:
                        best_cost = cost_plus
                        improved = True
                        break
                    
                    # Try -step
                    x[i] = np.clip(original_val - step, STEER_RANGE[0], STEER_RANGE[1])
                    cost_minus = evaluate_cost(x)
                    
                    if cost_minus < best_cost:
                        best_cost = cost_minus
                        improved = True
                        break
                    
                    # Restore
                    x[i] = original_val
            
            elapsed = time.time() - start_time
            if verbose:
                print(f"    Iter {iteration+1}: cost = {best_cost:.2f} ({elapsed:.1f}s, {self.eval_count} evals)")
            
            if not improved:
                if verbose:
                    print(f"    Converged - no improvement")
                break
        
        return x, best_cost
    
    def optimize_cmaes(self, data, initial_control_actions, max_evals=5000, verbose=True):
        """
        Optimize using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
        """
        if not HAS_CMA:
            print("CMA-ES not available. Falling back to coordinate descent.")
            return self.optimize_coordinate_descent(data, initial_control_actions, verbose=verbose)
        
        evaluate_cost, n_control, _ = self.create_cost_function(data)
        
        x0 = initial_control_actions[:n_control]
        initial_cost = evaluate_cost(x0)
        
        if verbose:
            print(f"    Initial cost: {initial_cost:.2f}")
            print(f"    Optimizing {n_control} variables with CMA-ES...")
        
        self.eval_count = 0
        self.best_cost_history = []
        best_cost = initial_cost
        best_x = x0.copy()
        
        # CMA-ES options
        opts = {
            'maxfevals': max_evals,
            'bounds': [STEER_RANGE[0], STEER_RANGE[1]],
            'tolfun': 0.1,  # Stop if change in function value is small
            'tolx': 1e-6,
            'verbose': -9 if not verbose else 1,  # Suppress internal output
            'CMA_diagonal': True,  # Use diagonal covariance (faster for high-dim)
            'popsize': 20,  # Population size
        }
        
        # Callback to track progress
        last_print_time = [time.time()]
        
        def callback_fn(es):
            nonlocal best_cost, best_x
            if es.result.fbest < best_cost:
                best_cost = es.result.fbest
                best_x = es.result.xbest.copy()
            
            self.best_cost_history.append(best_cost)
            
            # Print progress every 5 seconds
            if verbose and time.time() - last_print_time[0] > 5:
                print(f"    CMA-ES: gen {es.countiter}, {self.eval_count} evals, "
                      f"best={best_cost:.2f}, sigma={es.sigma:.4f}")
                last_print_time[0] = time.time()
        
        # Run CMA-ES
        sigma0 = 0.1  # Initial step size
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        while not es.stop():
            solutions = es.ask()
            fitness = [evaluate_cost(x) for x in solutions]
            es.tell(solutions, fitness)
            callback_fn(es)
        
        if verbose:
            print(f"    CMA-ES finished: {es.countiter} generations, {self.eval_count} evals, "
                  f"final cost = {best_cost:.2f}")
            print(f"    Improvement: {initial_cost:.2f} -> {best_cost:.2f} ({(initial_cost-best_cost)/initial_cost*100:.1f}%)")
        
        return best_x, best_cost


def optimize_pid_gains_cmaes(session, tokenizer, data, verbose=True):
    """
    Optimize PID gains using CMA-ES. This is a 4-dimensional problem
    (kp, ki, kd, kff) instead of 400-dimensional, so it's MUCH faster.
    """
    if not HAS_CMA:
        return None, None, float('inf')
    
    targets = data['target_lataccel'].values
    target_slice = targets[CONTROL_START_IDX:COST_END_IDX]
    
    eval_count = [0]
    
    def evaluate_pid(params):
        eval_count[0] += 1
        kp, ki, kd, kff = params
        actions, preds = simulate_with_pid(session, tokenizer, data, kp, ki, kd, kff)
        cost, _, _ = compute_cost(preds, target_slice[:len(preds)])
        return cost
    
    # Starting point
    x0 = [0.17, 0.10, -0.06, 0.13]
    initial_cost = evaluate_pid(x0)
    
    if verbose:
        print(f"    Tuning PID gains with CMA-ES (4D optimization)...")
        print(f"    Initial PID cost: {initial_cost:.2f}")
    
    # CMA-ES options - much faster for 4D
    opts = {
        'maxfevals': 500,
        'bounds': [[0.05, 0.01, -0.2, 0.0], [0.5, 0.3, 0.0, 0.3]],
        'tolfun': 0.01,
        'verbose': -9,
        'popsize': 12,
    }
    
    best_cost = initial_cost
    best_params = x0
    
    es = cma.CMAEvolutionStrategy(x0, 0.05, opts)
    
    start_time = time.time()
    last_print = time.time()
    
    while not es.stop():
        solutions = es.ask()
        fitness = [evaluate_pid(x) for x in solutions]
        es.tell(solutions, fitness)
        
        if es.result.fbest < best_cost:
            best_cost = es.result.fbest
            best_params = es.result.xbest.copy()
        
        if verbose and time.time() - last_print > 2:
            print(f"    PID CMA-ES: gen {es.countiter}, {eval_count[0]} evals, best={best_cost:.2f}")
            last_print = time.time()
    
    kp, ki, kd, kff = best_params
    actions, preds = simulate_with_pid(session, tokenizer, data, kp, ki, kd, kff)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"    PID optimization done in {elapsed:.1f}s")
        print(f"    Best gains: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}, kff={kff:.4f}")
        print(f"    Cost: {initial_cost:.2f} -> {best_cost:.2f} ({(initial_cost-best_cost)/initial_cost*100:.1f}% improvement)")
    
    return actions, best_params, best_cost


class CMAESOptimizer:
        """
        Simple random perturbation search as fallback.
        """
        evaluate_cost, n_control, _ = self.create_cost_function(data)
        
        x = initial_control_actions[:n_control].copy()
        best_cost = evaluate_cost(x)
        best_x = x.copy()
        
        if verbose:
            print(f"    Initial cost: {best_cost:.2f}")
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            # Generate random perturbation
            noise_scale = 0.1 * (1 - iteration / max_iter)  # Decreasing noise
            noise = np.random.randn(n_control) * noise_scale
            
            x_new = np.clip(x + noise, STEER_RANGE[0], STEER_RANGE[1])
            cost = evaluate_cost(x_new)
            
            if cost < best_cost:
                best_cost = cost
                best_x = x_new.copy()
                x = x_new
            
            if verbose and (iteration + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"    Iter {iteration+1}: best cost = {best_cost:.2f} ({elapsed:.1f}s)")
        
        return best_x, best_cost


def optimize_single_trajectory(args):
    """Optimize a single trajectory."""
    data_path, model_path, verbose, optimizer_method, use_cuda = args
    
    session = get_onnx_session(model_path, use_cuda=use_cuda)
    tokenizer = LataccelTokenizer()
    data = load_data(data_path)
    targets = data['target_lataccel'].values
    target_slice = targets[CONTROL_START_IDX:COST_END_IDX]
    n_data = len(data)
    n_control = COST_END_IDX - CONTROL_START_IDX
    
    # Get PID baseline (this is our starting point)
    pid_actions, pid_preds = simulate_with_pid(session, tokenizer, data)
    pid_cost, _, _ = compute_cost(pid_preds, target_slice[:len(pid_preds)])
    
    if verbose:
        print(f"    PID baseline cost: {pid_cost:.2f}")
    
    best_actions = pid_actions.copy()
    best_cost = pid_cost
    best_kp, best_ki, best_kd, best_kff = 0.17, 0.10, -0.06, 0.13
    
    # Try different PID gains to find better starting point
    pid_gains = [
        (0.17, 0.10, -0.06, 0.13),  # Original
        (0.20, 0.10, -0.06, 0.13),
        (0.20, 0.12, -0.06, 0.15),
        (0.25, 0.10, -0.05, 0.12),
        (0.25, 0.12, -0.06, 0.15),
        (0.15, 0.08, -0.04, 0.10),
        (0.18, 0.10, -0.05, 0.12),
        (0.22, 0.11, -0.05, 0.14),
        (0.19, 0.09, -0.07, 0.12),
        (0.16, 0.12, -0.04, 0.14),
    ]
    
    for kp, ki, kd, kff in pid_gains:
        actions, preds = simulate_with_pid(session, tokenizer, data, kp, ki, kd, kff)
        cost, _, _ = compute_cost(preds, target_slice[:len(preds)])
        if cost < best_cost:
            best_cost = cost
            best_actions = actions.copy()
            best_kp, best_ki, best_kd, best_kff = kp, ki, kd, kff
    
    if verbose:
        print(f"    Best PID cost: {best_cost:.2f} (kp={best_kp}, ki={best_ki}, kd={best_kd}, kff={best_kff})")
    
    # Now optimize starting from PID
    optimizer = CMAESOptimizer(session, tokenizer)
    
    # Start from best PID actions
    initial_control = best_actions[CONTROL_START_IDX:COST_END_IDX]
    
    if optimizer_method == 'cmaes':
        optimized_control, opt_cost = optimizer.optimize_cmaes(
            data, initial_control, max_evals=3000, verbose=verbose
        )
    elif optimizer_method == 'coord':
        optimized_control, opt_cost = optimizer.optimize_coordinate_descent(
            data, initial_control, max_iter=3, verbose=verbose
        )
    else:  # random
        optimized_control, opt_cost = optimizer.optimize_random_search(
            data, initial_control, max_iter=100, verbose=verbose
        )
    
    # Build full action array
    optimized_actions = best_actions.copy()
    optimized_actions[CONTROL_START_IDX:COST_END_IDX] = optimized_control
    
    # Verify the cost
    final_preds = simulate_trajectory(session, tokenizer, data, optimized_actions)
    final_cost, lat_cost, jerk_cost = compute_cost(final_preds, target_slice[:len(final_preds)])
    
    if final_cost < best_cost:
        best_cost = final_cost
        best_actions = optimized_actions.copy()
    else:
        # Keep PID if optimization didn't improve
        final_preds = simulate_trajectory(session, tokenizer, data, best_actions)
        final_cost, lat_cost, jerk_cost = compute_cost(final_preds, target_slice[:len(final_preds)])
    
    if verbose:
        print(f"    Final cost: {final_cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f})")
    
    return {
        'file': str(data_path),
        'targets': targets.tolist(),
        'states': [(data.iloc[i]['roll_lataccel'], data.iloc[i]['v_ego'], data.iloc[i]['a_ego']) 
                  for i in range(len(data))],
        'initial_actions': data['steer_command'].values.tolist(),
        'optimal_actions': best_actions.tolist(),
        'final_cost': final_cost,
        'pid_baseline_cost': pid_cost,
    }


def generate_optimal_sequences(model_path, data_dir, output_path, num_files=100, num_workers=4, optimizer_method='coord', use_cuda=True):
    """Generate optimal control sequences."""
    print(f"=== Generating Optimal Control Sequences with {optimizer_method.upper()} ===")
    
    # Check available ONNX providers
    available = ort.get_available_providers()
    print(f"Available ONNX providers: {available}")
    
    data_files = sorted(Path(data_dir).glob("*.csv"))[:num_files]
    
    verbose = num_files <= 5
    # Note: CUDA can't be shared across processes, so disable for multiprocessing
    use_cuda_actual = use_cuda and (num_workers <= 1 or num_files <= 4)
    if use_cuda and not use_cuda_actual:
        print("Note: Using CPU for multiprocessing (CUDA sessions can't be shared across processes)")
    
    args_list = [(str(f), model_path, verbose, optimizer_method, use_cuda_actual) for f in data_files]
    
    all_sequences = []
    
    if num_workers > 1 and num_files > 4:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(optimize_single_trajectory, args): args[0] 
                      for args in args_list}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Optimizing"):
                result = future.result()
                all_sequences.append(result)
                if not verbose:
                    tqdm.write(f"  {Path(result['file']).name}: cost={result['final_cost']:.2f} (PID baseline={result['pid_baseline_cost']:.2f})")
    else:
        for args in tqdm(args_list, desc="Optimizing"):
            result = optimize_single_trajectory(args)
            all_sequences.append(result)
            if not verbose:
                tqdm.write(f"  {Path(args[0]).name}: cost={result['final_cost']:.2f} (PID baseline={result['pid_baseline_cost']:.2f})")
    
    all_sequences.sort(key=lambda x: x['file'])
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_sequences, f)
    
    avg_cost = np.mean([s['final_cost'] for s in all_sequences])
    avg_pid = np.mean([s['pid_baseline_cost'] for s in all_sequences])
    print(f"\nSaved {len(all_sequences)} sequences to {output_path}")
    print(f"Average PID baseline cost: {avg_pid:.2f}")
    print(f"Average optimized cost: {avg_cost:.2f}")
    print(f"Improvement: {(avg_pid - avg_cost) / avg_pid * 100:.1f}%")
    
    return all_sequences


# ============ RNN Model and Training ============

class EncoderDecoderRNN(nn.Module):
    """Encoder-Decoder RNN for control prediction."""
    
    def __init__(self, state_dim=5, action_dim=1, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.state_proj = nn.Linear(state_dim, hidden_dim // 2)
        self.action_proj = nn.Linear(action_dim, hidden_dim // 2)
        
        self.encoder = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder_input = nn.Linear(hidden_dim + 1, hidden_dim)
        
        self.decoder = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
    
    def forward(self, past_states, past_actions, target_accel, hidden=None):
        state_emb = self.state_proj(past_states)
        action_emb = self.action_proj(past_actions)
        encoder_input = torch.cat([state_emb, action_emb], dim=-1)
        
        encoder_output, encoder_hidden = self.encoder(encoder_input, hidden)
        context = encoder_output[:, -1:, :]
        
        target_accel = target_accel.unsqueeze(1)
        decoder_in = torch.cat([context, target_accel], dim=-1)
        decoder_in = self.decoder_input(decoder_in)
        
        decoder_output, decoder_hidden = self.decoder(decoder_in, encoder_hidden)
        action = self.output_proj(decoder_output.squeeze(1)) * 2.0
        
        return action, decoder_hidden


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_length=CONTEXT_LENGTH):
        self.samples = []
        
        for seq in sequences:
            targets = np.array(seq['targets'])
            states = np.array(seq['states'])
            optimal_actions = np.array(seq['optimal_actions'])
            
            for i in range(CONTROL_START_IDX, min(COST_END_IDX - 1, len(targets) - 1)):
                if i < seq_length:
                    continue
                
                past_states = np.array([[
                    targets[j], targets[j], states[j, 0],
                    states[j, 1] / 30.0, states[j, 2] / 5.0
                ] for j in range(i - seq_length, i)], dtype=np.float32)
                
                past_actions = optimal_actions[i - seq_length:i].reshape(-1, 1).astype(np.float32)
                target_accel = np.array([targets[i]], dtype=np.float32)
                next_action = np.array([optimal_actions[i]], dtype=np.float32)
                
                if not (np.isnan(past_states).any() or np.isnan(next_action).any()):
                    self.samples.append({
                        'past_states': past_states,
                        'past_actions': past_actions,
                        'target_accel': target_accel,
                        'next_action': next_action,
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s['past_states']),
            torch.from_numpy(s['past_actions']),
            torch.from_numpy(s['target_accel']),
            torch.from_numpy(s['next_action']),
        )


def train_model(sequences, output_path, epochs=100, batch_size=256, lr=1e-3, device='cpu'):
    print("\n=== Training Encoder-Decoder RNN ===")
    
    dataset = SequenceDataset(sequences)
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: No training samples!")
        return None
    
    n_train = int(0.9 * len(dataset))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(n_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(n_train, len(dataset))),
        batch_size=batch_size
    )
    
    model = EncoderDecoderRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            past_states, past_actions, target_accel, next_action = [b.to(device) for b in batch]
            pred_action, _ = model(past_states, past_actions, target_accel)
            loss = F.mse_loss(pred_action, next_action)
            
            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                past_states, past_actions, target_accel, next_action = [b.to(device) for b in batch]
                pred_action, _ = model(past_states, past_actions, target_accel)
                loss = F.mse_loss(pred_action, next_action)
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
        
        avg_train = np.mean(train_losses) if train_losses else float('nan')
        avg_val = np.mean(val_losses) if val_losses else float('nan')
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: Train={avg_train:.6f}, Val={avg_val:.6f}")
        
        if avg_val < best_val_loss and not np.isnan(avg_val):
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_path)
    
    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved to {output_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--num_files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_optimization", action='store_true')
    parser.add_argument("--optimizer", type=str, default='coord', 
                       choices=['cmaes', 'coord', 'random'],
                       help="Optimizer method: cmaes (CMA-ES), coord (coordinate descent), random")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA for ONNX inference")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    sequences_path = os.path.join(args.output_dir, "optimal_sequences.pkl")
    model_out_path = os.path.join(args.output_dir, "highs_rnn_controller.pth")
    
    if args.skip_optimization and os.path.exists(sequences_path):
        print(f"Loading sequences from {sequences_path}")
        with open(sequences_path, 'rb') as f:
            sequences = pickle.load(f)
    else:
        sequences = generate_optimal_sequences(
            args.model_path, args.data_dir, sequences_path,
            num_files=args.num_files, num_workers=args.num_workers,
            optimizer_method=args.optimizer, use_cuda=not args.no_cuda
        )
    
    train_model(
        sequences, model_out_path,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
