"""
Visualize and Compare Controller Solutions

This utility graphs:
1. The HiGHS-optimized solution
2. The simple controller solution  
3. The target lateral acceleration from data

And displays the total cost for each solution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import onnxruntime as ort
from pathlib import Path
from collections import namedtuple

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


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


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


def get_onnx_model(model_path):
    """Load the ONNX physics model."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    with open(model_path, "rb") as f:
        session = ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])
    return session


def predict_with_onnx(session, tokenizer, states, actions, past_preds):
    """Run the ONNX physics model deterministically."""
    tokenized_actions = tokenizer.encode(past_preds)
    states_np = np.column_stack([actions, [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states]])
    input_data = {
        'states': np.expand_dims(states_np, axis=0).astype(np.float32),
        'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    res = session.run(None, input_data)[0]
    token = np.argmax(res[0, -1])
    return tokenizer.decode(token)


def compute_cost(target, pred):
    """Compute the total cost for a trajectory."""
    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk = np.diff(pred) / DEL_T
    jerk_cost = np.mean(jerk**2) * 100
    total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    return {
        'lataccel_cost': lat_accel_cost,
        'jerk_cost': jerk_cost,
        'total_cost': total_cost
    }


def simulate_with_actions(session, tokenizer, data, actions):
    """Simulate the physics with given actions and return the resulting lataccel trajectory."""
    states_data = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    targets = data['target_lataccel'].values
    
    # Initialize history
    state_history = states_data[:CONTEXT_LENGTH].copy()
    action_history = actions[:CONTEXT_LENGTH].tolist()
    lataccel_history = targets[:CONTEXT_LENGTH].tolist()
    current_lataccel = lataccel_history[-1]
    
    # Run up to control start (use ground truth)
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states_data[step])
        action_history.append(actions[step])
        
        pred = predict_with_onnx(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = targets[step]  # Use ground truth before control
        lataccel_history.append(current_lataccel)
    
    # Run control period
    predicted_lataccels = []
    for i, step in enumerate(range(CONTROL_START_IDX, min(COST_END_IDX, len(data)))):
        state_history.append(states_data[step])
        action_idx = step if step < len(actions) else -1
        action_history.append(actions[action_idx])
        
        pred = predict_with_onnx(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = pred
        lataccel_history.append(current_lataccel)
        predicted_lataccels.append(pred)
    
    return np.array(predicted_lataccels), np.array(lataccel_history)


class SimpleController:
    """Simple PID + Feedforward controller."""
    
    def __init__(self):
        self.p = 0.17
        self.i = 0.1
        self.d = -0.06
        self.kff = 0.13
        self.error_integral = 0.0
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        fb = self.p * error + self.i * self.error_integral + self.d * error_diff
        ff = self.kff * target_lataccel
        
        return np.clip(fb + ff, STEER_RANGE[0], STEER_RANGE[1])
    
    def reset(self):
        self.error_integral = 0.0
        self.prev_error = 0.0


def simulate_with_controller(session, tokenizer, data, controller):
    """Simulate using a controller to generate actions."""
    states_data = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    targets = data['target_lataccel'].values
    initial_actions = data['steer_command'].values
    
    controller.reset()
    
    # Initialize history
    state_history = states_data[:CONTEXT_LENGTH].copy()
    action_history = initial_actions[:CONTEXT_LENGTH].tolist()
    lataccel_history = targets[:CONTEXT_LENGTH].tolist()
    current_lataccel = lataccel_history[-1]
    
    generated_actions = initial_actions[:CONTEXT_LENGTH].tolist()
    
    # Run up to control start (use initial actions)
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states_data[step])
        action_history.append(initial_actions[step])
        generated_actions.append(initial_actions[step])
        
        pred = predict_with_onnx(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = targets[step]
        lataccel_history.append(current_lataccel)
    
    # Run control period with controller
    predicted_lataccels = []
    for step in range(CONTROL_START_IDX, min(COST_END_IDX, len(data))):
        state_history.append(states_data[step])
        
        # Generate action using controller
        action = controller.update(targets[step], current_lataccel)
        action_history.append(action)
        generated_actions.append(action)
        
        pred = predict_with_onnx(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = pred
        lataccel_history.append(current_lataccel)
        predicted_lataccels.append(pred)
    
    return np.array(predicted_lataccels), np.array(lataccel_history), np.array(generated_actions)


def visualize_comparison(data_path, model_path, sequences_path, file_index=0):
    """Visualize comparison between optimized and simple controller solutions."""
    
    # Load ONNX model
    print(f"Loading ONNX model from {model_path}...")
    session = get_onnx_model(model_path)
    tokenizer = LataccelTokenizer()
    
    # Load optimized sequences
    print(f"Loading optimized sequences from {sequences_path}...")
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)
    
    if file_index >= len(sequences):
        print(f"Error: file_index {file_index} >= number of sequences {len(sequences)}")
        return
    
    seq = sequences[file_index]
    data_file = seq['file']
    print(f"Using data file: {data_file}")
    
    # Load data
    data = load_data(data_file)
    targets = data['target_lataccel'].values
    
    # Get optimal actions - this is now the FULL array
    optimal_actions = np.array(seq['optimal_actions'])
    print(f"Optimal actions shape: {optimal_actions.shape}, expected: {len(data)}")
    
    # If optimal_actions is still just the control slice, build full array
    if len(optimal_actions) == COST_END_IDX - CONTROL_START_IDX:
        print("  (Converting slice to full array)")
        initial_actions = np.array(seq['initial_actions'])
        full_optimal_actions = initial_actions.copy()
        full_optimal_actions[CONTROL_START_IDX:COST_END_IDX] = optimal_actions
        optimal_actions = full_optimal_actions
    
    # Simulate with optimized actions
    print("Simulating with optimized actions...")
    opt_lataccels, opt_full_history = simulate_with_actions(session, tokenizer, data, optimal_actions)
    
    # Simulate with simple controller
    print("Simulating with simple controller...")
    simple_controller = SimpleController()
    simple_lataccels, simple_full_history, simple_actions = simulate_with_controller(
        session, tokenizer, data, simple_controller
    )
    
    # Compute costs
    target_slice = targets[CONTROL_START_IDX:CONTROL_START_IDX + len(opt_lataccels)]
    opt_cost = compute_cost(target_slice, opt_lataccels)
    simple_cost = compute_cost(target_slice[:len(simple_lataccels)], simple_lataccels)
    
    print(f"\n=== Cost Comparison ===")
    print(f"Optimized Solution:")
    print(f"  Lataccel Cost: {opt_cost['lataccel_cost']:.4f}")
    print(f"  Jerk Cost:     {opt_cost['jerk_cost']:.4f}")
    print(f"  Total Cost:    {opt_cost['total_cost']:.4f}")
    print(f"\nSimple Controller:")
    print(f"  Lataccel Cost: {simple_cost['lataccel_cost']:.4f}")
    print(f"  Jerk Cost:     {simple_cost['jerk_cost']:.4f}")
    print(f"  Total Cost:    {simple_cost['total_cost']:.4f}")
    
    improvement = (simple_cost['total_cost'] - opt_cost['total_cost']) / simple_cost['total_cost'] * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    time_steps = np.arange(len(targets))
    control_steps = np.arange(CONTROL_START_IDX, CONTROL_START_IDX + len(opt_lataccels))
    
    # Plot 1: Lateral Acceleration Comparison
    ax1 = axes[0]
    ax1.plot(time_steps, targets, 'b-', label='Target', alpha=0.8, linewidth=2)
    ax1.plot(control_steps, opt_lataccels, 'g-', label=f'Optimized (cost={opt_cost["total_cost"]:.1f})', linewidth=2)
    ax1.plot(control_steps[:len(simple_lataccels)], simple_lataccels, 'r-', 
             label=f'Simple Controller (cost={simple_cost["total_cost"]:.1f})', linewidth=2)
    ax1.axvline(x=CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax1.axvline(x=COST_END_IDX, color='gray', linestyle='--', alpha=0.5, label='Cost End')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Lateral Acceleration')
    ax1.set_title('Lateral Acceleration Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tracking Error
    ax2 = axes[1]
    opt_error = target_slice - opt_lataccels
    simple_error = target_slice[:len(simple_lataccels)] - simple_lataccels
    ax2.plot(control_steps, opt_error, 'g-', label='Optimized Error', linewidth=1.5)
    ax2.plot(control_steps[:len(simple_lataccels)], simple_error, 'r-', label='Simple Controller Error', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Tracking Error')
    ax2.set_title('Tracking Error (Target - Actual)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Control Actions
    ax3 = axes[2]
    ax3.plot(time_steps[:len(optimal_actions)], optimal_actions, 'g-', label='Optimized Actions', linewidth=1.5)
    ax3.plot(time_steps[:len(simple_actions)], simple_actions, 'r-', label='Simple Controller Actions', linewidth=1.5)
    ax3.axvline(x=CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(x=COST_END_IDX, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Steer Command')
    ax3.set_title('Control Actions')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add overall title with cost summary
    fig.suptitle(f'Solution Comparison - File {file_index}\n'
                 f'Optimized: {opt_cost["total_cost"]:.1f} | Simple: {simple_cost["total_cost"]:.1f} | '
                 f'Improvement: {improvement:.1f}%', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.savefig(f'solution_comparison_{file_index}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to solution_comparison_{file_index}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare controller solutions")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx",
                        help="Path to tinyphysics ONNX model")
    parser.add_argument("--sequences_path", type=str, default="./models/optimal_sequences.pkl",
                        help="Path to optimized sequences pickle file")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--file_index", type=int, default=0,
                        help="Index of the file to visualize (0-based)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: ONNX model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.sequences_path):
        print(f"Error: Optimized sequences not found at {args.sequences_path}")
        print("Run train_highs_rnn.py first to generate optimal sequences.")
        return
    
    visualize_comparison(
        args.data_dir,
        args.model_path,
        args.sequences_path,
        args.file_index
    )


if __name__ == "__main__":
    main()
