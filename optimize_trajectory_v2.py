"""
Trajectory Optimization using Analytical Solution and L-BFGS

This script implements the comma.ai controls challenge cost function:
1. Uses analytical optimization to optimize a lateral acceleration curve:
   total_cost = (lataccel_cost * 50) + jerk_cost
   This is a quadratic problem with closed-form solution via linear system
2. Uses L-BFGS to solve for steering inputs that match the optimized curve
   using the actual ONNX physics model for simulation
3. Visualizes the input and output curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import namedtuple
import onnxruntime as ort
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Constants from tinyphysics
ACC_G = 9.81
CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
COST_END_IDX = 500
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
    """Tokenizer for lateral acceleration values."""
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[np.clip(token, 0, self.vocab_size - 1)]


class TrajectoryOptimizer:
    """
    Two-stage trajectory optimizer using comma.ai cost function:
    1. Optimize lateral acceleration curve using analytical solution
    2. Optimize steering inputs using L-BFGS with physics model
    """
    
    def __init__(self, data_file='data/00000.csv', model_path='models/tinyphysics.onnx', n_points=None):
        """
        Initialize the optimizer with data and physics model.
        
        Args:
            data_file: Path to CSV file with trajectory data
            model_path: Path to ONNX physics model
            n_points: Number of points to use (default: COST_END_IDX - CONTROL_START_IDX)
        """
        self.dt = DEL_T
        self.data_file = data_file
        
        # Load physics model - prefer CUDA if available
        print(f"Loading physics model from {model_path}...")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        
        # Check for CUDA
        providers = ['CPUExecutionProvider']
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.insert(0, 'CUDAExecutionProvider')
            print("  Using CUDA for physics model inference")
        else:
            print("  Using CPU for physics model inference")
        
        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, providers)
        self.tokenizer = LataccelTokenizer()
        
        # Load data
        df = pd.read_csv(data_file)
        self.full_data = pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values  # Sign convention
        })
        
        # Extract control period data
        if n_points is None:
            n_points = COST_END_IDX - CONTROL_START_IDX
        self.n_points = n_points
        
        self.target_lataccel = self.full_data['target_lataccel'].values[CONTROL_START_IDX:CONTROL_START_IDX + n_points]
        self.v_ego = self.full_data['v_ego'].values[CONTROL_START_IDX:CONTROL_START_IDX + n_points]
        self.t = np.arange(n_points) * DEL_T
        
        # Build states for physics model
        self.states = [State(
            roll_lataccel=self.full_data.iloc[i]['roll_lataccel'],
            v_ego=self.full_data.iloc[i]['v_ego'],
            a_ego=self.full_data.iloc[i]['a_ego']
        ) for i in range(len(self.full_data))]
        
        print(f"Loaded {n_points} control points from {data_file}")
        
    def optimize_lataccel_curve_analytical(self):
        """
        Stage 1: Optimize lateral acceleration curve using analytical solution.
        
        Uses the actual cost function from the comma.ai controls challenge README:
        - lataccel_cost = sum((actual - target)^2) / steps * 100
        - jerk_cost = sum(((a[t] - a[t-1]) / dt)^2) / (steps - 1) * 100
        - total_cost = (lataccel_cost * 50) + jerk_cost
        
        This is a quadratic optimization problem with analytical solution.
        We minimize: 50 * sum((x - target)^2) + (1/dt^2) * sum((x[i+1] - x[i])^2)
        
        Returns:
            optimized_lataccel: Optimized lateral acceleration curve
        """
        print(f"\n=== Stage 1: Optimizing Lateral Acceleration Curve (Analytical) ===")
        print(f"Cost function: (lataccel_cost * 50) + jerk_cost")
        
        n = self.n_points
        alpha = 50.0  # LAT_ACCEL_COST_MULTIPLIER
        beta = 1.0 / (self.dt ** 2)  # Jerk penalty coefficient
        
        # Build tridiagonal matrix for the linear system Ax = b
        # The solution comes from taking derivatives and setting to zero
        main_diag = np.full(n, alpha + 2*beta)
        main_diag[0] = alpha + beta  # First element only has one neighbor
        main_diag[-1] = alpha + beta  # Last element only has one neighbor
        
        off_diag = np.full(n-1, -beta)
        
        A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        b = alpha * self.target_lataccel
        
        # Solve the linear system
        print("Solving analytical optimization (tridiagonal system)...")
        optimized_lataccel = np.linalg.solve(A, b)
        
        # Calculate costs for original target
        target_errors = np.zeros(n)  # Target has 0 error by definition
        target_lataccel_cost = np.mean(target_errors**2) * 100
        target_jerks = np.diff(self.target_lataccel) / self.dt
        target_jerk_cost = np.mean(target_jerks**2) * 100
        target_total_cost = target_lataccel_cost * 50 + target_jerk_cost
        
        # Calculate costs for optimized curve
        errors = optimized_lataccel - self.target_lataccel
        lataccel_cost = np.mean(errors**2) * 100
        
        jerks = np.diff(optimized_lataccel) / self.dt
        jerk_cost = np.mean(jerks**2) * 100
        
        total_cost = (lataccel_cost * 50) + jerk_cost
        
        print(f"✓ Optimization successful!")
        print(f"  Target curve:    cost={target_total_cost:.4f} (lat={target_lataccel_cost:.4f}, jerk={target_jerk_cost:.4f})")
        print(f"  Optimized curve: cost={total_cost:.4f} (lat={lataccel_cost:.4f}, jerk={jerk_cost:.4f})")
        print(f"  Cost reduction: {target_total_cost - total_cost:.4f} ({(1 - total_cost/target_total_cost)*100:.2f}%)")
        print(f"  RMS tracking error: {np.sqrt(np.mean(errors**2)):.4f} m/s²")
        print(f"  RMS jerk: {np.sqrt(np.mean(jerks**2)):.4f} m/s³")
            
        return optimized_lataccel
    
    def predict_lataccel(self, states, actions, past_lataccels):
        """Run physics model to predict next lateral acceleration."""
        tokenized = self.tokenizer.encode(np.array(past_lataccels))
        states_np = np.column_stack([actions, [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states]])
        input_data = {
            'states': np.expand_dims(states_np, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized, axis=0).astype(np.int64)
        }
        res = self.ort_session.run(None, input_data)[0]
        token = np.argmax(res[0, -1])
        return self.tokenizer.decode(token)
    
    def simulate_trajectory(self, actions):
        """
        Simulate the full trajectory using the physics model.
        
        Args:
            actions: Array of steering commands for control period
            
        Returns:
            achieved_lataccel: Resulting lateral acceleration curve
        """
        # Initialize history from pre-control data
        state_history = list(self.states[:CONTEXT_LENGTH])
        action_history = list(self.full_data['steer_command'].values[:CONTEXT_LENGTH])
        lataccel_history = list(self.full_data['target_lataccel'].values[:CONTEXT_LENGTH])
        current_lataccel = lataccel_history[-1]
        
        # Run through pre-control period to build history
        for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state_history.append(self.states[step])
            action_history.append(self.full_data['steer_command'].values[step])
            current_lataccel = self.full_data['target_lataccel'].values[step]
            lataccel_history.append(current_lataccel)
        
        # Simulate control period
        achieved = []
        for i, step in enumerate(range(CONTROL_START_IDX, CONTROL_START_IDX + len(actions))):
            state_history.append(self.states[step])
            action = np.clip(actions[i], STEER_RANGE[0], STEER_RANGE[1])
            action_history.append(action)
            
            # Predict next lataccel using physics model
            pred = self.predict_lataccel(
                state_history[-CONTEXT_LENGTH:],
                action_history[-CONTEXT_LENGTH:],
                lataccel_history[-CONTEXT_LENGTH:]
            )
            pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            current_lataccel = pred
            lataccel_history.append(current_lataccel)
            achieved.append(current_lataccel)
        
        return np.array(achieved)
    
    def generate_pid_rollout(self, target_lataccel, gains=None):
        """
        Generate initial steering using closed-loop PID+FF rollout with the physics model.
        
        This is MUCH better than a simple feedforward because it uses actual
        closed-loop control with the physics model, building up proper integral/derivative terms.
        
        Args:
            target_lataccel: Target lateral acceleration curve
            gains: (kp, ki, kd, kff) tuple or None for defaults
            
        Returns:
            actions: Steering actions for control period
            achieved: Achieved lateral acceleration
        """
        if gains is None:
            kp, ki, kd, kff = 0.17, 0.10, -0.06, 0.13  # From simple controller
        else:
            kp, ki, kd, kff = gains
            
        n = len(target_lataccel)
        
        # Initialize
        error_integral = 0.0
        prev_error = 0.0
        
        state_history = list(self.states[:CONTEXT_LENGTH])
        action_history = list(self.full_data['steer_command'].values[:CONTEXT_LENGTH])
        lataccel_history = list(self.full_data['target_lataccel'].values[:CONTEXT_LENGTH])
        current_lataccel = lataccel_history[-1]
        
        # Pre-control period
        for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state_history.append(self.states[step])
            action_history.append(self.full_data['steer_command'].values[step])
            current_lataccel = self.full_data['target_lataccel'].values[step]
            lataccel_history.append(current_lataccel)
        
        # Closed-loop rollout
        actions = []
        achieved = []
        
        for k in range(n):
            step = CONTROL_START_IDX + k
            target = target_lataccel[k]
            
            # PID control
            error = target - current_lataccel
            error_integral += error
            error_diff = error - prev_error
            prev_error = error
            
            fb = kp * error + ki * error_integral + kd * error_diff
            ff = kff * target
            action = np.clip(fb + ff, STEER_RANGE[0], STEER_RANGE[1])
            actions.append(action)
            
            # Simulate step
            state_history.append(self.states[step])
            action_history.append(action)
            
            pred = self.predict_lataccel(
                state_history[-CONTEXT_LENGTH:],
                action_history[-CONTEXT_LENGTH:],
                lataccel_history[-CONTEXT_LENGTH:]
            )
            pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            current_lataccel = pred
            lataccel_history.append(current_lataccel)
            achieved.append(pred)
        
        return np.array(actions), np.array(achieved)
    
    def simulate_with_gains(self, actions, eps=0.05):
        """
        Simulate trajectory and estimate local gains d(lataccel)/d(action) via finite difference.
        
        Returns:
            achieved: Achieved lataccel curve
            gains: Per-step local gain estimates
        """
        # Initialize
        state_history = list(self.states[:CONTEXT_LENGTH])
        action_history = list(self.full_data['steer_command'].values[:CONTEXT_LENGTH])
        lataccel_history = list(self.full_data['target_lataccel'].values[:CONTEXT_LENGTH])
        current_lataccel = lataccel_history[-1]
        
        for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state_history.append(self.states[step])
            action_history.append(self.full_data['steer_command'].values[step])
            current_lataccel = self.full_data['target_lataccel'].values[step]
            lataccel_history.append(current_lataccel)
        
        achieved = []
        gains = []
        
        for i, step in enumerate(range(CONTROL_START_IDX, CONTROL_START_IDX + len(actions))):
            state_history.append(self.states[step])
            action = actions[i]
            action_history.append(action)
            
            sh = state_history[-CONTEXT_LENGTH:]
            ah = action_history[-CONTEXT_LENGTH:]
            lh = lataccel_history[-CONTEXT_LENGTH:]
            
            # Nominal prediction
            pred0 = self.predict_lataccel(sh, ah, lh)
            pred0 = np.clip(pred0, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            
            # Finite difference for local gain
            a0 = float(ah[-1])
            a_plus = float(np.clip(a0 + eps, STEER_RANGE[0], STEER_RANGE[1]))
            a_minus = float(np.clip(a0 - eps, STEER_RANGE[0], STEER_RANGE[1]))
            
            ah_plus = list(ah)
            ah_minus = list(ah)
            ah_plus[-1] = a_plus
            ah_minus[-1] = a_minus
            
            pred_plus = self.predict_lataccel(sh, ah_plus, lh)
            pred_minus = self.predict_lataccel(sh, ah_minus, lh)
            pred_plus = np.clip(pred_plus, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            pred_minus = np.clip(pred_minus, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            
            gain = (pred_plus - pred_minus) / (2.0 * eps)
            
            current_lataccel = pred0
            lataccel_history.append(current_lataccel)
            achieved.append(pred0)
            gains.append(gain)
        
        return np.array(achieved), np.array(gains)
    
    def optimize_steering_ilc(self, target_lataccel, n_iterations=20, step_scale=0.08, 
                               max_action_delta=0.05, gain_floor=0.02):
        """
        Stage 2: Optimize steering using Differential Evolution (global optimizer).
        
        The ONNX physics model is an autoregressive encoder-decoder with tokenized
        output, making the loss landscape highly non-smooth with many local minima.
        L-BFGS gets trapped at the PID solution because gradients appear zero there.
        
        Differential Evolution is a population-based global optimizer that:
        1. Doesn't rely on gradients (derivative-free)
        2. Maintains diverse population to explore globally  
        3. Can escape local minima that trap gradient methods
        
        We parameterize steering with spline control points to keep dimensionality 
        manageable (~20 vars instead of 400).
        
        Args:
            target_lataccel: Target lateral acceleration curve to match
            
        Returns:
            optimized_steer: Optimized steering command curve
            resulting_lataccel: Resulting lateral acceleration from steering
        """
        from scipy.interpolate import CubicSpline
        from scipy.optimize import differential_evolution
        
        print(f"\n=== Stage 2: Optimizing Steering (Differential Evolution) ===")
        print(f"  Method: Global optimizer with spline parameterization")
        
        n = len(target_lataccel)
        
        # Fewer control points for DE (it's expensive, keep dimensionality low)
        n_control = 20
        control_indices = np.linspace(0, n - 1, n_control).astype(int)
        t_control = control_indices.astype(float)
        t_full = np.arange(n).astype(float)
        
        print(f"  Control points: {n_control} (optimization is {n_control}D)")
        
        # Step 1: Initialize with PID rollout for comparison
        print("\nStep 1: Generating PID rollout for baseline...")
        pid_actions, pid_achieved = self.generate_pid_rollout(target_lataccel)
        
        # Extract control points from PID solution
        pid_control = pid_actions[control_indices]
        
        # Compute initial cost
        lataccel_cost = np.mean((pid_achieved - target_lataccel)**2) * 100
        jerks = np.diff(pid_achieved) / self.dt
        jerk_cost = np.mean(jerks**2) * 100
        initial_cost = lataccel_cost * 50 + jerk_cost
        print(f"  PID baseline cost: {initial_cost:.2f}")
        
        # Step 2: Define objective for spline control points
        eval_count = [0]
        best_result = {'cost': initial_cost, 'actions': pid_actions.copy(), 'achieved': pid_achieved.copy()}
        
        def control_to_actions(control_points):
            """Convert control points to full action sequence via cubic spline."""
            control_points = np.clip(control_points, STEER_RANGE[0], STEER_RANGE[1])
            spline = CubicSpline(t_control, control_points, bc_type='clamped')
            actions = spline(t_full)
            actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
            return actions
        
        def objective(control_points):
            eval_count[0] += 1
            
            # Convert to full action sequence
            actions = control_to_actions(control_points)
            
            # Simulate with physics model
            achieved = self.simulate_trajectory(actions)
            
            # Compute comma.ai cost
            lataccel_cost = np.mean((achieved - target_lataccel)**2) * 100
            jerks = np.diff(achieved) / self.dt
            jerk_cost = np.mean(jerks**2) * 100
            cost = lataccel_cost * 50 + jerk_cost
            
            # Track best
            if cost < best_result['cost']:
                best_result['cost'] = cost
                best_result['actions'] = actions.copy()
                best_result['achieved'] = achieved.copy()
            
            return cost
        
        # Step 3: Set up Differential Evolution
        print("\nStep 2: Differential Evolution optimization...")
        print("  Population: 30, Max iterations: 50")
        
        # Define bounds centered around PID solution with some exploration room
        # Allow deviation of ±0.3 from PID to explore nearby solutions
        bounds = []
        for pid_val in pid_control:
            lo = max(STEER_RANGE[0], pid_val - 0.3)
            hi = min(STEER_RANGE[1], pid_val + 0.3)
            bounds.append((lo, hi))
        
        # Progress tracking  
        pbar = tqdm(total=50, desc='DE Iterations')
        last_iter = [0]
        
        def callback(xk, convergence):
            pbar.update(1)
            pbar.set_postfix({'evals': eval_count[0], 'best': f'{best_result["cost"]:.2f}'})
            return False  # Don't stop early
        
        # Run Differential Evolution
        # Use 'best1bin' strategy which is good for continuous optimization
        # seed with PID solution to help convergence
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=50,
            popsize=30,  # 30 * n_control = 600 population members
            mutation=(0.5, 1.0),  # Mutation factor range
            recombination=0.7,   # Crossover probability
            seed=42,
            callback=callback,
            polish=False,  # Don't use L-BFGS at end (it gets stuck)
            init='latinhypercube',  # Good initial coverage
            workers=1,  # Single worker (ONNX session isn't thread-safe)
            updating='deferred',
            disp=False
        )
        pbar.close()
        
        print(f"\n  Function evaluations: {eval_count[0]}")
        print(f"  DE converged: {result.success}")
        print(f"  Final cost from DE: {result.fun:.2f}")
        
        # Use best result found during optimization
        optimized_steer = best_result['actions']
        resulting_lataccel = best_result['achieved']
        
        # Compute final metrics
        tracking_error = np.sqrt(np.mean((resulting_lataccel - target_lataccel)**2))
        improvement = (1 - best_result['cost']/initial_cost) * 100 if initial_cost > 0 else 0
        
        print(f"\n✓ Spline L-BFGS optimization complete!")
        print(f"  Final RMS tracking error: {tracking_error:.4f} m/s²")
        print(f"  Improvement from PID: {initial_cost:.2f} → {best_result['cost']:.2f} ({improvement:.1f}%)")
        
        # Compute comma.ai cost
        lataccel_cost = np.sum((resulting_lataccel - target_lataccel)**2) / n * 100
        jerks = np.diff(resulting_lataccel) / self.dt
        jerk_cost = np.sum(jerks**2) / max(1, n - 1) * 100
        total_cost = lataccel_cost * 50 + jerk_cost
        
        print(f"\nAchieved trajectory costs (comma.ai):")
        print(f"  Lataccel cost: {lataccel_cost:.6f}")
        print(f"  Jerk cost:     {jerk_cost:.6f}")
        print(f"  Total cost:    {total_cost:.6f}  (lat*50 + jerk)")
        
        return optimized_steer, resulting_lataccel
    
    def visualize_results(self, optimized_lataccel, optimized_steer, resulting_lataccel):
        """
        Create visualization of the optimization results.
        """
        print(f"\n=== Generating Visualization ===")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Lateral Acceleration Comparison
        ax1 = axes[0]
        ax1.plot(self.t, self.target_lataccel, 'b-', label='Target (Original)', linewidth=2, alpha=0.7)
        ax1.plot(self.t, optimized_lataccel, 'r--', label='Optimized (Analytical)', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12)
        ax1.set_title('Stage 1: Lateral Acceleration Optimization (Analytical)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Steering Commands
        ax2 = axes[1]
        ax2.plot(self.t, optimized_steer, 'g-', label='Optimized Steering (Differential Evolution)', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Steering Command', fontsize=12)
        ax2.set_title('Stage 2: Steering Input Optimization (Differential Evolution)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        # Plot 3: Final Lateral Acceleration Comparison
        ax3 = axes[2]
        ax3.plot(self.t, optimized_lataccel, 'r--', label='Target (Analytical Output)', linewidth=2, alpha=0.7)
        ax3.plot(self.t, resulting_lataccel, 'purple', label='Achieved (from Steering)', linewidth=2)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12)
        ax3.set_title('Stage 2: Lateral Acceleration Tracking (DE)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Compute and display metrics
        jerk_original = np.diff(self.target_lataccel) / self.dt
        jerk_optimized = np.diff(optimized_lataccel) / self.dt
        
        # Calculate costs using comma.ai formula
        errors_orig = np.zeros(len(self.target_lataccel))  # Target has 0 error by definition
        lataccel_cost_orig = 0.0
        jerk_cost_orig = np.sum(jerk_original**2) / (len(jerk_original)) * 100
        total_cost_orig = (lataccel_cost_orig * 50) + jerk_cost_orig
        
        errors_opt = optimized_lataccel - self.target_lataccel
        lataccel_cost_opt = np.sum(errors_opt**2) / len(errors_opt) * 100
        jerk_cost_opt = np.sum(jerk_optimized**2) / len(jerk_optimized) * 100
        total_cost_opt = (lataccel_cost_opt * 50) + jerk_cost_opt
        
        metrics_text = (
            f"Comma.ai Cost Function Results:\n"
            f"Original  - Total: {total_cost_orig:.2f}, Lataccel: {lataccel_cost_orig:.2f}, Jerk: {jerk_cost_orig:.2f}\n"
            f"Optimized - Total: {total_cost_opt:.2f}, Lataccel: {lataccel_cost_opt:.2f}, Jerk: {jerk_cost_opt:.2f}\n"
            f"Tracking RMSE: {np.sqrt(np.mean((resulting_lataccel - optimized_lataccel)**2)):.4f} m/s²"
        )
        
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig('trajectory_optimization_results.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to 'trajectory_optimization_results.png'")
        plt.show()
        
    def run_optimization(self):
        """
        Run the complete two-stage optimization pipeline.
        """
        print("="*70)
        print("  TRAJECTORY OPTIMIZATION: Analytical + ILC")
        print("  Using comma.ai Cost Function")
        print("="*70)
        
        # Stage 1: Optimize lateral acceleration curve
        optimized_lataccel = self.optimize_lataccel_curve_analytical()
        
        # Stage 2: Optimize steering inputs using ILC (fast!)
        optimized_steer, resulting_lataccel = self.optimize_steering_ilc(
            optimized_lataccel
        )
        
        # Visualize results
        self.visualize_results(optimized_lataccel, optimized_steer, resulting_lataccel)
        
        print("\n" + "="*70)
        print("  OPTIMIZATION COMPLETE")
        print("="*70)


def main():
    """
    Main function to run the trajectory optimization.
    """
    # Create optimizer with data file and physics model
    optimizer = TrajectoryOptimizer(
        data_file='data/00000.csv',
        model_path='models/tinyphysics.onnx',
        n_points=None  # Use full control period (400 points)
    )
    
    # Run optimization using comma.ai's actual cost function:
    # total_cost = (lataccel_cost * 50) + jerk_cost
    # Stage 1: Analytical curve optimization
    # Stage 2: ILC steering optimization with physics model (FAST!)
    optimizer.run_optimization()


if __name__ == '__main__':
    main()
