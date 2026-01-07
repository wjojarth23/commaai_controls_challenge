"""
Trajectory Optimization using HiGHS and L-BFGS

This script:
1. Uses HiGHS to optimize a lateral acceleration curve by balancing
   acceleration magnitude and jerk (rate of change of acceleration)
2. Uses L-BFGS to solve for steering inputs that match the optimized curve
3. Visualizes the input and output curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import linprog
import warnings

warnings.filterwarnings('ignore')


class TrajectoryOptimizer:
    """
    Two-stage trajectory optimizer:
    1. Optimize lateral acceleration curve using linear programming (HiGHS)
    2. Optimize steering inputs using L-BFGS
    """
    
    def __init__(self, data_file='data/00000.csv', n_points=100):
        """
        Initialize the optimizer with data.
        
        Args:
            data_file: Path to CSV file with trajectory data
            n_points: Number of points to use for optimization
        """
        self.n_points = n_points
        self.dt = 0.1  # Time step (10 Hz)
        
        # Load data
        self.data = pd.read_csv(data_file)
        
        # Extract target trajectory (first n_points)
        self.target_lataccel = self.data['targetLateralAcceleration'].values[:n_points]
        self.v_ego = self.data['vEgo'].values[:n_points]
        self.t = self.data['t'].values[:n_points]
        
        print(f"Loaded {len(self.target_lataccel)} data points from {data_file}")
        
    def optimize_lataccel_curve_highs(self):
        """
        Stage 1: Optimize lateral acceleration curve using HiGHS (via scipy's linprog).
        
        Uses the actual cost function from the comma.ai controls challenge:
        - lataccel_cost = sum((actual - target)^2) / steps * 100
        - jerk_cost = sum(((a[t] - a[t-1]) / dt)^2) / (steps - 1) * 100
        - total_cost = (lataccel_cost * 50) + jerk_cost
        
        This is formulated as a quadratic program solved via linearization.
        
        Returns:
            optimized_lataccel: Optimized lateral acceleration curve
        """
        print(f"\n=== Stage 1: Optimizing Lateral Acceleration Curve (HiGHS) ===")
        print(f"Using comma.ai cost function: (lataccel_cost * 50) + jerk_cost")
        
        n = self.n_points
        
        # Decision variables: [a_0, a_1, ..., a_{n-1}, error_sq_0, ..., error_sq_{n-1}, jerk_sq_0, ..., jerk_sq_{n-2}]
        # Variables: [a (n), error_squared (n), jerk_squared (n-1)]
        n_vars = n + n + (n - 1)
        
        # Objective function from README:
        # total_cost = (lataccel_cost * 50) + jerk_cost
        # where lataccel_cost = sum(error^2) / n * 100
        #       jerk_cost = sum(jerk^2) / (n-1) * 100
        # This becomes: (50 * 100 / n) * sum(error^2) + (100 / (n-1)) * sum(jerk^2)
        c = np.zeros(n_vars)
        # Acceleration variables don't directly contribute (only through error and jerk)
        c[:n] = 0
        # Error squared terms: weighted by (50 * 100 / n)
        c[n:2*n] = (50 * 100) / n
        # Jerk squared terms: weighted by (100 / (n-1))
        c[2*n:] = 100 / (n - 1)
        
        # Inequality constraints: A_ub @ x <= b_ub
        # Constraints:
        # 1. error_sq[i] >= (a[i] - target[i])^2  (linearized using first-order approximation)
        # 2. jerk_sq[i] >= ((a[i+1] - a[i])/dt)^2  (linearized)
        # 3. Acceleration bounds: -5 <= a[i] <= 5
        
        # For linearization, we approximate around the target trajectory:
        # error_sq[i] >= 2*target[i]*(a[i] - target[i]) + target[i]^2
        # which simplifies to: error_sq[i] >= 2*target[i]*a[i] - target[i]^2
        # Rearranged: -2*target[i]*a[i] + error_sq[i] >= -target[i]^2
        
        A_ub = []
        b_ub = []
        
        # Error squared constraints (linearized around target)
        # We use: error^2 ≈ (a - target)^2
        # For LP, we approximate: error_sq[i] >= |a[i] - target[i]| via two constraints
        for i in range(n):
            # error_sq[i] >= (a[i] - target[i])  =>  -a[i] + error_sq[i] >= -target[i]
            row = np.zeros(n_vars)
            row[i] = -1.0               # -a[i]
            row[n + i] = 1.0            # error_sq[i]
            A_ub.append(row)
            b_ub.append(-self.target_lataccel[i])
            
            # error_sq[i] >= -(a[i] - target[i])  =>  a[i] + error_sq[i] >= target[i]
            row = np.zeros(n_vars)
            row[i] = 1.0                # a[i]
            row[n + i] = 1.0            # error_sq[i]
            A_ub.append(row)
            b_ub.append(self.target_lataccel[i])
        
        # Jerk squared constraints (linearized)
        # jerk = (a[i+1] - a[i]) / dt
        # We approximate: jerk_sq[i] >= |jerk[i]| via two constraints
        for i in range(n - 1):
            # jerk_sq[i] >= (a[i+1] - a[i])/dt
            row = np.zeros(n_vars)
            row[i] = -1.0 / self.dt           # -a[i]
            row[i + 1] = 1.0 / self.dt        # a[i+1]
            row[2*n + i] = -1.0               # -jerk_sq[i]
            A_ub.append(row)
            b_ub.append(0)
            
            # jerk_sq[i] >= -(a[i+1] - a[i])/dt
            row = np.zeros(n_vars)
            row[i] = 1.0 / self.dt            # a[i]
            row[i + 1] = -1.0 / self.dt       # -a[i+1]
            row[2*n + i] = -1.0               # -jerk_sq[i]
            A_ub.append(row)
            b_ub.append(0)
        
        # Acceleration upper bounds: a[i] <= 5
        for i in range(n):
            row = np.zeros(n_vars)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(5.0)
            
        # Acceleration lower bounds: -a[i] <= 5 (i.e., a[i] >= -5)
        for i in range(n):
            row = np.zeros(n_vars)
            row[i] = -1.0
            A_ub.append(row)
            b_ub.append(5.0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds on variables
        # a: acceleration bounds [-5, 5]
        # error_sq: must be non-negative
        # jerk_sq: must be non-negative
        bounds = (
            [(-5, 5) for _ in range(n)] +           # acceleration
            [(0, None) for _ in range(n)] +          # error_squared
            [(0, None) for _ in range(n - 1)]        # jerk_squared
        )
        
        # Solve using HiGHS method via linprog
        print("Solving linear program with HiGHS...")
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                        method='highs', options={'disp': False})
        
        if result.success:
            print(f"✓ Optimization successful!")
            optimized_lataccel = result.x[:n]
            error_sq = result.x[n:2*n]
            jerk_sq = result.x[2*n:]
            
            # Calculate actual costs using the formula from README
            errors = optimized_lataccel - self.target_lataccel
            lataccel_cost = np.sum(errors**2) / n * 100
            
            jerks = np.diff(optimized_lataccel) / self.dt
            jerk_cost = np.sum(jerks**2) / (n - 1) * 100
            
            total_cost = (lataccel_cost * 50) + jerk_cost
            
            print(f"  Total cost: {total_cost:.4f}")
            print(f"    Lataccel cost: {lataccel_cost:.4f} (weight: 50x)")
            print(f"    Jerk cost: {jerk_cost:.4f}")
            print(f"  RMS error: {np.sqrt(np.mean(errors**2)):.4f} m/s²")
            print(f"  RMS jerk: {np.sqrt(np.mean(jerks**2)):.4f} m/s³")
        else:
            print(f"✗ Optimization failed: {result.message}")
            optimized_lataccel = self.target_lataccel
            
        return optimized_lataccel
    
    def optimize_steering_lbfgs(self, target_lataccel, initial_steer=None):
        """
        Stage 2: Optimize steering inputs using L-BFGS to match target lateral acceleration.
        
        Uses a simplified vehicle model:
        lataccel ≈ k * v^2 * steer
        
        where k is a gain factor related to vehicle geometry.
        
        Args:
            target_lataccel: Target lateral acceleration curve to match
            initial_steer: Initial guess for steering inputs
            
        Returns:
            optimized_steer: Optimized steering command curve
        """
        print(f"\n=== Stage 2: Optimizing Steering Inputs (L-BFGS) ===")
        
        n = len(target_lataccel)
        
        # Initial guess
        if initial_steer is None:
            # Simple feedforward guess based on vehicle model
            # lataccel ≈ k * v^2 * steer => steer ≈ lataccel / (k * v^2)
            k_vehicle = 0.05  # Approximate gain
            initial_steer = target_lataccel / (k_vehicle * self.v_ego**2 + 1e-6)
            initial_steer = np.clip(initial_steer, -2.0, 2.0)
        
        # Define objective function
        def objective(steer):
            """
            Minimize tracking error and steering smoothness.
            """
            # Simple vehicle model
            k_vehicle = 0.05
            predicted_lataccel = k_vehicle * self.v_ego**2 * steer
            
            # Tracking error
            tracking_error = np.sum((predicted_lataccel - target_lataccel)**2)
            
            # Steering smoothness (penalize large changes)
            steer_rate = np.diff(steer) / self.dt
            smoothness_penalty = 0.1 * np.sum(steer_rate**2)
            
            # Steering magnitude penalty (prefer smaller steering)
            magnitude_penalty = 0.01 * np.sum(steer**2)
            
            return tracking_error + smoothness_penalty + magnitude_penalty
        
        # Gradient of objective
        def gradient(steer):
            """
            Compute gradient of objective function.
            """
            k_vehicle = 0.05
            predicted_lataccel = k_vehicle * self.v_ego**2 * steer
            
            # Gradient of tracking error
            grad_tracking = 2 * k_vehicle * self.v_ego**2 * (predicted_lataccel - target_lataccel)
            
            # Gradient of smoothness penalty
            grad_smoothness = np.zeros(n)
            steer_rate = np.diff(steer) / self.dt
            for i in range(n - 1):
                grad_smoothness[i] += 0.1 * 2 * steer_rate[i] / self.dt
                grad_smoothness[i + 1] -= 0.1 * 2 * steer_rate[i] / self.dt
            
            # Gradient of magnitude penalty
            grad_magnitude = 0.01 * 2 * steer
            
            return grad_tracking + grad_smoothness + grad_magnitude
        
        # Bounds: steering limited to [-2, 2]
        bounds = [(-2.0, 2.0) for _ in range(n)]
        
        # Optimize using L-BFGS-B
        print("Solving with L-BFGS-B optimizer...")
        result = minimize(objective, initial_steer, method='L-BFGS-B',
                         jac=gradient, bounds=bounds,
                         options={'maxiter': 500, 'disp': False})
        
        if result.success:
            print(f"✓ Optimization successful!")
            print(f"  Objective value: {result.fun:.4f}")
            print(f"  Iterations: {result.nit}")
            optimized_steer = result.x
        else:
            print(f"✗ Optimization failed: {result.message}")
            optimized_steer = initial_steer
            
        # Compute resulting lateral acceleration
        k_vehicle = 0.05
        resulting_lataccel = k_vehicle * self.v_ego**2 * optimized_steer
        tracking_error = np.sqrt(np.mean((resulting_lataccel - target_lataccel)**2))
        print(f"  RMS tracking error: {tracking_error:.4f} m/s²")
        
        return optimized_steer, resulting_lataccel
    
    def visualize_results(self, optimized_lataccel, optimized_steer, resulting_lataccel):
        """
        Create visualization of the optimization results.
        
        Args:
            optimized_lataccel: Optimized lateral acceleration from Stage 1
            optimized_steer: Optimized steering commands from Stage 2
            resulting_lataccel: Resulting lateral acceleration from steering commands
        """
        print(f"\n=== Generating Visualization ===")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Lateral Acceleration Comparison
        ax1 = axes[0]
        ax1.plot(self.t, self.target_lataccel, 'b-', label='Target (Original)', linewidth=2, alpha=0.7)
        ax1.plot(self.t, optimized_lataccel, 'r--', label='Optimized (HiGHS)', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12)
        ax1.set_title('Stage 1: Lateral Acceleration Optimization (HiGHS)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Steering Commands
        ax2 = axes[1]
        ax2.plot(self.t, optimized_steer, 'g-', label='Optimized Steering (L-BFGS)', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Steering Command', fontsize=12)
        ax2.set_title('Stage 2: Steering Input Optimization (L-BFGS)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        # Plot 3: Final Lateral Acceleration Comparison
        ax3 = axes[2]
        ax3.plot(self.t, optimized_lataccel, 'r--', label='Target (HiGHS Output)', linewidth=2, alpha=0.7)
        ax3.plot(self.t, resulting_lataccel, 'purple', label='Achieved (from Steering)', linewidth=2)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12)
        ax3.set_title('Stage 2: Lateral Acceleration Tracking', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Compute and display metrics
        jerk_original = np.diff(self.target_lataccel) / self.dt
        jerk_optimized = np.diff(optimized_lataccel) / self.dt
        
        metrics_text = (
            f"Metrics:\n"
            f"Original - RMS ):
        """
        Run the complete two-stage optimization pipeline using the comma.ai cost function.
        """
        print("="*70)
        print("  TRAJECTORY OPTIMIZATION: HiGHS + L-BFGS")
        print("="*70)
        
        # Stage 1: Optimize lateral acceleration curve
        optimized_lataccel = self.optimize_lataccel_curve_highs("""
        Run the complete two-stage optimization pipeline.
        
        Args:
            jerk_weight: Weight for jerk penalty in Stage 1
            accel_weight: Weight for acceleration magnitude in Stage 1
        """
        print("="*70)
        print("  TRAJECTORY OPTIMIZATION: HiGHS + L-BFGS")
        print("="*70)
        
        # Stage 1: Optimize lateral acceleration curve
        optimized_lataccel = self.optimize_lataccel_curve_highs(
            jerk_weight=jerk_weight,
            accel_weight=accel_weight
        )
        
        # Stage 2: Optimize steering inputs
        optimized_steer, resulting_lataccel = self.optimize_steering_lbfgs(
            optimized_lataccel
        )
        
        # Visualize results
        self.visualize_results(optimized_lataccel, optimized_steer, resulting_lataccel)
        
        print("\n" + "="*70)
        print("  OPTIMIZATION COMPLETE")
        print("="*70)

 using comma.ai's actual cost function:
    # total_cost = (lataccel_cost * 50) + jerk_cost
    optimizer.run_optimization(
    """
    # Create optimizer with first data file
    optimizer = TrajectoryOptimizer(data_file='data/00000.csv', n_points=100)
    
    # Run optimization
    # jerk_weight: higher value = smoother acceleration curve
    # accel_weight: higher value = smaller accelerations
    optimizer.run_optimization(jerk_weight=2.0, accel_weight=0.5)


if __name__ == '__main__':
    main()
