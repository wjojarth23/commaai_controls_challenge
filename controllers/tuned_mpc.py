from . import BaseController
import numpy as np
import pandas as pd
import os

class Controller(BaseController):
  def __init__(self):
    self.model_params = self.fit_model()
    self.H = 20 # Horizon (2 seconds)
    self.N = 50 # Number of samples
    self.sigma = 0.5 # Noise std dev

  def fit_model(self):
    # Load data
    # We use a few files to fit the model
    data_dir = './data'
    if not os.path.exists(data_dir):
        # Fallback if data dir not found (e.g. in test env), though it should exist
        return np.array([0.8, 0.0, 0.0, 0.0])

    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:20] 
    X_list = []
    y_list = []
    
    for f in data_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            # Process like tinyphysics.py
            roll_lataccel = np.sin(df['roll'].values) * 9.81
            v_ego = df['vEgo'].values
            target_lataccel = df['targetLateralAcceleration'].values
            steer_command = -df['steerCommand'].values 
            
            # y_next = a*y_curr + b*u*v^2 + c*roll + d
            y_curr = target_lataccel[:-1]
            y_next = target_lataccel[1:]
            u_curr = steer_command[:-1]
            v_curr = v_ego[:-1]
            roll_curr = roll_lataccel[:-1]
            
            X = np.column_stack([
                y_curr,
                u_curr * (v_curr**2),
                roll_curr,
                np.ones_like(y_curr)
            ])
            X_list.append(X)
            y_list.append(y_next)
        except Exception:
            continue
            
    if not X_list:
        return np.array([0.8, 0.0, 0.0, 0.0])

    X_total = np.vstack(X_list)
    y_total = np.concatenate(y_list)
    
    # Remove NaNs/Infs
    mask = np.isfinite(X_total).all(axis=1) & np.isfinite(y_total)
    X_total = X_total[mask]
    y_total = y_total[mask]
    
    # Fit linear model using Ridge Regression for stability
    # params = (X.T @ X + lambda*I)^-1 @ X.T @ y
    lmbda = 1e-3
    n_features = X_total.shape[1]
    params = np.linalg.inv(X_total.T @ X_total + lmbda * np.eye(n_features)) @ X_total.T @ y_total
    
    print(f"Tuned Model Params: {params}")
    return params

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    a, b, c, d = self.model_params
    
    # Future plan data
    # future_plan.lataccel is the target
    # future_plan.roll_lataccel
    # future_plan.v_ego
    
    # We need to handle the case where future_plan is shorter than H
    # But typically it is 50 steps.
    
    H = min(self.H, len(future_plan.lataccel))
    
    if H < 1:
        return 0.0
    
    target_future = np.array(future_plan.lataccel[:H])
    roll_future = np.array(future_plan.roll_lataccel[:H])
    v_future = np.array(future_plan.v_ego[:H])
    
    # Generate random action sequences
    # Shape: (N, H)
    u_seqs = np.random.normal(0, self.sigma, (self.N, H))
    
    # Simulate trajectories
    # y_preds shape: (N, H)
    y_preds = np.zeros((self.N, H))
    y_curr = np.full(self.N, current_lataccel)
    
    # Vectorized simulation
    for t in range(H):
        # y_next = a*y_curr + b*u*v^2 + c*roll + d
        # v and roll are constant across samples for time t
        v_t = v_future[t]
        roll_t = roll_future[t]
        
        y_next = a * y_curr + b * u_seqs[:, t] * (v_t**2) + c * roll_t + d
        y_preds[:, t] = y_next
        y_curr = y_next
        
    # Compute Cost
    # J = 50 * (y - target)^2 + 100 * (y_t - y_{t-1})^2
    
    # Lataccel cost
    # target_future shape (H,) -> broadcast to (N, H)
    lat_cost = np.mean((y_preds - target_future[None, :])**2, axis=1) * 50
    
    # Jerk cost
    # We need y_{-1} which is current_lataccel
    # Diff: [y_0 - y_{-1}, y_1 - y_0, ...]
    y_prev = np.hstack([np.full((self.N, 1), current_lataccel), y_preds[:, :-1]])
    diffs = (y_preds - y_prev) / 0.1 # DEL_T = 0.1
    jerk_cost = np.mean(diffs**2, axis=1) * 100 # Note: The formula in README is mean of squared diffs?
    # README: (Sigma(diff)/steps)^2 ? No.
    # README: (Sigma( (a_t - a_{t-1})/dt )^2 ) / (steps-1) * 100
    # This implies sum of squared jerks, averaged.
    # My implementation: mean(diffs**2) * 100 matches the code in tinyphysics.py
    # tinyphysics.py: np.mean((np.diff(pred) / DEL_T)**2) * 100
    
    total_cost = lat_cost + jerk_cost
    
    # Select best sequence
    best_idx = np.argmin(total_cost)
    best_u = u_seqs[best_idx, 0]
    
    return best_u
