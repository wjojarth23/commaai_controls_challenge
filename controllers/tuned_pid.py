from . import BaseController
import numpy as np
import pandas as pd
from pathlib import Path

class Controller(BaseController):
  def __init__(self):
    # Default PID gains (from simple.py)
    self.p = 0.17
    self.i = 0.10
    self.d = -0.03
    self.kff = 0.13
    
    self.error_integral = 0.0
    self.prev_error = 0.0
    
    # Tuning
    try:
      self.tune()
    except Exception as e:
      print(f"Tuning failed: {e}. Using default gains.")

  def tune(self):
    print("Tuning PID controller...")
    # 1. Load Data
    # Assuming data is in ../data relative to this file
    data_path = Path(__file__).resolve().parents[1] / "data"
    files = sorted(data_path.glob("*.csv"))
    
    # Use a subset of files for speed
    files = files[:20]
    
    if not files:
      print("No data found for tuning.")
      return

    all_y = []
    all_u = []
    all_v = []
    all_roll = []
    
    # For simulation
    sim_targets = []
    sim_vs = []
    sim_rolls = []
    
    for f in files:
      try:
        df = pd.read_csv(f)
        # Preprocess to match tinyphysics
        # steer_command is negated in tinyphysics
        u = -df['steerCommand'].values
        v = df['vEgo'].values
        # roll_lataccel = sin(roll) * 9.81
        roll = np.sin(df['roll'].values) * 9.81
        y = df['targetLateralAcceleration'].values
        
        # Collect data for fitting (y_next = f(y, u, v, roll))
        all_y.append(y)
        all_u.append(u)
        all_v.append(v)
        all_roll.append(roll)
        
        # Collect data for simulation
        sim_targets.append(y)
        sim_vs.append(v)
        sim_rolls.append(roll)
      except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

    if not all_y:
        return

    # Concatenate for fitting
    X_list = []
    Y_list = []
    
    for i in range(len(all_y)):
      y = all_y[i]
      u = all_u[i]
      v = all_v[i]
      r = all_roll[i]
      
      # Use valid range
      n = len(y) - 1
      if n < 1: continue
      
      y_curr = y[:-1]
      y_next = y[1:]
      u_curr = u[:-1]
      v_curr = v[:-1]
      r_curr = r[:-1]
      
      # Features: y_t, u_t*v_t^2, r_t, 1
      x_i = np.column_stack([
        y_curr,
        u_curr * (v_curr**2),
        r_curr,
        np.ones_like(y_curr)
      ])
      
      X_list.append(x_i)
      Y_list.append(y_next)
      
    X = np.vstack(X_list)
    Y = np.concatenate(Y_list)
    
    # Fit linear model
    # y_{t+1} = a*y_t + b*u_t*v^2 + c*r_t + d
    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    a, b, c, d = coeffs
    print(f"Identified model: a={a:.4f}, b={b:.2e}, c={c:.4f}, d={d:.4f}")
    
    # 2. Optimization
    # Prepare simulation data as numpy arrays (N_steps, N_traj)
    min_len = min(len(t) for t in sim_targets)
    # Limit to 500 steps as in tinyphysics cost evaluation
    sim_len = min(min_len, 500)
    
    sim_targets_arr = np.array([t[:sim_len] for t in sim_targets]).T # (T, N)
    sim_vs_arr = np.array([v[:sim_len] for v in sim_vs]).T
    sim_rolls_arr = np.array([r[:sim_len] for r in sim_rolls]).T
    
    def run_sim(params):
      kp, ki, kd, kff = params
      
      # State
      y_sim = np.zeros_like(sim_targets_arr)
      y_sim[0] = sim_targets_arr[0] # Start at target
      
      error_int = np.zeros(sim_targets_arr.shape[1])
      prev_error = np.zeros(sim_targets_arr.shape[1])
      
      # Loop
      for t in range(sim_len - 1):
        target = sim_targets_arr[t]
        current = y_sim[t]
        v = sim_vs_arr[t]
        roll = sim_rolls_arr[t]
        
        error = target - current
        error_int += error
        error_diff = error - prev_error
        prev_error = error
        
        # PID
        u = kp * error + ki * error_int + kd * error_diff + kff * target
        
        # Clip u
        u = np.clip(u, -2, 2)
        
        # Physics step
        # y_{t+1} = a*y_t + b*u_t*v^2 + c*r_t + d
        y_next = a * current + b * (u * (v**2)) + c * roll + d
        y_sim[t+1] = y_next
        
      # Compute cost
      # tinyphysics uses CONTROL_START_IDX=100, COST_END_IDX=500
      start_idx = 100
      end_idx = 500
      
      if sim_len <= start_idx:
        start_idx = 0
      if sim_len < end_idx:
        end_idx = sim_len
        
      y_eval = y_sim[start_idx:end_idx]
      r_eval = sim_targets_arr[start_idx:end_idx]
      
      mse = np.mean((y_eval - r_eval)**2)
      
      # Jerk: diff along time axis
      jerk = np.diff(y_sim, axis=0) / 0.1
      jerk_eval = jerk[start_idx:end_idx-1] # Adjust length
      mse_jerk = np.mean(jerk_eval**2)
      
      cost = mse * 5000 + mse_jerk * 100
      return cost

    # Coordinate Descent
    best_params = np.array([self.p, self.i, self.d, self.kff])
    best_cost = run_sim(best_params)
    print(f"Initial cost: {best_cost:.2f}")
    
    # Search range
    steps = [0.05, 0.02, 0.01, 0.05] # Step sizes for kp, ki, kd, kff
    
    for iter in range(10): # 10 passes
      changed = False
      for i in range(4):
        # Try +step
        params_plus = best_params.copy()
        params_plus[i] += steps[i]
        cost_plus = run_sim(params_plus)
        
        if cost_plus < best_cost:
          best_cost = cost_plus
          best_params = params_plus
          changed = True
          continue
          
        # Try -step
        params_minus = best_params.copy()
        params_minus[i] -= steps[i]
        cost_minus = run_sim(params_minus)
        
        if cost_minus < best_cost:
          best_cost = cost_minus
          best_params = params_minus
          changed = True
          
      if not changed:
        # Reduce step size
        steps = [s * 0.5 for s in steps]
        if max(steps) < 1e-4:
            break
      else:
          print(f"Iter {iter}: Cost {best_cost:.2f}, Params {best_params}")
        
    self.p, self.i, self.d, self.kff = best_params
    print(f"Tuned gains: kp={self.p:.4f}, ki={self.i:.4f}, kd={self.d:.4f}, kff={self.kff:.4f}")

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = target_lataccel - current_lataccel
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      
      # PID feedback
      fb = self.p * error + self.i * self.error_integral + self.d * error_diff
      
      # Feedforward on target (direct)
      ff = self.kff * target_lataccel
      
      return fb + ff
