"""
Acceleration Curve Optimizer

Optimizes a lateral acceleration curve to minimize:
- Tracking error to desired acceleration (weighted by 50)
- Jerk (rate of change of acceleration)

Then reverse-engineers the steering actions needed to achieve that curve
using a polynomial surrogate model of the physics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from scipy.ndimage import gaussian_filter1d
import argparse
import onnxruntime as ort
from collections import namedtuple
from tqdm import tqdm

# Constants
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
ACC_G = 9.81
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
VOCAB_SIZE = 1024

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


def load_target_curve(data_path):
    """Load the target lateral acceleration curve from CSV."""
    df = pd.read_csv(data_path)
    target = df['targetLateralAcceleration'].values
    return target[CONTROL_START_IDX:COST_END_IDX]


def compute_cost(curve, target):
    """
    Compute the total cost for a given acceleration curve.
    
    Returns: total_cost, lat_accel_cost, jerk_cost
    """
    # Tracking cost
    lat_accel_cost = np.mean((curve - target)**2) * 100
    
    # Jerk cost
    jerk = np.diff(curve) / DEL_T
    jerk_cost = np.mean(jerk**2) * 100
    
    # Total
    total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    
    return total_cost, lat_accel_cost, jerk_cost


def optimize_curve_scipy(target, verbose=True):
    """
    Optimize the acceleration curve using scipy.
    
    The problem: find curve that minimizes 50*tracking_error + jerk
    
    This is a quadratic optimization problem that can be solved efficiently.
    """
    n = len(target)
    
    def objective(curve):
        total_cost, _, _ = compute_cost(curve, target)
        return total_cost
    
    # Start from the target (perfect tracking, but may have high jerk)
    x0 = target.copy()
    
    if verbose:
        initial_cost, init_lat, init_jerk = compute_cost(x0, target)
        print(f"Initial (target curve): cost={initial_cost:.2f} (lat={init_lat:.2f}, jerk={init_jerk:.2f})")
    
    # Optimize
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': verbose}
    )
    
    optimized = result.x
    final_cost, final_lat, final_jerk = compute_cost(optimized, target)
    
    if verbose:
        print(f"Optimized: cost={final_cost:.2f} (lat={final_lat:.2f}, jerk={final_jerk:.2f})")
    
    return optimized, final_cost


def optimize_curve_analytical(target, verbose=True):
    """
    Solve the optimization problem analytically.
    
    We want to minimize: 50 * mean((x - target)^2) * 100 + mean((diff(x)/dt)^2) * 100
    
    This is equivalent to minimizing:
    50 * sum((x - target)^2) + (1/dt^2) * sum((x[i+1] - x[i])^2)
    
    Taking derivatives and setting to zero gives a linear system: Ax = b
    """
    n = len(target)
    alpha = LAT_ACCEL_COST_MULTIPLIER  # 50
    beta = 1.0 / (DEL_T ** 2)  # 100
    
    # Build tridiagonal matrix
    main_diag = np.full(n, alpha + 2*beta)
    main_diag[0] = alpha + beta
    main_diag[-1] = alpha + beta
    
    off_diag = np.full(n-1, -beta)
    
    A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    b = alpha * target
    
    optimized = np.linalg.solve(A, b)
    
    if verbose:
        initial_cost, init_lat, init_jerk = compute_cost(target, target)
        print(f"Target curve: cost={initial_cost:.2f} (lat={init_lat:.2f}, jerk={init_jerk:.2f})")
        
        final_cost, final_lat, final_jerk = compute_cost(optimized, target)
        print(f"Optimized:    cost={final_cost:.2f} (lat={final_lat:.2f}, jerk={final_jerk:.2f})")
    
    return optimized, compute_cost(optimized, target)[0]


def optimize_curve_constrained(target, start_lataccel, verbose=True):
    """
    Optimize curve with MAX_ACC_DELTA constraint.
    
    The physics model can only change lataccel by ±MAX_ACC_DELTA per step.
    We need to respect this constraint.
    
    We use iterative projection: optimize unconstrained, then project to feasible set.
    """
    n = len(target)
    
    # First, get unconstrained solution
    unconstrained, _ = optimize_curve_analytical(target, verbose=False)
    
    # Now project to respect MAX_ACC_DELTA constraint
    # This is a sequential constraint: |x[i+1] - x[i]| <= MAX_ACC_DELTA
    
    # Forward pass: ensure each step is reachable from previous
    constrained = np.zeros(n)
    constrained[0] = np.clip(unconstrained[0], 
                             start_lataccel - MAX_ACC_DELTA,
                             start_lataccel + MAX_ACC_DELTA)
    
    for i in range(1, n):
        constrained[i] = np.clip(unconstrained[i],
                                 constrained[i-1] - MAX_ACC_DELTA,
                                 constrained[i-1] + MAX_ACC_DELTA)
    
    # Now re-optimize within the feasible region using iterative refinement
    # We alternate between: 1) optimal for cost, 2) project to feasible
    for iteration in range(10):
        # Re-solve with current constrained as starting point
        alpha = LAT_ACCEL_COST_MULTIPLIER
        beta = 1.0 / (DEL_T ** 2)
        
        main_diag = np.full(n, alpha + 2*beta)
        main_diag[0] = alpha + beta
        main_diag[-1] = alpha + beta
        off_diag = np.full(n-1, -beta)
        A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        b = alpha * target
        
        unconstrained = np.linalg.solve(A, b)
        
        # Project to feasible
        new_constrained = np.zeros(n)
        new_constrained[0] = np.clip(unconstrained[0],
                                     start_lataccel - MAX_ACC_DELTA,
                                     start_lataccel + MAX_ACC_DELTA)
        for i in range(1, n):
            new_constrained[i] = np.clip(unconstrained[i],
                                         new_constrained[i-1] - MAX_ACC_DELTA,
                                         new_constrained[i-1] + MAX_ACC_DELTA)
        
        if np.allclose(constrained, new_constrained, atol=1e-6):
            break
        constrained = new_constrained
    
    if verbose:
        initial_cost, init_lat, init_jerk = compute_cost(target, target)
        print(f"Target curve:      cost={initial_cost:.2f} (lat={init_lat:.2f}, jerk={init_jerk:.2f})")
        
        uncon_cost, _, _ = compute_cost(unconstrained, target)
        print(f"Unconstrained:     cost={uncon_cost:.2f}")
        
        final_cost, final_lat, final_jerk = compute_cost(constrained, target)
        print(f"Constrained:       cost={final_cost:.2f} (lat={final_lat:.2f}, jerk={final_jerk:.2f})")
        
        # Check if constraint is satisfied
        max_delta = np.max(np.abs(np.diff(constrained)))
        print(f"Max delta:         {max_delta:.4f} (limit: {MAX_ACC_DELTA})")
    
    return constrained, compute_cost(constrained, target)[0]


def optimize_curve_smooth(target, sigma=2.0, verbose=True):
    """
    Simple approach: Gaussian smoothing of the target.
    Higher sigma = smoother = less jerk but more tracking error.
    """
    smoothed = gaussian_filter1d(target, sigma=sigma)
    
    if verbose:
        initial_cost, init_lat, init_jerk = compute_cost(target, target)
        print(f"Target curve: cost={initial_cost:.2f} (lat={init_lat:.2f}, jerk={init_jerk:.2f})")
        
        final_cost, final_lat, final_jerk = compute_cost(smoothed, target)
        print(f"Smoothed (sigma={sigma}): cost={final_cost:.2f} (lat={final_lat:.2f}, jerk={final_jerk:.2f})")
    
    return smoothed, compute_cost(smoothed, target)[0]


def plot_curves(target, optimized, title="Acceleration Curve Optimization"):
    """Plot the target and optimized curves with costs."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    t = np.arange(len(target)) * DEL_T
    
    # Cost computation
    target_cost, target_lat, target_jerk = compute_cost(target, target)
    opt_cost, opt_lat, opt_jerk = compute_cost(optimized, target)
    
    # Plot 1: Acceleration curves
    ax1 = axes[0]
    ax1.plot(t, target, 'b-', label=f'Target (cost={target_cost:.2f})', alpha=0.7, linewidth=1)
    ax1.plot(t, optimized, 'r-', label=f'Optimized (cost={opt_cost:.2f})', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Lateral Acceleration (m/s²)')
    ax1.set_title(f'{title}\nCost: {target_cost:.2f} → {opt_cost:.2f} ({(target_cost-opt_cost)/target_cost*100:.1f}% improvement)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tracking error
    ax2 = axes[1]
    error = optimized - target
    ax2.plot(t, error, 'g-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.fill_between(t, error, 0, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tracking Error (m/s²)')
    ax2.set_title(f'Tracking Error: MSE={opt_lat:.4f}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Jerk
    ax3 = axes[2]
    target_jerk_curve = np.diff(target) / DEL_T
    opt_jerk_curve = np.diff(optimized) / DEL_T
    t_jerk = t[:-1] + DEL_T/2
    
    ax3.plot(t_jerk, target_jerk_curve, 'b-', label=f'Target jerk (cost={target_jerk:.2f})', alpha=0.7, linewidth=1)
    ax3.plot(t_jerk, opt_jerk_curve, 'r-', label=f'Optimized jerk (cost={opt_jerk:.2f})', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Jerk (m/s³)')
    ax3.set_title(f'Jerk: {target_jerk:.2f} → {opt_jerk:.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============ Physics Model and Surrogate ============

class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[np.clip(token, 0, self.vocab_size - 1)]


def load_full_data(data_path):
    """Load full CSV data for simulation."""
    df = pd.read_csv(data_path)
    return pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values
    })


def get_onnx_session(model_path):
    """Load ONNX model."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ['CPUExecutionProvider']
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        providers.insert(0, 'CUDAExecutionProvider')
    with open(model_path, "rb") as f:
        return ort.InferenceSession(f.read(), options, providers)


def predict_lataccel_single(session, tokenizer, states, actions, past_lataccels):
    """Single step ONNX prediction."""
    tokenized = tokenizer.encode(np.array(past_lataccels))
    states_np = np.column_stack([actions, [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states]])
    input_data = {
        'states': np.expand_dims(states_np, axis=0).astype(np.float32),
        'tokens': np.expand_dims(tokenized, axis=0).astype(np.int64)
    }
    res = session.run(None, input_data)[0]
    token = np.argmax(res[0, -1])
    return tokenizer.decode(token)


class PhysicsSurrogate:
    """
    A polynomial surrogate model that approximates the physics network.
    
    The physics model maps: (state_history, action_history, lataccel_history) -> next_lataccel
    
    We simplify this to: next_lataccel ≈ f(current_lataccel, action, state)
    
    Key insight: the physics model has MAX_ACC_DELTA = 0.5 constraint, so 
    next_lataccel is bounded within ±0.5 of current_lataccel.
    
    We model: delta_lataccel = g(action, v_ego, current_lataccel)
    where delta_lataccel = next_lataccel - current_lataccel
    """
    
    def __init__(self):
        self.coeffs = None
        self.mean_action = 0
        self.std_action = 1
        self.mean_v = 0
        self.std_v = 1
    
    def collect_training_data(self, session, tokenizer, data, n_samples=500):
        """
        Collect input-output pairs from the real physics model.
        We sample random actions and observe the resulting lataccel change.
        """
        print("Collecting training data from physics model...")
        
        states = [State(
            roll_lataccel=data.iloc[i]['roll_lataccel'],
            v_ego=data.iloc[i]['v_ego'],
            a_ego=data.iloc[i]['a_ego']
        ) for i in range(len(data))]
        targets = data['target_lataccel'].values
        
        X = []  # [action, v_ego, current_lataccel, roll_lataccel]
        Y = []  # delta_lataccel
        
        # Initialize history
        state_history = list(states[:CONTEXT_LENGTH])
        action_history = list(data['steer_command'].values[:CONTEXT_LENGTH])
        lataccel_history = list(targets[:CONTEXT_LENGTH])
        current_lataccel = lataccel_history[-1]
        
        # Run through pre-control to build history
        for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state_history.append(states[step])
            action_history.append(data['steer_command'].values[step])
            current_lataccel = targets[step]
            lataccel_history.append(current_lataccel)
        
        # Collect samples during control period
        for step in tqdm(range(CONTROL_START_IDX, min(CONTROL_START_IDX + n_samples, COST_END_IDX)), 
                        desc="Sampling"):
            state = states[step]
            
            # Try several different actions at this state
            for action in np.linspace(STEER_RANGE[0], STEER_RANGE[1], 10):
                # Make copies of history
                sh = state_history.copy()
                ah = action_history.copy()
                lh = lataccel_history.copy()
                
                sh.append(state)
                ah.append(action)
                
                pred = predict_lataccel_single(
                    session, tokenizer,
                    sh[-CONTEXT_LENGTH:],
                    ah[-CONTEXT_LENGTH:],
                    lh[-CONTEXT_LENGTH:]
                )
                pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
                delta = pred - current_lataccel
                
                X.append([action, state.v_ego, current_lataccel, state.roll_lataccel])
                Y.append(delta)
            
            # Advance with a reasonable action
            state_history.append(state)
            best_action = 0.13 * targets[step]  # Simple feedforward
            action_history.append(best_action)
            
            pred = predict_lataccel_single(
                session, tokenizer,
                state_history[-CONTEXT_LENGTH:],
                action_history[-CONTEXT_LENGTH:],
                lataccel_history[-CONTEXT_LENGTH:]
            )
            pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            current_lataccel = pred
            lataccel_history.append(current_lataccel)
        
        return np.array(X), np.array(Y)
    
    def fit(self, X, Y):
        """
        Fit a polynomial model: delta = a0 + a1*action + a2*v + a3*lataccel + a4*action*v + ...
        """
        print("Fitting surrogate model...")
        
        # Normalize inputs
        self.mean_action = X[:, 0].mean()
        self.std_action = X[:, 0].std() + 1e-6
        self.mean_v = X[:, 1].mean()
        self.std_v = X[:, 1].std() + 1e-6
        
        # Build feature matrix (polynomial features)
        action = (X[:, 0] - self.mean_action) / self.std_action
        v_ego = (X[:, 1] - self.mean_v) / self.std_v
        lataccel = X[:, 2]
        roll = X[:, 3]
        
        # Features: 1, action, v, lataccel, action^2, action*v, action*lataccel
        features = np.column_stack([
            np.ones(len(X)),
            action,
            v_ego,
            lataccel,
            action ** 2,
            action * v_ego,
            action * lataccel,
            roll,
            action * roll,
        ])
        
        # Least squares fit
        self.coeffs, residuals, rank, s = np.linalg.lstsq(features, Y, rcond=None)
        
        # Compute R^2
        Y_pred = features @ self.coeffs
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        print(f"Surrogate fit: R² = {r2:.4f}")
        print(f"Coefficients: {self.coeffs}")
        
        return r2
    
    def predict_delta(self, action, v_ego, lataccel, roll_lataccel):
        """Predict lataccel change given action and state."""
        action_norm = (action - self.mean_action) / self.std_action
        v_norm = (v_ego - self.mean_v) / self.std_v
        
        features = np.array([
            1,
            action_norm,
            v_norm,
            lataccel,
            action_norm ** 2,
            action_norm * v_norm,
            action_norm * lataccel,
            roll_lataccel,
            action_norm * roll_lataccel,
        ])
        
        delta = features @ self.coeffs
        return np.clip(delta, -MAX_ACC_DELTA, MAX_ACC_DELTA)
    
    def invert_action(self, desired_lataccel, current_lataccel, v_ego, roll_lataccel):
        """
        Find the action that produces the desired lataccel.
        
        We want: desired = current + delta(action)
        So: delta = desired - current
        
        Solve for action using numerical optimization.
        """
        target_delta = desired_lataccel - current_lataccel
        target_delta = np.clip(target_delta, -MAX_ACC_DELTA, MAX_ACC_DELTA)
        
        def objective(action):
            pred_delta = self.predict_delta(action, v_ego, current_lataccel, roll_lataccel)
            return (pred_delta - target_delta) ** 2
        
        # Binary search / optimization
        result = minimize_scalar(
            objective, 
            bounds=(STEER_RANGE[0], STEER_RANGE[1]),
            method='bounded'
        )
        
        return np.clip(result.x, STEER_RANGE[0], STEER_RANGE[1])


def generate_actions_sequential(optimal_curve, session, tokenizer, data, verbose=True):
    """
    DEPRECATED - produces erratic steering. Use ILC instead.
    """
    pass  # Keeping for reference but not using


class GainSurrogate:
    """Surrogate for local sensitivity: gain ≈ d(next_lataccel)/d(action) along a trajectory.

    We fit a simple polynomial regression on features at each step:
      gain = w0 + w1*a + w2*v + w3*lat + w4*roll + ...
    """

    def __init__(self):
        self.coeffs = None
        self.mean_action = 0.0
        self.std_action = 1.0
        self.mean_v = 0.0
        self.std_v = 1.0

    def fit(self, action, v_ego, lataccel, roll_lataccel, gain):
        action = np.asarray(action)
        v_ego = np.asarray(v_ego)
        lataccel = np.asarray(lataccel)
        roll_lataccel = np.asarray(roll_lataccel)
        gain = np.asarray(gain)

        self.mean_action = float(action.mean())
        self.std_action = float(action.std() + 1e-6)
        self.mean_v = float(v_ego.mean())
        self.std_v = float(v_ego.std() + 1e-6)

        a = (action - self.mean_action) / self.std_action
        v = (v_ego - self.mean_v) / self.std_v
        lat = lataccel
        roll = roll_lataccel

        features = np.column_stack([
            np.ones(len(action)),
            a,
            v,
            lat,
            roll,
            a * v,
            a * lat,
            a * roll,
            a ** 2,
        ])

        self.coeffs, *_ = np.linalg.lstsq(features, gain, rcond=None)
        return self

    def predict(self, action, v_ego, lataccel, roll_lataccel):
        action = np.asarray(action)
        v_ego = np.asarray(v_ego)
        lataccel = np.asarray(lataccel)
        roll_lataccel = np.asarray(roll_lataccel)

        a = (action - self.mean_action) / self.std_action
        v = (v_ego - self.mean_v) / self.std_v
        lat = lataccel
        roll = roll_lataccel

        features = np.column_stack([
            np.ones(len(action)),
            a,
            v,
            lat,
            roll,
            a * v,
            a * lat,
            a * roll,
            a ** 2,
        ])

        return features @ self.coeffs


def fit_cubic_error_to_delta(err, target_delta):
    """Fit a cubic delta = c0 + c1*e + c2*e^2 + c3*e^3 via least squares."""
    e = np.asarray(err)
    y = np.asarray(target_delta)
    X = np.column_stack([np.ones_like(e), e, e**2, e**3])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def fit_cubic_ridge(err, target_delta, lam=1e-3):
    """Ridge-regularized cubic fit to avoid overfitting noisy gains."""
    e = np.asarray(err)
    y = np.asarray(target_delta)
    X = np.column_stack([np.ones_like(e), e, e**2, e**3])
    I = np.eye(X.shape[1])
    coeffs = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return coeffs


def eval_cubic(coeffs, err):
    e = np.asarray(err)
    return coeffs[0] + coeffs[1]*e + coeffs[2]*e**2 + coeffs[3]*e**3


def simulate_full_with_local_gains(session, tokenizer, data, actions, states, target_full, eps=0.05):
    """Run full simulation and also estimate per-step local gain d(lataccel)/d(action).

    Gain is estimated by symmetric finite difference on the *current* step action:
      gain[t] ≈ (f(a+eps) - f(a-eps)) / (2 eps)

    Returns: achieved_curve, gains, feature dict (for fitting a surrogate)
    """
    state_history = list(states[:CONTEXT_LENGTH])
    action_history = list(actions[:CONTEXT_LENGTH])
    lataccel_history = list(target_full[:CONTEXT_LENGTH])
    current_lataccel = lataccel_history[-1]

    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])
        current_lataccel = target_full[step]
        lataccel_history.append(current_lataccel)

    achieved = []
    gains = []
    feat_action = []
    feat_v = []
    feat_lat = []
    feat_roll = []

    for step in range(CONTROL_START_IDX, COST_END_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])

        sh = state_history[-CONTEXT_LENGTH:]
        ah = action_history[-CONTEXT_LENGTH:]
        lh = lataccel_history[-CONTEXT_LENGTH:]

        # nominal
        pred0 = predict_lataccel_single(session, tokenizer, sh, ah, lh)
        pred0 = np.clip(pred0, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # local gain by perturbing the last action in the context window
        a0 = float(ah[-1])
        a_plus = float(np.clip(a0 + eps, STEER_RANGE[0], STEER_RANGE[1]))
        a_minus = float(np.clip(a0 - eps, STEER_RANGE[0], STEER_RANGE[1]))

        ah_plus = list(ah)
        ah_minus = list(ah)
        ah_plus[-1] = a_plus
        ah_minus[-1] = a_minus

        pred_plus = predict_lataccel_single(session, tokenizer, sh, ah_plus, lh)
        pred_minus = predict_lataccel_single(session, tokenizer, sh, ah_minus, lh)
        pred_plus = np.clip(pred_plus, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        pred_minus = np.clip(pred_minus, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        gain = (pred_plus - pred_minus) / (2.0 * eps)

        current_lataccel = pred0
        lataccel_history.append(current_lataccel)
        achieved.append(pred0)
        gains.append(gain)

        feat_action.append(a0)
        feat_v.append(states[step].v_ego)
        feat_lat.append(pred0)
        feat_roll.append(states[step].roll_lataccel)

    feats = {
        'action': np.asarray(feat_action),
        'v_ego': np.asarray(feat_v),
        'lataccel': np.asarray(feat_lat),
        'roll_lataccel': np.asarray(feat_roll),
    }

    return np.asarray(achieved), np.asarray(gains), feats


def generate_actions_gain_ilc(
    desired_curve,
    session,
    tokenizer,
    data,
    n_iterations=20,
    eps_gain=0.05,
    step_scale=0.15,
    max_action_delta=0.05,
    action_smooth_sigma=1.2,
    gain_floor=0.03,
    verbose=True,
):
    """Refine a PID action sequence by correcting under/over-acceleration with a learned gain surrogate.

    Procedure:
    1) Build an initial closed-loop PID+FF rollout for desired_curve (so actions are sane).
    2) Estimate local gains d(lataccel)/d(action) along that trajectory using finite difference.
    3) Fit a simple surrogate gain model.
    4) Iteratively update actions:
         action += step_scale * (desired - achieved) / gain_hat
       with clipping + smoothing.
    """
    init_actions, init_achieved, states = generate_actions_pid_rollout(
        desired_curve, session, tokenizer, data, verbose=verbose
    )
    target_full = data['target_lataccel'].values
    target_curve = target_full[CONTROL_START_IDX:COST_END_IDX]

    actions = init_actions.copy()
    best_cost = float('inf')
    best_actions = actions.copy()
    best_achieved = init_achieved.copy()

    # Train initial gain surrogate from the PID trajectory
    achieved0, gains0, feats0 = simulate_full_with_local_gains(
        session, tokenizer, data, actions, states, target_full, eps=eps_gain
    )
    gain_model = GainSurrogate().fit(
        feats0['action'], feats0['v_ego'], feats0['lataccel'], feats0['roll_lataccel'], gains0
    )

    for it in range(1, n_iterations + 1):
        achieved, gains_raw, feats = simulate_full_with_local_gains(
            session, tokenizer, data, actions, states, target_full, eps=eps_gain
        )

        cost, lat_cost, jerk_cost = compute_cost(achieved, target_curve)
        if cost < best_cost:
            best_cost = cost
            best_actions = actions.copy()
            best_achieved = achieved.copy()

        err = desired_curve - achieved

        # Predict a smoother gain (surrogate) and combine with raw to avoid sign mistakes
        gain_hat = gain_model.predict(
            feats['action'], feats['v_ego'], feats['lataccel'], feats['roll_lataccel']
        )
        gain_hat = 0.7 * gain_hat + 0.3 * gains_raw

        # Prevent division blow-ups; keep sign.
        gain_safe = np.where(np.abs(gain_hat) < gain_floor, np.sign(gain_hat + 1e-12) * gain_floor, gain_hat)

        delta_u = step_scale * (err / gain_safe)
        delta_u = np.clip(delta_u, -max_action_delta, max_action_delta)

        u = actions[CONTROL_START_IDX:COST_END_IDX] + delta_u
        if action_smooth_sigma is not None and action_smooth_sigma > 0:
            u = gaussian_filter1d(u, sigma=action_smooth_sigma)
        u = np.clip(u, STEER_RANGE[0], STEER_RANGE[1])

        actions[CONTROL_START_IDX:COST_END_IDX] = u
        actions[COST_END_IDX:] = actions[COST_END_IDX - 1]

        # Re-fit gain model occasionally on the new trajectory
        if it % 5 == 0:
            gain_model.fit(feats['action'], feats['v_ego'], feats['lataccel'], feats['roll_lataccel'], gains_raw)

        if verbose and (it == 1 or it % 5 == 0 or it == n_iterations):
            print(
                f"  Iter {it:2d}: cost={cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f}) "
                f"mean|err|={np.mean(np.abs(err)):.4f} best={best_cost:.2f}"
            )

    return best_actions, best_achieved


def plot_steer_response_curve(desired_curve, session, tokenizer, data, timestep_idx, n_samples=100, verbose=True):
    """
    Plot the steer-to-lataccel response curve at a specific timestep.
    
    Runs a baseline PID trajectory to the specified timestep, then tests n_samples different
    steer values to see what lateral acceleration each produces.
    
    Args:
        desired_curve: Target lateral acceleration curve
        session: ONNX session
        tokenizer: Lataccel tokenizer
        data: Full data DataFrame
        timestep_idx: Index in the control window (0 to len(desired_curve)-1)
        n_samples: Number of steer values to test (default 100)
    """
    if verbose:
        print(f"\nGenerating steer response curve at timestep {timestep_idx}...")
    
    # Generate baseline PID trajectory to get to the timestep
    baseline_gains = (0.17, 0.10, -0.06, 0.13)
    baseline_actions, baseline_achieved, states = generate_actions_pid_rollout(
        desired_curve, session, tokenizer, data, verbose=False
    )
    
    target_full = data['target_lataccel'].values
    
    # Build history up to the target timestep
    state_history = list(states[:CONTEXT_LENGTH])
    action_history = list(baseline_actions[:CONTEXT_LENGTH])
    lataccel_history = list(target_full[:CONTEXT_LENGTH])
    
    # Simulate forward to the target step
    current_lataccel = lataccel_history[-1]
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX + timestep_idx):
        state_history.append(states[step])
        action_history.append(baseline_actions[step])
        
        if step < CONTROL_START_IDX:
            current_lataccel = target_full[step]
        else:
            sh = state_history[-CONTEXT_LENGTH:]
            ah = action_history[-CONTEXT_LENGTH:]
            lh = lataccel_history[-CONTEXT_LENGTH:]
            current_lataccel = predict_lataccel_single(session, tokenizer, sh, ah, lh)
            current_lataccel = np.clip(current_lataccel, 
                                      lataccel_history[-1] - MAX_ACC_DELTA,
                                      lataccel_history[-1] + MAX_ACC_DELTA)
        
        lataccel_history.append(current_lataccel)
    
    # Now we're at the target timestep
    step = CONTROL_START_IDX + timestep_idx
    state_history.append(states[step])
    
    # Test different steer values
    steer_values = np.linspace(STEER_RANGE[0], STEER_RANGE[1], n_samples)
    resulting_lataccels = []
    
    if verbose:
        print(f"Testing {n_samples} steer values...")
    
    for steer in tqdm(steer_values, disable=not verbose):
        # Create modified action history with this steer value
        ah_test = list(action_history[-CONTEXT_LENGTH+1:])  # Take last CONTEXT_LENGTH-1
        ah_test.append(steer)  # Add the test steer as the last one
        
        sh = state_history[-CONTEXT_LENGTH:]
        lh = lataccel_history[-CONTEXT_LENGTH:]
        
        pred = predict_lataccel_single(session, tokenizer, sh, ah_test, lh)
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        resulting_lataccels.append(pred)
    
    resulting_lataccels = np.array(resulting_lataccels)
    
    # Plot
    t = timestep_idx * DEL_T
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.scatter(steer_values, resulting_lataccels, c='tab:blue', s=30, alpha=0.6, label='Predicted response')
    
    # Mark the baseline PID action
    baseline_steer = baseline_actions[CONTROL_START_IDX + timestep_idx]
    baseline_lat = baseline_achieved[timestep_idx]
    ax.scatter([baseline_steer], [baseline_lat], c='red', s=100, marker='x', 
               linewidth=3, label=f'Baseline PID (steer={baseline_steer:.3f}, lat={baseline_lat:.3f})', zorder=5)
    
    # Mark desired lataccel
    desired_lat = desired_curve[timestep_idx]
    ax.axhline(desired_lat, color='green', linestyle='--', linewidth=2, 
               label=f'Desired lataccel = {desired_lat:.3f}')
    
    # Mark current lataccel constraint bounds (theoretical)
    ax.axhline(current_lataccel - MAX_ACC_DELTA, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Theoretical MAX_ACC_DELTA bounds (±{MAX_ACC_DELTA})')
    ax.axhline(current_lataccel + MAX_ACC_DELTA, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Mark actual achievable bounds
    min_achievable = resulting_lataccels.min()
    max_achievable = resulting_lataccels.max()
    ax.axhline(min_achievable, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7,
               label=f'Actual achievable range')
    ax.axhline(max_achievable, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7)
    
    # Shade unreachable region if desired is outside achievable range
    is_reachable = min_achievable <= desired_lat <= max_achievable
    if not is_reachable:
        if desired_lat < min_achievable:
            ax.axhspan(desired_lat, min_achievable, alpha=0.15, color='red', 
                      label='UNREACHABLE gap')
        else:
            ax.axhspan(max_achievable, desired_lat, alpha=0.15, color='red',
                      label='UNREACHABLE gap')
    
    ax.set_xlabel('Steer Command', fontsize=12)
    ax.set_ylabel('Resulting Lateral Acceleration (m/s²)', fontsize=12)
    
    # Check reachability
    gap = 0.0
    if not is_reachable:
        if desired_lat < min_achievable:
            gap = min_achievable - desired_lat
        else:
            gap = desired_lat - max_achievable
    
    reachability_text = "✓ REACHABLE" if is_reachable else f"✗ UNREACHABLE (gap: {gap:.3f})"
    ax.set_title(f'Steer Response Curve at Timestep {timestep_idx} (t={t:.2f}s)\n'
                 f'Current lataccel: {current_lataccel:.3f} m/s² | {reachability_text}', fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    if verbose:
        print(f"Current lataccel: {current_lataccel:.3f}")
        print(f"Desired lataccel: {desired_lat:.3f}")
        print(f"Baseline steer: {baseline_steer:.3f} → {baseline_lat:.3f}")
        print(f"Theoretical bounds (MAX_ACC_DELTA): [{current_lataccel - MAX_ACC_DELTA:.3f}, {current_lataccel + MAX_ACC_DELTA:.3f}]")
        print(f"Actual achievable range: [{min_achievable:.3f}, {max_achievable:.3f}]")
        print(f"Achievable span: {max_achievable - min_achievable:.3f}")
        print(f"Reachability: {reachability_text}")
        if not is_reachable:
            print(f"  → The optimal curve asks for acceleration the physics cannot deliver!")
            print(f"  → This is why action generation methods struggle to match the optimal curve.")
    
    return fig


def generate_actions_greedy_search(desired_curve, session, tokenizer, data, n_samples=200, verbose=True):
    """
    Brute force greedy search: at each timestep, test n_samples steer values 
    and pick the one that produces lataccel closest to desired.
    
    This is a greedy sequential approach - we commit to the best action at each step
    and move forward in closed-loop.
    """
    if verbose:
        print(f"\nGenerating actions using greedy search with {n_samples} samples per step...")
    
    states = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    
    target_full = data['target_lataccel'].values
    
    # Initialize
    actions = np.zeros(len(data))
    actions[:CONTROL_START_IDX] = data['steer_command'].values[:CONTROL_START_IDX]
    
    state_history = list(states[:CONTEXT_LENGTH])
    action_history = list(actions[:CONTEXT_LENGTH])
    lataccel_history = list(target_full[:CONTEXT_LENGTH])
    
    # Warmup to CONTROL_START_IDX
    current_lataccel = lataccel_history[-1]
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])
        current_lataccel = target_full[step]
        lataccel_history.append(current_lataccel)
    
    achieved = []
    steer_range = np.linspace(STEER_RANGE[0], STEER_RANGE[1], n_samples)
    
    # Greedy search through control window
    for idx in tqdm(range(len(desired_curve)), desc="Greedy search", disable=not verbose):
        step = CONTROL_START_IDX + idx
        state_history.append(states[step])
        
        desired_lat = desired_curve[idx]
        best_steer = 0.0
        best_pred = current_lataccel
        best_error = float('inf')
        
        # Test all steer values
        for steer in steer_range:
            ah_test = list(action_history[-CONTEXT_LENGTH+1:])
            ah_test.append(steer)
            
            sh = state_history[-CONTEXT_LENGTH:]
            lh = lataccel_history[-CONTEXT_LENGTH:]
            
            pred = predict_lataccel_single(session, tokenizer, sh, ah_test, lh)
            pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            
            error = abs(pred - desired_lat)
            if error < best_error:
                best_error = error
                best_steer = steer
                best_pred = pred
        
        # Commit to best action
        actions[step] = best_steer
        action_history.append(best_steer)
        current_lataccel = best_pred
        lataccel_history.append(current_lataccel)
        achieved.append(best_pred)
    
    # Fill remaining
    actions[COST_END_IDX:] = actions[COST_END_IDX - 1]
    
    achieved = np.array(achieved)
    
    if verbose:
        cost, lat_cost, jerk_cost = compute_cost(achieved, desired_curve)
        print(f"Greedy search result: cost={cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f})")
        
        errors = np.abs(achieved - desired_curve)
        print(f"Mean absolute error: {errors.mean():.4f}")
        print(f"Max absolute error: {errors.max():.4f}")
        print(f"Median absolute error: {np.median(errors):.4f}")
        
        # Check first few steps to see if we start off-track
        print(f"\nFirst 10 steps:")
        for i in range(min(10, len(achieved))):
            print(f"  Step {i}: desired={desired_curve[i]:.3f}, achieved={achieved[i]:.3f}, error={errors[i]:.3f}, steer={actions[CONTROL_START_IDX+i]:.3f}")
        
        action_changes = np.abs(np.diff(actions[CONTROL_START_IDX:COST_END_IDX]))
        print(f"\nMean |Δsteer|: {action_changes.mean():.4f}")
        print(f"Max |Δsteer|: {action_changes.max():.4f}")
    
    return actions, achieved


def generate_actions_beam_search(desired_curve, session, tokenizer, data, 
                                  beam_width=5, n_samples=50, verbose=True):
    """
    Beam search: maintain top-K trajectories at each step.
    
    At each timestep:
    1. For each of the K current trajectories
    2. Try n_samples different steer values
    3. Keep the best K*n_samples combinations
    4. Select top K to continue
    
    This is much smarter than greedy since we explore multiple paths.
    Complexity: O(beam_width * n_samples * n_steps * inference_time)
    """
    if verbose:
        print(f"\nBeam search: beam_width={beam_width}, n_samples={n_samples}")
    
    states = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    
    target_full = data['target_lataccel'].values
    n_steps = len(desired_curve)
    
    # Initialize beams: each beam is (actions_so_far, state_history, action_history, lataccel_history, current_lat, cumulative_cost)
    initial_actions = data['steer_command'].values[:CONTROL_START_IDX].tolist()
    initial_states = states[:CONTEXT_LENGTH]
    initial_lataccels = target_full[:CONTEXT_LENGTH].tolist()
    
    # Warmup to CONTROL_START_IDX
    state_hist = list(states[:CONTEXT_LENGTH])
    action_hist = list(initial_actions[:CONTEXT_LENGTH])
    lat_hist = list(initial_lataccels)
    
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_hist.append(states[step])
        action_hist.append(initial_actions[step])
        lat_hist.append(target_full[step])
    
    beams = [{
        'actions': [],  # actions in control window
        'achieved': [],  # achieved lataccels
        'state_hist': state_hist.copy(),
        'action_hist': action_hist.copy(),
        'lat_hist': lat_hist.copy(),
        'current_lat': lat_hist[-1],
        'cost': 0.0
    }]
    
    steer_values = np.linspace(STEER_RANGE[0], STEER_RANGE[1], n_samples)
    
    # Beam search through control window
    for step_idx in tqdm(range(n_steps), desc="Beam search", disable=not verbose):
        step = CONTROL_START_IDX + step_idx
        desired_lat = desired_curve[step_idx]
        
        # Expand all beams
        candidates = []
        
        for beam in beams:
            # Try all steer values for this beam
            for steer in steer_values:
                # Build histories for prediction (need CONTEXT_LENGTH of each)
                # State history: take last states + append new state, then take last CONTEXT_LENGTH
                sh_for_pred = (beam['state_hist'] + [states[step]])[-CONTEXT_LENGTH:]
                
                # Action history: take last CONTEXT_LENGTH-1 actions + new steer
                ah_for_pred = beam['action_hist'][-(CONTEXT_LENGTH-1):] + [steer]
                
                # Lataccel history: last CONTEXT_LENGTH
                lh_for_pred = beam['lat_hist'][-CONTEXT_LENGTH:]
                
                # Predict
                pred = predict_lataccel_single(session, tokenizer, sh_for_pred, ah_for_pred, lh_for_pred)
                pred = np.clip(pred, beam['current_lat'] - MAX_ACC_DELTA, beam['current_lat'] + MAX_ACC_DELTA)
                
                # Compute incremental cost (squared error at this step)
                error = pred - desired_lat
                step_cost = error ** 2
                
                # Create new beam
                new_beam = {
                    'actions': beam['actions'] + [steer],
                    'achieved': beam['achieved'] + [pred],
                    'state_hist': beam['state_hist'] + [states[step]],
                    'action_hist': beam['action_hist'] + [steer],
                    'lat_hist': beam['lat_hist'] + [pred],
                    'current_lat': pred,
                    'cost': beam['cost'] + step_cost
                }
                candidates.append(new_beam)
        
        # Keep top beam_width
        candidates.sort(key=lambda x: x['cost'])
        beams = candidates[:beam_width]
    
    # Select best beam
    best_beam = beams[0]
    
    # Convert to full action array
    actions = np.zeros(len(data))
    actions[:CONTROL_START_IDX] = data['steer_command'].values[:CONTROL_START_IDX]
    actions[CONTROL_START_IDX:COST_END_IDX] = best_beam['actions']
    actions[COST_END_IDX:] = actions[COST_END_IDX - 1]
    
    achieved = np.array(best_beam['achieved'])
    
    if verbose:
        cost, lat_cost, jerk_cost = compute_cost(achieved, desired_curve)
        print(f"Beam search result: cost={cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f})")
        
        errors = np.abs(achieved - desired_curve)
        print(f"Mean absolute error: {errors.mean():.4f}")
        print(f"Max absolute error: {errors.max():.4f}")
    
    return actions, achieved


def visualize_error_and_tweak(desired_curve, session, tokenizer, data, verbose=True, playhead_idx=None):
    """Visualization: desired curve, 5 PID rollouts, and an interactive playhead scatter."""
    from matplotlib.widgets import Slider
    
    t = np.arange(len(desired_curve)) * DEL_T

    gains_list = [
        (0.12, 0.08, -0.04, 0.11),
        (0.15, 0.09, -0.05, 0.12),
        (0.17, 0.10, -0.06, 0.13),  # baseline
        (0.20, 0.11, -0.07, 0.14),
        (0.24, 0.12, -0.08, 0.15),
    ]

    family = generate_pid_family(desired_curve, session, tokenizer, data, gains_list)

    # pick playhead
    idx = playhead_idx if playhead_idx is not None else len(desired_curve) // 2
    idx = max(0, min(len(desired_curve) - 1, idx))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 0.3], hspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax_slider = fig.add_subplot(gs[2])

    # Top: lataccel curves
    ax0.plot(t, desired_curve, 'k--', label='Desired', linewidth=2)
    for j, f in enumerate(family):
        g = f['gains']
        c = f['achieved']
        cost = f['cost']
        label = f"PID{j+1} p={g[0]:.2f} i={g[1]:.2f} d={g[2]:.2f} kff={g[3]:.2f} (cost={cost:.1f})"
        ax0.plot(t, c, linewidth=1.2, alpha=0.8, label=label)
    vline = ax0.axvline(t[idx], color='m', linestyle='--', linewidth=1.5, label=f'Playhead')
    ax0.set_title('Lateral Accel: Desired vs PID variants')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Lat Accel (m/s²)')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Bottom: scatter plot
    steers = [f['actions'][idx] for f in family]
    lats = [f['achieved'][idx] for f in family]
    scatter = ax1.scatter(steers, lats, c='tab:blue', s=50, zorder=3)
    annotations = []
    for j, (s, la) in enumerate(zip(steers, lats)):
        ann = ax1.annotate(f"PID{j+1}", (s, la), textcoords="offset points", xytext=(5,5), fontsize=8)
        annotations.append(ann)
    desired_line = ax1.axhline(desired_curve[idx], color='k', linestyle='--', linewidth=1.5, label='Desired lataccel')
    ax1.set_xlabel('Steer command')
    ax1.set_ylabel('Achieved lat accel (m/s²)')
    ax1.set_title(f'Playhead: steer vs lataccel (index={idx}, t={t[idx]:.2f}s)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Slider
    slider = Slider(ax_slider, 'Playhead Index', 0, len(desired_curve)-1, valinit=idx, valstep=1)

    def update(val):
        new_idx = int(slider.val)
        vline.set_xdata([t[new_idx], t[new_idx]])
        
        new_steers = [f['actions'][new_idx] for f in family]
        new_lats = [f['achieved'][new_idx] for f in family]
        scatter.set_offsets(np.c_[new_steers, new_lats])
        
        for j, (s, la) in enumerate(zip(new_steers, new_lats)):
            annotations[j].set_position((s, la))
        
        desired_line.set_ydata([desired_curve[new_idx], desired_curve[new_idx]])
        ax1.set_title(f'Playhead: steer vs lataccel (index={new_idx}, t={t[new_idx]:.2f}s)')
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    return fig


def generate_actions_ilc(optimal_curve, session, tokenizer, data, 
                         n_iterations=50, learning_rate=0.5, verbose=True):
    """
    Iterative Learning Control (ILC) approach.
    
    Instead of trying to invert each step independently:
    1. Start with a PID-like controller targeting the optimal curve
    2. Run full simulation, get achieved curve
    3. Compute error at each step
    4. Adjust actions: action += lr * error
    5. Repeat until convergence
    
    This respects the sequential dynamics and produces smooth actions.
    """
    if verbose:
        print("Generating actions using Iterative Learning Control...")
    
    states = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    target = data['target_lataccel'].values
    n = len(optimal_curve)
    
    # Initialize actions with PID-like controller targeting optimal curve
    actions = np.zeros(len(data))
    actions[:CONTROL_START_IDX] = data['steer_command'].values[:CONTROL_START_IDX]
    
    # Run a "virtual" PID to initialize actions based on optimal curve
    kp, ki, kd, kff = 0.20, 0.10, -0.05, 0.13
    error_integral = 0.0
    prev_error = 0.0
    current_lataccel = target[CONTROL_START_IDX - 1]
    
    for i in range(n):
        target_accel = optimal_curve[i]
        error = target_accel - current_lataccel
        error_integral += error
        error_diff = error - prev_error
        prev_error = error
        
        action = kp * error + ki * error_integral + kd * error_diff + kff * target_accel
        actions[CONTROL_START_IDX + i] = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        
        # Assume we achieve something close to target (rough estimate)
        current_lataccel = current_lataccel + 0.3 * (target_accel - current_lataccel)
    
    actions[COST_END_IDX:] = actions[COST_END_IDX - 1]
    
    # Smooth initial actions
    actions[CONTROL_START_IDX:COST_END_IDX] = gaussian_filter1d(
        actions[CONTROL_START_IDX:COST_END_IDX], sigma=1.5
    )
    actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
    
    best_cost = float('inf')
    best_actions = actions.copy()
    best_achieved = None
    lr = learning_rate
    
    for iteration in range(n_iterations):
        # Run simulation with current actions
        achieved = simulate_full(session, tokenizer, data, actions, states, target)
        
        # Compute cost
        cost, lat_cost, jerk_cost = compute_cost(achieved, target[CONTROL_START_IDX:COST_END_IDX])
        
        if cost < best_cost:
            best_cost = cost
            best_actions = actions.copy()
            best_achieved = achieved.copy()
        
        # Compute error between optimal and achieved
        error = optimal_curve - achieved
        mean_abs_error = np.mean(np.abs(error))
        
        # Update actions using error
        # Positive error = we're below target = need more steer (usually)
        adjustment = lr * error
        
        # Smooth the adjustment
        adjustment = gaussian_filter1d(adjustment, sigma=3.0)
        
        # Apply adjustment
        actions[CONTROL_START_IDX:COST_END_IDX] += adjustment
        
        # Regularize: blend towards simple feedforward to prevent drift
        ff_actions = kff * optimal_curve
        actions[CONTROL_START_IDX:COST_END_IDX] = (
            0.95 * actions[CONTROL_START_IDX:COST_END_IDX] + 
            0.05 * ff_actions
        )
        
        # Smooth actions
        actions[CONTROL_START_IDX:COST_END_IDX] = gaussian_filter1d(
            actions[CONTROL_START_IDX:COST_END_IDX], sigma=1.0
        )
        
        # Clip
        actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
        
        # Adaptive learning rate
        lr *= 0.95
        
        if verbose and (iteration % 5 == 0 or iteration == n_iterations - 1):
            print(f"  Iter {iteration+1:2d}: cost={cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f}), "
                  f"err={mean_abs_error:.4f}, lr={lr:.4f}")
        
        # Early stopping if converged
        if mean_abs_error < 0.01:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break
    
    if verbose:
        print(f"  Best cost: {best_cost:.2f}")
    
    return best_actions, best_achieved


def generate_actions_pid_rollout(desired_curve, session, tokenizer, data, gains=None, verbose=True):
    """Generate an initial steering sequence by running the Simple PID+FF controller in closed-loop.

    This uses the *real* physics model for the closed-loop rollout, so the integral/derivative terms
    reflect the plant dynamics (unlike the previous "virtual" PID estimate).

    desired_curve is length (COST_END_IDX - CONTROL_START_IDX).
    """
    if verbose:
        print("Generating initial actions via closed-loop Simple PID+FF rollout...")

    states = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    target_full = data['target_lataccel'].values

    n = len(desired_curve)
    actions = np.zeros(len(data))
    actions[:CONTROL_START_IDX] = data['steer_command'].values[:CONTROL_START_IDX]

    # Simple controller gains (controllers/simple.py) or override
    if gains is None:
        p, i, d, kff = 0.17, 0.10, -0.06, 0.13
    else:
        p, i, d, kff = gains
    error_integral = 0.0
    prev_error = 0.0

    # Histories seeded with the pre-control segment.
    state_history = list(states[:CONTEXT_LENGTH])
    action_history = list(actions[:CONTEXT_LENGTH])
    lataccel_history = list(target_full[:CONTEXT_LENGTH])
    current_lataccel = lataccel_history[-1]

    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])
        current_lataccel = target_full[step]
        lataccel_history.append(current_lataccel)

    # Closed-loop rollout for the control window
    achieved = []
    for k in range(n):
        step = CONTROL_START_IDX + k
        target_lataccel = desired_curve[k]

        error = target_lataccel - current_lataccel
        error_integral += error
        error_diff = error - prev_error
        prev_error = error

        fb = p * error + i * error_integral + d * error_diff
        ff = kff * target_lataccel
        action = np.clip(fb + ff, STEER_RANGE[0], STEER_RANGE[1])
        actions[step] = action

        state_history.append(states[step])
        action_history.append(action)

        pred = predict_lataccel_single(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = pred
        lataccel_history.append(current_lataccel)
        achieved.append(pred)

    actions[COST_END_IDX:] = actions[COST_END_IDX - 1]

    return actions, np.array(achieved), states


def generate_pid_family(desired_curve, session, tokenizer, data, gains_list):
    """Run multiple PID gain sets and return list of (gains, steer_segment, achieved_curve, cost)."""
    family = []
    for g in gains_list:
        actions, achieved, _ = generate_actions_pid_rollout(desired_curve, session, tokenizer, data, gains=g, verbose=False)
        cost, lat_cost, jerk_cost = compute_cost(achieved, desired_curve)
        family.append({
            'gains': g,
            'actions': actions[CONTROL_START_IDX:COST_END_IDX].copy(),
            'achieved': achieved.copy(),
            'cost': cost,
            'lat_cost': lat_cost,
            'jerk_cost': jerk_cost,
        })
    return family


def regress_optimal_steer(desired_curve, family, smooth_sigma=1.5):
    """Per-timestep linear regression: fit lataccel = m*steer + b from PID samples and solve for desired.

    If slope is tiny, fall back to median steer. Smooth the resulting steer sequence.
    """
    n = len(desired_curve)
    steers = np.stack([f['actions'] for f in family], axis=1)  # shape (n, k)
    lats = np.stack([f['achieved'] for f in family], axis=1)  # shape (n, k)

    s_opt = np.zeros(n)
    for t in range(n):
        S = steers[t]
        Y = lats[t]
        X = np.column_stack([S, np.ones_like(S)])  # [steer, 1]
        coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
        m, b = coeffs
        if abs(m) < 1e-3:
            s_opt[t] = np.median(S)
        else:
            s_opt[t] = (desired_curve[t] - b) / m
    # smooth and clip
    s_opt = gaussian_filter1d(s_opt, sigma=smooth_sigma)
    s_opt = np.clip(s_opt, STEER_RANGE[0], STEER_RANGE[1])
    return s_opt


def _action_smoothness_grad(u):
    """Gradient of sum(diff(u)^2) w.r.t u (1D)."""
    # For objective: sum_{t} (u[t+1]-u[t])^2
    # grad is 2*(2*u[t] - u[t-1] - u[t+1]) for interior.
    g = np.zeros_like(u)
    if len(u) < 2:
        return g
    dif = np.diff(u)
    # contributes -2*dif[t] to u[t], +2*dif[t] to u[t+1]
    g[:-1] -= 2.0 * dif
    g[1:] += 2.0 * dif
    return g


def generate_actions_spsa(
    initial_actions,
    session,
    tokenizer,
    data,
    states,
    target_curve,
    n_iterations=40,
    a0=0.03,
    c0=0.04,
    alpha=0.602,
    gamma=0.101,
    action_smooth_weight=0.02,
    delta_smooth_sigma=2.0,
    action_smooth_sigma=1.0,
    knot_stride=5,
    grad_clip=50.0,
    max_step=0.05,
    verbose=True,
):
    """Black-box gradient descent on the steering sequence using SPSA.

    SPSA estimates the gradient with only two rollouts per iteration:
      g \approx (J(u+cΔ) - J(u-cΔ)) / (2c) * Δ

    This is appropriate here because the ONNX model is tokenized/quantized and not differentiable.
    """
    if verbose:
        print("Optimizing actions with SPSA (2 rollouts/iter)...")

    def expand_knots(knots, n):
        if len(knots) == n:
            return knots
        xk = np.linspace(0, n - 1, num=len(knots))
        x = np.arange(n)
        return np.interp(x, xk, knots)

    target_full = data['target_lataccel'].values
    u0 = initial_actions[CONTROL_START_IDX:COST_END_IDX].copy()
    n = len(u0)

    # Optimize in a lower-dimensional *residual* space for stability.
    # This guarantees the initial iterate equals the provided initial_actions.
    if knot_stride is None or knot_stride <= 1:
        theta = np.zeros_like(u0)
        idx = None
    else:
        n_knots = int(np.ceil(n / knot_stride))
        idx = np.linspace(0, n - 1, num=n_knots).round().astype(int)
        theta = np.zeros(n_knots, dtype=u0.dtype)

    def theta_to_u(theta_vec):
        residual = expand_knots(theta_vec, n)
        u_vec = u0 + residual
        if action_smooth_sigma is not None and action_smooth_sigma > 0:
            # light smoothing on actions is OK; keep baseline exact by smoothing residual only
            u_vec = u0 + gaussian_filter1d(residual, sigma=action_smooth_sigma)
        return np.clip(u_vec, STEER_RANGE[0], STEER_RANGE[1])

    u = theta_to_u(theta)

    best_cost = float('inf')
    best_u = u.copy()
    best_achieved = None

    # Baseline evaluation
    base_actions = initial_actions.copy()
    base_actions[CONTROL_START_IDX:COST_END_IDX] = u
    achieved = simulate_full(session, tokenizer, data, base_actions, states, target_full)
    cost, lat_cost, jerk_cost = compute_cost(achieved, target_curve)
    best_cost = cost
    best_u = u.copy()
    best_achieved = achieved
    if verbose:
        print(f"  Init: cost={cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f})")

    rng = np.random.default_rng(0)
    for k in range(1, n_iterations + 1):
        ak = a0 / (k ** alpha)
        ck = c0 / (k ** gamma)

        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        if delta_smooth_sigma is not None and delta_smooth_sigma > 0:
            delta = gaussian_filter1d(delta, sigma=delta_smooth_sigma)
            delta = np.sign(delta + 1e-12)  # keep it roughly Rademacher

        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta

        u_plus = theta_to_u(theta_plus)
        u_minus = theta_to_u(theta_minus)

        actions_plus = initial_actions.copy()
        actions_minus = initial_actions.copy()
        actions_plus[CONTROL_START_IDX:COST_END_IDX] = u_plus
        actions_minus[CONTROL_START_IDX:COST_END_IDX] = u_minus

        achieved_plus = simulate_full(session, tokenizer, data, actions_plus, states, target_full)
        j_plus, _, _ = compute_cost(achieved_plus, target_curve)

        achieved_minus = simulate_full(session, tokenizer, data, actions_minus, states, target_full)
        j_minus, _, _ = compute_cost(achieved_minus, target_curve)

        # SPSA gradient estimate
        ghat = (j_plus - j_minus) / (2.0 * ck) * delta

        # Clip gradient to control variance spikes from the quantized model
        if grad_clip is not None and grad_clip > 0:
            ghat = np.clip(ghat, -grad_clip, grad_clip)

        # Smoothness regularization on actions (keeps steer reasonable)
        if action_smooth_weight is not None and action_smooth_weight > 0:
            # regularize in action space then project back to knots
            u_curr = expand_knots(theta, n)
            reg_u = _action_smoothness_grad(u_curr)
            if len(theta) == n:
                reg_theta = reg_u
            else:
                idx = np.linspace(0, n - 1, num=len(theta)).round().astype(int)
                reg_theta = reg_u[idx]
            ghat = ghat + action_smooth_weight * reg_theta

        # Gradient step (on residual parameters)
        step = -ak * ghat
        if max_step is not None and max_step > 0:
            step = np.clip(step, -max_step, max_step)
        theta = theta + step

        # Post-processing: rebuild action curve from residual parameters
        u = theta_to_u(theta)

        # Evaluate current iterate occasionally (or always; n is small)
        actions_eval = initial_actions.copy()
        actions_eval[CONTROL_START_IDX:COST_END_IDX] = u
        achieved_eval = simulate_full(session, tokenizer, data, actions_eval, states, target_full)
        cost, lat_cost, jerk_cost = compute_cost(achieved_eval, target_curve)

        if cost < best_cost:
            best_cost = cost
            best_u = u.copy()
            best_achieved = achieved_eval

        if verbose and (k % 5 == 0 or k == 1 or k == n_iterations):
            print(
                f"  Iter {k:2d}: cost={cost:.2f} (lat={lat_cost:.2f}, jerk={jerk_cost:.2f}) "
                f"ak={ak:.4f} ck={ck:.4f} best={best_cost:.2f}"
            )

    best_actions = initial_actions.copy()
    best_actions[CONTROL_START_IDX:COST_END_IDX] = best_u
    best_actions[COST_END_IDX:] = best_actions[COST_END_IDX - 1]
    return best_actions, best_achieved


def simulate_full(session, tokenizer, data, actions, states, target):
    """Run full simulation and return the achieved lataccel curve."""
    state_history = list(states[:CONTEXT_LENGTH])
    action_history = list(actions[:CONTEXT_LENGTH])
    lataccel_history = list(target[:CONTEXT_LENGTH])
    current_lataccel = lataccel_history[-1]
    
    # Build up history through pre-control period
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])
        current_lataccel = target[step]
        lataccel_history.append(current_lataccel)
    
    # Run control period
    achieved = []
    for step in range(CONTROL_START_IDX, COST_END_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])
        
        pred = predict_lataccel_single(
            session, tokenizer,
            state_history[-CONTEXT_LENGTH:],
            action_history[-CONTEXT_LENGTH:],
            lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
        current_lataccel = pred
        lataccel_history.append(current_lataccel)
        achieved.append(pred)
    
    return np.array(achieved)


def generate_actions_from_curve(optimal_curve, data, surrogate):
    """
    Given the optimal acceleration curve, generate the steering actions
    that will (approximately) produce this curve.
    """
    n = len(optimal_curve)
    actions = np.zeros(len(data))
    
    # Copy initial actions
    actions[:CONTROL_START_IDX] = data['steer_command'].values[:CONTROL_START_IDX]
    
    # Use the surrogate to invert the curve
    current_lataccel = data['target_lataccel'].values[CONTROL_START_IDX - 1]
    
    for i in range(n):
        step = CONTROL_START_IDX + i
        desired_lataccel = optimal_curve[i]
        v_ego = data.iloc[step]['v_ego']
        roll_lataccel = data.iloc[step]['roll_lataccel']
        
        action = surrogate.invert_action(desired_lataccel, current_lataccel, v_ego, roll_lataccel)
        actions[step] = action
        
        # Update current lataccel using surrogate (for next iteration)
        delta = surrogate.predict_delta(action, v_ego, current_lataccel, roll_lataccel)
        current_lataccel = current_lataccel + delta
    
    # Fill rest
    actions[COST_END_IDX:] = actions[COST_END_IDX - 1]
    
    return actions


def simulate_with_actions(session, tokenizer, data, actions):
    """Run full simulation with given actions, return achieved lataccel curve."""
    states = [State(
        roll_lataccel=data.iloc[i]['roll_lataccel'],
        v_ego=data.iloc[i]['v_ego'],
        a_ego=data.iloc[i]['a_ego']
    ) for i in range(len(data))]
    targets = data['target_lataccel'].values
    
    state_history = list(states[:CONTEXT_LENGTH])
    action_history = list(actions[:CONTEXT_LENGTH])
    lataccel_history = list(targets[:CONTEXT_LENGTH])
    current_lataccel = lataccel_history[-1]
    
    for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state_history.append(states[step])
        action_history.append(actions[step])
        current_lataccel = targets[step]
        lataccel_history.append(current_lataccel)
    
    control_lataccels = []
    for step in range(CONTROL_START_IDX, min(COST_END_IDX, len(data))):
        state_history.append(states[step])
        action_history.append(actions[step])
        
        pred = predict_lataccel_single(
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


def main():
    parser = argparse.ArgumentParser(description="Optimize acceleration curves")
    parser.add_argument("--data_path", type=str, default="./data/00000.csv")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--mode", type=str, default="optimize",
                       choices=["optimize", "full", "spsa", "gain", "viz", "pid_regress", "steer_response", "greedy", "beam"],
                       help="optimize: just curve optimization, full: curve + ILC action generation, spsa: PID init + SPSA optimization, gain: PID init + gain-based corrections, viz: PID family visualization, pid_regress: regress steer from PID family without plotting, steer_response: plot steer-to-lataccel response at a timestep, greedy: brute force greedy search, beam: beam search (smart brute force)")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no_show", action="store_true",
                        help="Do not open an interactive plot window")
    parser.add_argument("--viz_index", type=int, default=None,
                        help="Playhead index for viz mode or timestep for steer_response mode (0-based within control window)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of steer values to sample for steer_response mode")
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam width for beam search mode")
    args = parser.parse_args()
    
    # Load target curve
    print(f"Loading data from {args.data_path}")
    data = load_full_data(args.data_path)
    target = data['target_lataccel'].values
    target_curve = target[CONTROL_START_IDX:COST_END_IDX]
    start_lataccel = target[CONTROL_START_IDX - 1]
    print(f"Curve length: {len(target_curve)} points")
    print(f"Starting lataccel: {start_lataccel:.3f}")
    
    # Step 1: Optimize the curve with physics constraints
    print("\n=== Step 1: Curve Optimization (with MAX_ACC_DELTA constraint) ===")
    optimal_curve, _ = optimize_curve_constrained(target_curve, start_lataccel, verbose=True)
    
    if args.mode in ("full", "spsa", "gain", "viz", "pid_regress", "steer_response", "greedy", "beam"):
        # Step 2: Build surrogate OR use sequential search
        print("\n=== Step 2: Loading Physics Model ===")
        session = get_onnx_session(args.model_path)
        tokenizer = LataccelTokenizer()
        data = load_full_data(args.data_path)

        target_full = data['target_lataccel'].values
        target_curve = target_full[CONTROL_START_IDX:COST_END_IDX]

        if args.mode == "full":
            # Step 3: Generate actions using ILC (smooth, iterative approach)
            print("\n=== Step 3: Generating Actions (Iterative Learning Control) ===")
            actions, achieved_curve = generate_actions_ilc(
                optimal_curve, session, tokenizer, data,
                n_iterations=30, learning_rate=0.5, verbose=True
            )
        elif args.mode == "spsa":
            print("\n=== Step 3: PID Init (closed-loop rollout) ===")
            init_actions, init_achieved, states = generate_actions_pid_rollout(
                optimal_curve, session, tokenizer, data, verbose=True
            )
            init_cost, init_lat, init_jerk = compute_cost(init_achieved, target_curve)
            print(f"  PID-init achieved cost: {init_cost:.2f} (lat={init_lat:.2f}, jerk={init_jerk:.2f})")

            print("\n=== Step 4: SPSA Optimization (black-box gradient descent) ===")
            actions, achieved_curve = generate_actions_spsa(
                init_actions,
                session,
                tokenizer,
                data,
                states,
                target_curve,
                n_iterations=35,
                a0=0.02,
                c0=0.03,
                knot_stride=6,
                grad_clip=30.0,
                max_step=0.03,
                action_smooth_weight=0.05,
                verbose=True,
            )
        elif args.mode == "gain":
            print("\n=== Step 3: PID Init + Gain-Based Corrections ===")
            actions, achieved_curve = generate_actions_gain_ilc(
                optimal_curve,
                session,
                tokenizer,
                data,
                n_iterations=25,
                eps_gain=0.05,
                step_scale=0.10,
                max_action_delta=0.04,
                action_smooth_sigma=1.2,
                gain_floor=0.03,
                verbose=True,
            )
        elif args.mode == "viz":
            print("\n=== Step 3: Visualization (PID + colored error + cubic tweak) ===")
            fig = visualize_error_and_tweak(
                optimal_curve, session, tokenizer, data, verbose=True, playhead_idx=args.viz_index
            )
            if args.save:
                plt.savefig(args.save, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {args.save}")
            if not args.no_show:
                plt.show()
            return
        elif args.mode == "steer_response":
            print("\n=== Step 3: Steer Response Curve ===")
            timestep_idx = args.viz_index if args.viz_index is not None else len(optimal_curve) // 2
            timestep_idx = max(0, min(len(optimal_curve) - 1, timestep_idx))
            fig = plot_steer_response_curve(
                optimal_curve, session, tokenizer, data, timestep_idx, 
                n_samples=args.n_samples, verbose=True
            )
            if args.save:
                plt.savefig(args.save, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {args.save}")
            if not args.no_show:
                plt.show()
            return
        elif args.mode == "greedy":
            print("\n=== Step 3: Greedy Brute Force Search ===")
            actions, achieved_curve = generate_actions_greedy_search(
                optimal_curve, session, tokenizer, data, 
                n_samples=args.n_samples, verbose=True
            )
        elif args.mode == "beam":
            print("\n=== Step 3: Beam Search (Smart Brute Force) ===")
            actions, achieved_curve = generate_actions_beam_search(
                optimal_curve, session, tokenizer, data,
                beam_width=args.beam_width, n_samples=args.n_samples, verbose=True
            )
        else:
            print("\n=== Step 3: PID family regression (no plotting) ===")
            gains_list = [
                (0.12, 0.08, -0.04, 0.11),
                (0.15, 0.09, -0.05, 0.12),
                (0.17, 0.10, -0.06, 0.13),
                (0.20, 0.11, -0.07, 0.14),
                (0.24, 0.12, -0.08, 0.15),
            ]
            family = generate_pid_family(optimal_curve, session, tokenizer, data, gains_list)
            regressed_steer = regress_optimal_steer(optimal_curve, family, smooth_sigma=1.5)

            actions = np.zeros(len(data))
            actions[:CONTROL_START_IDX] = data['steer_command'].values[:CONTROL_START_IDX]
            actions[CONTROL_START_IDX:COST_END_IDX] = regressed_steer
            actions[COST_END_IDX:] = actions[COST_END_IDX - 1]

            states = [State(
                roll_lataccel=data.iloc[i]['roll_lataccel'],
                v_ego=data.iloc[i]['v_ego'],
                a_ego=data.iloc[i]['a_ego']
            ) for i in range(len(data))]
            target_full = data['target_lataccel'].values
            achieved_curve = simulate_full(session, tokenizer, data, actions, states, target_full)

            reg_cost, reg_lat, reg_jerk = compute_cost(achieved_curve, optimal_curve)
            print(f"Regressed steer cost: {reg_cost:.2f} (lat={reg_lat:.2f}, jerk={reg_jerk:.2f})")
            # proceed to common reporting/plotting below
        
        # Compute costs
        target_cost, target_lat, target_jerk = compute_cost(target_curve, target_curve)
        optimal_cost, opt_lat, opt_jerk = compute_cost(optimal_curve, target_curve)
        achieved_cost, ach_lat, ach_jerk = compute_cost(achieved_curve, target_curve)
        
        print(f"\nResults:")
        print(f"  Target curve cost:   {target_cost:.2f} (lat={target_lat:.2f}, jerk={target_jerk:.2f})")
        print(f"  Optimal curve cost:  {optimal_cost:.2f} (lat={opt_lat:.2f}, jerk={opt_jerk:.2f})")
        print(f"  Achieved curve cost: {achieved_cost:.2f} (lat={ach_lat:.2f}, jerk={ach_jerk:.2f})")
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        t = np.arange(len(target_curve)) * DEL_T
        
        axes[0].plot(t, target_curve, 'b-', label=f'Target (cost={target_cost:.2f})', alpha=0.5)
        axes[0].plot(t, optimal_curve, 'g--', label=f'Optimal (cost={optimal_cost:.2f})', linewidth=2)
        axes[0].plot(t, achieved_curve, 'r-', label=f'Achieved (cost={achieved_cost:.2f})', linewidth=1.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Lat Accel (m/s²)')
        axes[0].set_title('Curve Optimization + Action Generation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, actions[CONTROL_START_IDX:COST_END_IDX], 'purple', linewidth=1)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Steer Command')
        axes[1].set_title('Generated Actions')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    else:
        # Just plot the curve optimization
        fig = plot_curves(target_curve, optimal_curve, title="Analytical Curve Optimization")
    
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
