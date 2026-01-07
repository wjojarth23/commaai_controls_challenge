from . import BaseController
import numpy as np
import json
import os
from collections import deque

# Path to tuned parameters file
PARAMS_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'mpc_params.json')


class Controller(BaseController):
    """
    MPC-style controller with:
    1. Pre-tuned parameters loaded from file (tuned across 1000 files)
    2. Light online adaptation from first 10 steps of ground truth
    3. Feedforward + feedback control for smooth tracking
    
    The controller uses a filtered error signal and rate limiting to prevent
    the zigzag behavior common in naive implementations.
    """
    
    def __init__(self, params_dict=None):
        """
        Initialize controller, optionally with parameters dict (for tuning).
        If params_dict is None, loads from file or uses defaults.
        """
        # Default parameters (will be overwritten by file or params_dict)
        self.kp = 0.3                    # Proportional gain
        self.ki = 0.05                   # Integral gain
        self.kd = 0.1                    # Derivative gain (on lataccel rate)
        self.ff_target = 0.2             # Feedforward from target
        self.ff_future = 0.05            # Feedforward from future trajectory
        self.ff_roll = 0.1               # Feedforward for road roll compensation
        self.action_rate_limit = 0.4     # Max steer change per timestep
        self.lookahead_steps = 5         # How far to look ahead in future plan
        self.error_filter_alpha = 0.3    # Error smoothing (0=max smooth, 1=no smooth)
        
        # Load tuned parameters from file if available
        if params_dict is not None:
            self._apply_params(params_dict)
        else:
            self._load_params_from_file()
        
        # State variables
        self.prev_action = 0.0
        self.prev_lataccel = 0.0
        self.filtered_error = 0.0
        self.error_integral = 0.0
        self.step_count = 0
        
        # Online adaptation
        self.calibration_steps = 10
        self.adapted = False
        self.action_buffer = deque(maxlen=20)
        self.lataccel_buffer = deque(maxlen=20)
        self.gain_adjustment = 1.0  # Multiplier learned from ground truth
        
        # Limits
        self.steer_min = -2.0
        self.steer_max = 2.0
        self.integral_limit = 1.5
        self.dt = 0.1
    
    def _apply_params(self, params):
        """Apply parameters from dict."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _load_params_from_file(self):
        """Load tuned parameters from JSON file if it exists."""
        try:
            if os.path.exists(PARAMS_FILE):
                with open(PARAMS_FILE, 'r') as f:
                    data = json.load(f)
                if 'params' in data:
                    self._apply_params(data['params'])
        except Exception:
            pass  # Use defaults if file doesn't exist or is invalid
    
    def _adapt_from_ground_truth(self):
        """
        Use the first 10 steps of ground truth to slightly adjust gains.
        This accounts for per-segment vehicle parameter variations.
        """
        if len(self.action_buffer) < 5 or len(self.lataccel_buffer) < 5:
            self.adapted = True
            return
        
        actions = np.array(list(self.action_buffer))
        lataccels = np.array(list(self.lataccel_buffer))
        
        # Estimate how responsive the vehicle is: d(lataccel)/d(action)
        if len(lataccels) > 2:
            lataccel_changes = np.diff(lataccels)
            actions_used = actions[:-1]
            
            # Only use meaningful samples
            mask = np.abs(actions_used) > 0.02
            if np.sum(mask) >= 2:
                changes = lataccel_changes[mask]
                acts = actions_used[mask]
                
                if np.var(acts) > 1e-8:
                    responsiveness = np.abs(np.mean(changes / (acts + 1e-6)))
                    
                    # Expected responsiveness based on our model
                    expected = 0.3
                    
                    # Adjust gain: if vehicle is more responsive, reduce our gains
                    if responsiveness > 0.01:
                        ratio = expected / (responsiveness + 1e-6)
                        self.gain_adjustment = np.clip(ratio, 0.7, 1.4)
        
        self.adapted = True
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute steering action.
        
        Args:
            target_lataccel: Target lateral acceleration
            current_lataccel: Current lateral acceleration
            state: State tuple (roll_lataccel, v_ego, a_ego)
            future_plan: Future trajectory (lataccel, roll_lataccel, v_ego, a_ego)
        
        Returns:
            Steering command in [-2, 2]
        """
        self.step_count += 1
        
        # Store for adaptation
        self.action_buffer.append(self.prev_action)
        self.lataccel_buffer.append(current_lataccel)
        
        # Adapt after calibration period
        if self.step_count == self.calibration_steps and not self.adapted:
            self._adapt_from_ground_truth()
        
        # =====================================================================
        # ERROR COMPUTATION WITH FILTERING
        # =====================================================================
        raw_error = target_lataccel - current_lataccel
        
        # Low-pass filter on error to reduce noise-induced oscillations
        self.filtered_error = (self.error_filter_alpha * raw_error + 
                               (1 - self.error_filter_alpha) * self.filtered_error)
        
        # =====================================================================
        # FEEDBACK CONTROL (PID-like)
        # =====================================================================
        
        # Proportional
        p_term = self.kp * self.filtered_error
        
        # Integral with anti-windup
        self.error_integral += self.filtered_error * self.dt
        self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.error_integral
        
        # Derivative on measurement (not error) for stability
        lataccel_rate = (current_lataccel - self.prev_lataccel) / self.dt
        d_term = -self.kd * lataccel_rate
        
        feedback = p_term + i_term + d_term
        
        # =====================================================================
        # FEEDFORWARD CONTROL
        # =====================================================================
        
        # Direct feedforward from target
        ff_direct = self.ff_target * target_lataccel
        
        # Road roll compensation
        roll_lataccel = state.roll_lataccel if hasattr(state, 'roll_lataccel') else 0.0
        ff_roll_comp = self.ff_roll * roll_lataccel
        
        # Anticipatory feedforward from future trajectory
        ff_future_comp = 0.0
        if future_plan is not None and hasattr(future_plan, 'lataccel') and future_plan.lataccel:
            future = future_plan.lataccel[:self.lookahead_steps]
            if len(future) > 0:
                # Weighted average of future targets
                weights = np.exp(-0.5 * np.arange(len(future)))
                weights /= np.sum(weights)
                future_avg = np.sum(np.array(future) * weights)
                
                # Feedforward based on where we need to go
                ff_future_comp = self.ff_future * (future_avg - target_lataccel)
        
        feedforward = ff_direct + ff_roll_comp + ff_future_comp
        
        # =====================================================================
        # COMBINE AND APPLY LIMITS
        # =====================================================================
        
        # Apply learned gain adjustment
        raw_action = self.gain_adjustment * (feedback + feedforward)
        
        # Rate limiting (crucial for smooth control)
        action = np.clip(
            raw_action,
            self.prev_action - self.action_rate_limit,
            self.prev_action + self.action_rate_limit
        )
        
        # Absolute limits
        action = np.clip(action, self.steer_min, self.steer_max)
        
        # Update state
        self.prev_action = action
        self.prev_lataccel = current_lataccel
        
        return action
