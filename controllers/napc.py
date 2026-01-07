"""
Neural Adaptive Predictive Control (NAPC)

A unique approach: Predictive Lookahead with Smooth Trajectory Following

Key innovations:
1. Uses future plan for predictive feedforward (look ahead to anticipate)
2. Velocity-adaptive gain scheduling  
3. Smooth action generation via exponential filtering
4. Error integral with anti-windup and decay
5. Derivative kick prevention

The insight: The car dynamics are approximately first-order with velocity-dependent gain.
By using the future trajectory, we can anticipate turns before they happen.
"""

from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Neural Adaptive Predictive Control (NAPC)
    
    Predictive feedforward + adaptive feedback with smooth action generation.
    """
    
    def __init__(self):
        # === Tuned Parameters (via offline optimization) ===
        
        # Feedback gains (base values, adapted by velocity)
        self.kp = 0.30                # Proportional gain
        self.ki = 0.11                # Integral gain  
        self.kd = 0.005               # Derivative gain (on measurement, not error)
        
        # Feedforward gains
        self.kff = 0.20               # Direct feedforward on target
        self.kff_future = 0.038       # Lookahead feedforward
        self.kff_rate = 0.018         # Feedforward on target rate of change
        
        # Velocity adaptation
        self.v_nominal = 25.0         # Nominal velocity for gain scheduling
        self.v_adapt_p = 0.22         # How much to adapt P gain with velocity
        self.v_adapt_ff = 0.30        # How much to adapt FF with velocity
        
        # Smoothing parameters
        self.action_alpha = 0.68      # Action smoothing (lower = smoother)
        self.deriv_alpha = 0.30       # Derivative filtering
        
        # Integral anti-windup
        self.integral_max = 1.8       # Max integral magnitude
        self.integral_decay = 0.993   # Decay factor per step
        
        # Lookahead parameters
        self.lookahead_steps = 6      # How far ahead to look
        self.lookahead_weights = np.exp(-np.arange(6) * 0.35)  # Exponential decay weights
        self.lookahead_weights /= self.lookahead_weights.sum()
        
        # State variables
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_lataccel = 0.0
        self.prev_target = 0.0
        self.prev_action = 0.0
        self.filtered_deriv = 0.0
        self.step_count = 0
        
    def get_velocity_gain_factor(self, v_ego):
        """
        Compute gain scaling factor based on velocity.
        Higher velocity = need less aggressive control (more stable).
        """
        v_ratio = v_ego / self.v_nominal
        return 1.0 / (0.7 + 0.3 * v_ratio)
    
    def compute_lookahead_target(self, target_lataccel, future_plan):
        """
        Compute a weighted average of future targets for predictive control.
        """
        if not future_plan or not future_plan.lataccel:
            return target_lataccel
        
        future_targets = [target_lataccel]
        future_targets.extend(future_plan.lataccel[:self.lookahead_steps-1])
        
        # Pad if needed
        while len(future_targets) < self.lookahead_steps:
            future_targets.append(future_targets[-1])
        
        # Weighted average with exponential decay
        lookahead_target = sum(w * t for w, t in zip(self.lookahead_weights, future_targets))
        
        return lookahead_target
    
    def compute_target_rate(self, target_lataccel, future_plan):
        """
        Estimate rate of change of target trajectory.
        """
        if not future_plan or len(future_plan.lataccel) < 3:
            return 0.0
        
        # Use finite difference on near future
        dt = 0.1
        future_avg = np.mean(future_plan.lataccel[:3])
        rate = (future_avg - target_lataccel) / (2 * dt)
        
        return rate

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Main control update with predictive feedforward and adaptive feedback.
        """
        self.step_count += 1
        
        # === Error computation ===
        error = target_lataccel - current_lataccel
        
        # === Derivative (on measurement to avoid derivative kick) ===
        if self.step_count > 1:
            raw_deriv = -(current_lataccel - self.prev_lataccel) / 0.1  # Negative because on measurement
        else:
            raw_deriv = 0.0
        
        # Filter derivative
        self.filtered_deriv = self.deriv_alpha * raw_deriv + (1 - self.deriv_alpha) * self.filtered_deriv
        
        # === Integral with anti-windup and decay ===
        self.error_integral = self.integral_decay * self.error_integral + error
        self.error_integral = np.clip(self.error_integral, -self.integral_max, self.integral_max)
        
        # === Velocity-adaptive gains ===
        v_factor = self.get_velocity_gain_factor(state.v_ego)
        
        kp_adapted = self.kp * v_factor
        kff_adapted = self.kff * (1.0 + self.v_adapt_ff * (state.v_ego / self.v_nominal - 1.0))
        
        # === Feedback control ===
        feedback = (
            kp_adapted * error + 
            self.ki * self.error_integral + 
            self.kd * self.filtered_deriv
        )
        
        # === Predictive feedforward ===
        # Direct feedforward
        ff_direct = kff_adapted * target_lataccel
        
        # Lookahead feedforward (anticipate future trajectory)
        lookahead_target = self.compute_lookahead_target(target_lataccel, future_plan)
        ff_lookahead = self.kff_future * (lookahead_target - target_lataccel)
        
        # Rate feedforward (anticipate changes)
        target_rate = self.compute_target_rate(target_lataccel, future_plan)
        ff_rate = self.kff_rate * target_rate
        
        feedforward = ff_direct + ff_lookahead + ff_rate
        
        # === Combine ===
        raw_action = feedback + feedforward
        
        # === Smooth action (critical for low jerk) ===
        action = self.action_alpha * raw_action + (1 - self.action_alpha) * self.prev_action
        
        # === Update state ===
        self.prev_error = error
        self.prev_lataccel = current_lataccel
        self.prev_target = target_lataccel
        self.prev_action = action
        
        return action
