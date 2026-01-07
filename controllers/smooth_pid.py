from . import BaseController
import numpy as np

# Steering limits (match simulator)
STEER_MIN, STEER_MAX = -2.0, 2.0


class Controller(BaseController):
    """PID on lataccel with explicit smoothness constraints on steer rate and steer acceleration.

    - Base PID (with feedforward) tracks lateral acceleration.
    - First derivative (steer rate) is limited by max_delta.
    - Second derivative (steer acceleration) is actively damped toward 0 via a small PID.
    """

    def __init__(self):
        # Lataccel PID + feedforward
        self.kp = 0.17
        self.ki = 0.10
        self.kd = -0.06
        self.kff = 0.13

        # Smoothness limits
        self.max_delta = 0.10            # max change in steer per step
        self.max_delta_change = 0.04     # max change in steer rate per step

        # Second-derivative (steer acceleration) damping gains (very gentle)
        self.dd_kp = 0.02
        self.dd_kd = 0.005

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_action = 0.0
        self.prev_delta = 0.0            # last steer rate (Δaction)
        self.prev_delta_change = 0.0     # last steer acceleration (Δ²action)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Base PID on lateral acceleration
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        base_action = (self.kp * error) + (self.ki * self.error_integral) + (self.kd * error_diff) + (self.kff * target_lataccel)

        # Desired change in steer relative to previous action
        desired_delta = base_action - self.prev_action

        # Steer acceleration (second derivative) damping toward zero
        delta_change = desired_delta - self.prev_delta  # current steer acceleration proposal
        dd_error = -delta_change                        # want steer acceleration -> 0
        dd_error_diff = dd_error - (-self.prev_delta_change)
        smooth_correction = (self.dd_kp * dd_error) + (self.dd_kd * dd_error_diff)
        desired_delta += smooth_correction

        # Limit change of steer rate (second derivative clamp)
        desired_delta = np.clip(
            desired_delta,
            self.prev_delta - self.max_delta_change,
            self.prev_delta + self.max_delta_change,
        )

        # Limit steer rate magnitude (first derivative clamp)
        desired_delta = np.clip(desired_delta, -self.max_delta, self.max_delta)

        # Apply delta to get new action
        action = self.prev_action + desired_delta
        action = float(np.clip(action, STEER_MIN, STEER_MAX))

        # Update stored values
        self.prev_delta_change = desired_delta - self.prev_delta
        self.prev_delta = desired_delta
        self.prev_action = action
        self.prev_error = error

        return action
