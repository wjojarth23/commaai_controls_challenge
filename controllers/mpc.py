from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Minimal but optimized controller.
    
    Analysis: The simple controller with PID+FF gets ~92.
    To get to ~50, we need roughly 45% improvement.
    
    Key insight: The system has significant dynamics that a simple
    PID can't fully capture. We need:
    1. Better feedforward using future trajectory
    2. Possibly nonlinear gain scheduling
    3. Better derivative estimation
    """
    
    def __init__(self):
        # Tuned PID gains
        self.kp = 0.17
        self.ki = 0.10
        self.kd = -0.03
        
        # Feedforward
        self.kff = 0.13
        
        # State
        self.error_integral = 0.0
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # PID
        fb = self.kp * error + self.ki * self.error_integral + self.kd * error_diff
        
        # Feedforward
        ff = self.kff * target_lataccel
        
        return fb + ff
