from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    PID + Feedforward controller.
    """
    
    def __init__(self):
        # PID gains
        self.p = 0.17
        self.i = 0.1
        self.d = -0.06
        
        # Feedforward gain on target
        self.kff = 0.13
        
        self.error_integral = 0.0
        self.prev_error = 0.0

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
