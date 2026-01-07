"""
PPO Controller

This controller uses a PPO-trained neural network to output steering commands.
The model was trained to minimize the total cost function:
  total_cost = lataccel_cost * 50 + jerk_cost

Uses CUDA-accelerated ONNX inference when available.
"""

from . import BaseController
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path


# Constants
DEL_T = 0.1
STEER_RANGE = [-2, 2]


class Controller(BaseController):
    """
    PPO-trained controller for lateral acceleration control.
    
    The neural network takes as input:
    - Current error (target - current acceleration)
    - Target lateral acceleration
    - Current lateral acceleration
    - Previous lateral acceleration
    - Roll lateral acceleration
    - Normalized velocity
    - Normalized longitudinal acceleration
    - Previous action
    - Future targets (4 steps)
    - Approximate desired jerk
    
    Total: 13 input features
    """
    
    def __init__(self, model_path=None):
        # Find model path
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "ppo_controller.onnx"
        
        self.model_path = Path(model_path)
        
        # Setup ONNX with CUDA if available
        options = ort.SessionOptions()
        options.log_severity_level = 3
        
        providers = []
        available = ort.get_available_providers()

        # Honor environment variable CPU_ONLY to force CPU-only inference
        cpu_only_env = os.environ.get('CPU_ONLY', os.environ.get('ONNX_CPU_ONLY', '0'))
        cpu_only = str(cpu_only_env).lower() in ('1', 'true', 'yes')

        if not cpu_only and 'CUDAExecutionProvider' in available:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))

        providers.append('CPUExecutionProvider')
        
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.session = ort.InferenceSession(f.read(), options, providers)
            self.model_loaded = True
        else:
            print(f"Warning: Model not found at {self.model_path}. Using fallback PID controller.")
            self.model_loaded = False
            self.session = None
        
        # State tracking
        self.prev_action = 0.0
        self.prev_lataccel = 0.0
        self.step_count = 0
        
        # Fallback PID parameters
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        """
        Update controller and return steering action.
        
        Args:
            target_lataccel: Target lateral acceleration
            current_lataccel: Current lateral acceleration
            state: Vehicle state (roll_lataccel, v_ego, a_ego)
            future_plan: Future plan with upcoming targets
        
        Returns:
            Steering action in STEER_RANGE
        """
        # Use neural network if available
        if self.model_loaded and self.session is not None:
            error = target_lataccel - current_lataccel
            
            # Get future targets from future_plan
            future_targets = [target_lataccel] * 4  # Default to current target
            if future_plan is not None and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) >= 4:
                future_targets = [future_plan.lataccel[i] for i in range(4)]
            
            # Approximate desired jerk
            desired_jerk = (target_lataccel - self.prev_lataccel) / DEL_T / 10.0
            
            # Build observation (13 features)
            obs = np.array([[
                error,                          # Current error
                target_lataccel,                # Target
                current_lataccel,               # Current actual
                self.prev_lataccel,             # Previous actual
                state.roll_lataccel,            # Roll
                state.v_ego / 30.0,             # Normalized velocity
                state.a_ego / 5.0,              # Normalized accel
                self.prev_action,               # Previous action
                future_targets[0],              # Future target +1
                future_targets[1],              # Future target +2
                future_targets[2],              # Future target +3
                future_targets[3],              # Future target +4
                desired_jerk,                   # Desired jerk
            ]], dtype=np.float32)
            
            # Run inference
            action = self.session.run(None, {'state': obs})[0][0, 0]
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            
            self.prev_action = action
            self.prev_lataccel = current_lataccel
            self.step_count += 1
            
            return float(action)
        
        # Fallback to PID
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        
        self.prev_action = action
        self.prev_lataccel = current_lataccel
        self.step_count += 1
        
        return float(action)
