"""
Differentiable Neural Network Controller (DiffNet)

This controller uses the key insight that the tinyphysics.onnx simulator is
a neural network, making it differentiable. We train a small neural network
controller by backpropagating the cost function THROUGH the physics simulator.

Strategy:
1. Load the tinyphysics ONNX model as a PyTorch module (frozen weights)
2. Connect a small "Controller Network" to the inputs
3. Backpropagate the Total Cost loss through the physics simulator
4. The optimal control policy is "distilled" into the small network

This is essentially "model-based policy optimization" where the model IS
the actual simulator we're optimizing for.
"""

from . import BaseController
import numpy as np
import os

# Path to the trained model weights
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'diffnet_controller.pth')


class Controller(BaseController):
    """
    Neural Network controller trained via backpropagation through
    the differentiable physics simulator.
    """
    
    def __init__(self):
        self.model = None
        self.device = 'cpu'
        self._load_model()
        
        # State for inference
        self.prev_action = 0.0
        self.prev_lataccel = 0.0
        self.prev_target = 0.0
        self.step_count = 0
        
    def _load_model(self):
        """Load the trained PyTorch model."""
        try:
            import torch
            self.torch = torch
            
            # Define the same architecture used during training
            self.model = ControllerNetwork()
            
            if os.path.exists(MODEL_WEIGHTS_PATH):
                self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
                print(f"DiffNet: Loaded trained weights from {MODEL_WEIGHTS_PATH}")
            else:
                print(f"DiffNet: No trained weights found at {MODEL_WEIGHTS_PATH}, using random init")
                print("  Run 'python train_diffnet.py' to train the controller")
            
            self.model.eval()
        except ImportError:
            print("DiffNet: PyTorch not available, falling back to simple PID")
            self.model = None
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute control action using the trained neural network.
        """
        if self.model is None:
            # Fallback PID if model not loaded
            return self._fallback_pid(target_lataccel, current_lataccel, state)
        
        # Prepare input features
        features = self._prepare_features(target_lataccel, current_lataccel, state, future_plan)
        
        with self.torch.no_grad():
            input_tensor = self.torch.tensor(features, dtype=self.torch.float32).unsqueeze(0)
            action = self.model(input_tensor).item()
        
        # Update state for next step
        self.prev_action = action
        self.prev_lataccel = current_lataccel
        self.prev_target = target_lataccel
        self.step_count += 1
        
        return action
    
    def _prepare_features(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Prepare input features for the neural network.
        
        Features (normalized):
        - Error (target - current)
        - Target lateral acceleration
        - Current lateral acceleration  
        - Roll lateral acceleration
        - Velocity (normalized)
        - Acceleration
        - Previous action
        - Error derivative (current - prev)
        - Target derivative (target - prev_target)
        - Future trajectory features (lookahead)
        """
        error = target_lataccel - current_lataccel
        error_deriv = current_lataccel - self.prev_lataccel
        target_deriv = target_lataccel - self.prev_target
        
        # Base features
        features = [
            error,                              # 0: error
            target_lataccel / 5.0,              # 1: normalized target
            current_lataccel / 5.0,             # 2: normalized current
            state.roll_lataccel / 5.0,          # 3: normalized roll
            state.v_ego / 30.0,                 # 4: normalized velocity
            state.a_ego / 5.0,                  # 5: normalized acceleration
            self.prev_action,                   # 6: previous action
            error_deriv,                        # 7: error derivative
            target_deriv,                       # 8: target derivative
        ]
        
        # Future plan features (lookahead targets)
        if future_plan and future_plan.lataccel:
            # Take future targets at specific lookahead steps
            lookahead_steps = [0, 2, 5, 10, 20, 40]
            for step in lookahead_steps:
                if step < len(future_plan.lataccel):
                    features.append(future_plan.lataccel[step] / 5.0)
                else:
                    features.append(target_lataccel / 5.0)
        else:
            # Pad with current target if no future plan
            for _ in range(6):
                features.append(target_lataccel / 5.0)
        
        return np.array(features, dtype=np.float32)
    
    def _fallback_pid(self, target_lataccel, current_lataccel, state):
        """Simple PID fallback if neural network not available."""
        error = target_lataccel - current_lataccel
        error_deriv = current_lataccel - self.prev_lataccel
        
        kp, ki, kd = 0.3, 0.05, -0.1
        action = kp * error + kd * error_deriv + 0.15 * target_lataccel
        
        self.prev_lataccel = current_lataccel
        return np.clip(action, -2, 2)


class ControllerNetwork:
    """
    Small neural network for control policy.
    
    This is a pure NumPy implementation for inference,
    but during training we use the PyTorch version.
    """
    
    def __init__(self):
        # Default to PyTorch if available
        try:
            import torch
            import torch.nn as nn
            
            # Input: 15 features, Output: 1 action
            self.net = nn.Sequential(
                nn.Linear(15, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Tanh()  # Output in [-1, 1], scaled to [-2, 2]
            )
            self._is_torch = True
        except ImportError:
            self._is_torch = False
            self.weights = None
    
    def __call__(self, x):
        if self._is_torch:
            return self.net(x).squeeze(-1) * 2.0  # Scale to [-2, 2]
        else:
            raise NotImplementedError("NumPy inference not implemented")
    
    def parameters(self):
        if self._is_torch:
            return self.net.parameters()
        return []
    
    def eval(self):
        if self._is_torch:
            self.net.eval()
    
    def train(self, mode=True):
        if self._is_torch:
            self.net.train(mode)
    
    def state_dict(self):
        if self._is_torch:
            # Return with 'net.' prefix to match training script
            return {'net.' + k: v for k, v in self.net.state_dict().items()}
        return {}
    
    def load_state_dict(self, state_dict):
        if self._is_torch:
            # Handle both formats: with and without 'net.' prefix
            if any(k.startswith('net.') for k in state_dict.keys()):
                # Remove 'net.' prefix
                state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
            self.net.load_state_dict(state_dict)
