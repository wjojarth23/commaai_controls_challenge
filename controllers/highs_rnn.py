"""
HiGHS-Optimized Encoder-Decoder RNN Controller

This controller uses an encoder-decoder RNN architecture trained on optimal
control sequences computed using the HiGHS optimization solver.

Architecture:
- Encoder: GRU that processes the history of (state, action) pairs
- Decoder: GRU that generates the next action given target acceleration
           and the encoded context from the past

The model was trained on optimal sequences where HiGHS solved:
    minimize: sum((target - pred)^2) * LAT_ACCEL_COST_MULTIPLIER + sum((jerk)^2)
    subject to: physics dynamics constraints, control limits

This allows the neural network to learn a policy that approximates the
globally optimal solution without running expensive optimization at inference time.
"""

from . import BaseController
import numpy as np
import os

# Path to the trained model weights
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'highs_rnn_controller.pth')

# Constants
CONTEXT_LENGTH = 20


class EncoderDecoderRNN:
    """PyTorch-free inference wrapper for the encoder-decoder RNN."""
    
    def __init__(self, weights_dict, hidden_dim=128, num_layers=2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.weights = weights_dict
        self.hidden = None
    
    def reset(self):
        self.hidden = None


class Controller(BaseController):
    """
    Encoder-Decoder RNN controller trained on HiGHS-optimized sequences.
    """
    
    def __init__(self):
        self.model = None
        self.device = 'cpu'
        self._load_model()
        
        # History buffers
        self.state_history = []
        self.action_history = []
        self.target_history = []
        self.hidden = None
        self.step_count = 0
        
        # For fallback
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def _load_model(self):
        """Load the trained PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            
            # Define the encoder-decoder architecture (must match training)
            class EncoderDecoderRNNModule(nn.Module):
                def __init__(self, state_dim=5, action_dim=1, hidden_dim=128, num_layers=2, dropout=0.1):
                    super().__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    
                    self.state_proj = nn.Linear(state_dim, hidden_dim // 2)
                    self.action_proj = nn.Linear(action_dim, hidden_dim // 2)
                    
                    self.encoder = nn.GRU(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0
                    )
                    
                    self.decoder_input = nn.Linear(hidden_dim + 1, hidden_dim)
                    
                    self.decoder = nn.GRU(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0
                    )
                    
                    self.output_proj = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, action_dim),
                        nn.Tanh()
                    )
                
                def forward(self, past_states, past_actions, target_accel, hidden=None):
                    batch_size = past_states.shape[0]
                    
                    state_emb = self.state_proj(past_states)
                    action_emb = self.action_proj(past_actions)
                    encoder_input = torch.cat([state_emb, action_emb], dim=-1)
                    
                    encoder_output, encoder_hidden = self.encoder(encoder_input, hidden)
                    context = encoder_output[:, -1:, :]
                    
                    target_accel = target_accel.unsqueeze(1)
                    decoder_in = torch.cat([context, target_accel], dim=-1)
                    decoder_in = self.decoder_input(decoder_in)
                    
                    decoder_output, decoder_hidden = self.decoder(decoder_in, encoder_hidden)
                    action = self.output_proj(decoder_output.squeeze(1)) * 2.0
                    
                    return action, decoder_hidden
            
            self.model = EncoderDecoderRNNModule()
            
            if os.path.exists(MODEL_WEIGHTS_PATH):
                self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
                print(f"HiGHS-RNN: Loaded trained weights from {MODEL_WEIGHTS_PATH}")
            else:
                print(f"HiGHS-RNN: No trained weights found at {MODEL_WEIGHTS_PATH}, using random init")
                print("  Run 'python train_highs_rnn.py' to train the controller")
            
            self.model.eval()
            
        except ImportError:
            print("HiGHS-RNN: PyTorch not available, falling back to simple PID")
            self.model = None
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute control action using the trained encoder-decoder RNN.
        """
        self.step_count += 1
        
        # Build state features
        state_feat = [
            target_lataccel,                # target lataccel
            current_lataccel,               # current lataccel
            state.roll_lataccel,            # roll_lataccel
            state.v_ego / 30.0,             # v_ego normalized
            state.a_ego / 5.0,              # a_ego normalized
        ]
        
        self.state_history.append(state_feat)
        self.target_history.append(target_lataccel)
        
        # If model not loaded, use fallback
        if self.model is None:
            return self._fallback_pid(target_lataccel, current_lataccel, state)
        
        # Need enough history for encoder
        if len(self.state_history) < CONTEXT_LENGTH:
            # Use simple proportional control during warmup
            action = 0.15 * (target_lataccel - current_lataccel) + 0.1 * target_lataccel
            self.action_history.append(action)
            return np.clip(action, -2, 2)
        
        # Prepare input tensors
        past_states = np.array(self.state_history[-CONTEXT_LENGTH:], dtype=np.float32)
        past_actions = np.array(self.action_history[-CONTEXT_LENGTH:], dtype=np.float32).reshape(-1, 1)
        target_accel = np.array([target_lataccel], dtype=np.float32)
        
        with self.torch.no_grad():
            past_states_t = self.torch.tensor(past_states).unsqueeze(0)
            past_actions_t = self.torch.tensor(past_actions).unsqueeze(0)
            target_accel_t = self.torch.tensor(target_accel).unsqueeze(0)
            
            action, self.hidden = self.model(past_states_t, past_actions_t, target_accel_t, self.hidden)
            action = action.squeeze().item()
        
        # Clip to valid range
        action = np.clip(action, -2, 2)
        self.action_history.append(action)
        
        return action
    
    def _fallback_pid(self, target_lataccel, current_lataccel, state):
        """Fallback PID controller when model is not available."""
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # PID gains
        kp = 0.17
        ki = 0.1
        kd = -0.06
        kff = 0.13
        
        action = kp * error + ki * self.error_integral + kd * error_diff + kff * target_lataccel
        self.action_history.append(action)
        return np.clip(action, -2, 2)
