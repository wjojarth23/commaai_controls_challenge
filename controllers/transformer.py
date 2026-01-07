"""
Transformer-based controller for steering prediction.
Uses a trained transformer model to predict steering commands.
"""

from . import BaseController
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# Constants (must match training)
ACC_G = 9.81
CONTEXT_LENGTH = 10  # Last 1.0s of data at 10 FPS
NUM_FEATURES = 6


class SteerTransformer(nn.Module):
    """Transformer model for predicting steering commands."""
    
    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        context_length: int = CONTEXT_LENGTH,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, context_length, hidden_dim) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, context_length, input_dim)
        Returns:
            Predicted steering command of shape (batch_size,)
        """
        # Project input to hidden dimension
        x = self.input_proj(x)  # (batch, seq, hidden)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq, hidden)
        
        # Use the last timestep for prediction
        x = x[:, -1, :]  # (batch, hidden)
        
        # Output prediction
        out = self.output_head(x).squeeze(-1)  # (batch,)
        
        return out


class Controller(BaseController):
    """
    Transformer-based controller that predicts steering commands
    based on recent history of states and actions.
    """
    
    def __init__(self, model_path: str = None):
        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "steer_transformer.pt"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        self.model = SteerTransformer(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            context_length=config['context_length'],
            dropout=0.0,  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.context_length = config['context_length']
        
        # History buffers
        self.v_ego_history = []
        self.a_ego_history = []
        self.roll_lataccel_history = []
        self.target_lataccel_history = []
        self.current_lataccel_history = []
        self.steer_history = []
        
        # For smoothing output
        self.prev_steer = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Predict the next steering command using the transformer model.
        
        Args:
            target_lataccel: The target lateral acceleration.
            current_lataccel: The current lateral acceleration.
            state: The current state (roll_lataccel, v_ego, a_ego).
            future_plan: Future plan containing upcoming targets.
        
        Returns:
            Steering command.
        """
        # Update history
        self.v_ego_history.append(state.v_ego)
        self.a_ego_history.append(state.a_ego)
        self.roll_lataccel_history.append(state.roll_lataccel)
        self.target_lataccel_history.append(target_lataccel)
        self.current_lataccel_history.append(current_lataccel)
        
        # If we don't have enough history, use a simple proportional control
        if len(self.v_ego_history) < self.context_length:
            # Simple fallback: proportional control
            error = target_lataccel - current_lataccel
            steer = error * 0.3
            self.steer_history.append(steer)
            self.prev_steer = steer
            return steer
        
        # Build input sequence
        seq_features = []
        for i in range(-self.context_length, 0):
            features = [
                self.v_ego_history[i] / 40.0,
                self.a_ego_history[i] / 5.0,
                self.roll_lataccel_history[i] / 5.0,
                self.target_lataccel_history[i] / 5.0,
                self.current_lataccel_history[i] / 5.0,
                self.steer_history[i] / 2.0 if len(self.steer_history) >= -i else 0.0,
            ]
            seq_features.append(features)
        
        # Convert to tensor
        x = torch.tensor([seq_features], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x).item()
        
        # Denormalize prediction
        steer = pred * 2.0
        
        # Smooth the output to reduce jerk
        alpha = 0.7  # Smoothing factor
        steer = alpha * steer + (1 - alpha) * self.prev_steer
        
        # Clip to valid range
        steer = np.clip(steer, -2.0, 2.0)
        
        # Store for next iteration
        self.steer_history.append(steer)
        self.prev_steer = steer
        
        return steer
