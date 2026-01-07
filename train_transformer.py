"""
Train a transformer model to predict steering commands.
Uses the ground truth data from the first 1 second (100 steps) of each segment.
Validates using the actual simulator on held-out files.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import tempfile
import importlib
import copy

# Constants from tinyphysics.py
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
DEL_T = 0.1

# Transformer model hyperparameters
CONTEXT_LENGTH = 10  # Use last 1.0s of data (10 steps at 10 FPS)
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100

# DataLoader worker count (use half the CPUs by default)
NUM_WORKERS = max(1, os.cpu_count() // 2)

# Input features: v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, steer_command
NUM_FEATURES = 6


class SteerDataset(Dataset):
    """Dataset for training the steering transformer."""
    
    def __init__(self, data_dir: str = None, max_files: int = None):
        self.sequences = []
        self.targets = []
        
        # Allow empty init (for manual file loading)
        if data_dir is None:
            return
        
        data_path = Path(data_dir)
        files = sorted(data_path.glob("*.csv"))
        if max_files:
            files = files[:max_files]
        
        print(f"Loading {len(files)} data files...")
        for file in tqdm(files):
            self._load_file(file)
        
        # Convert to torch tensors once to avoid repeated allocations in __getitem__
        self.sequences = torch.from_numpy(np.array(self.sequences, dtype=np.float32))
        self.targets = torch.from_numpy(np.array(self.targets, dtype=np.float32))
        print(f"Loaded {len(self.sequences)} training samples")
    
    def _load_file(self, file_path: Path):
        """Load a single CSV file and extract training sequences."""
        df = pd.read_csv(file_path)
        
        # Process data similar to tinyphysics.py
        roll_lataccel = np.sin(df['roll'].values) * ACC_G
        v_ego = df['vEgo'].values
        a_ego = df['aEgo'].values
        target_lataccel = df['targetLateralAcceleration'].values
        steer_command = -df['steerCommand'].values  # Right-positive convention
        
        # For the first 100 steps, current_lataccel = target_lataccel (ground truth)
        # We use this as our training data
        current_lataccel = target_lataccel.copy()
        
        # Normalize features for better training
        # Stack features: [v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, steer_command]
        features = np.stack([
            v_ego / 40.0,  # Normalize speed (typical ~30-40 m/s)
            a_ego / 5.0,   # Normalize acceleration
            roll_lataccel / 5.0,  # Normalize roll lateral accel
            target_lataccel / 5.0,  # Normalize target
            current_lataccel / 5.0,  # Normalize current
            steer_command / 2.0,  # Normalize steer (range is -2 to 2)
        ], axis=1)
        
        # Create sequences: use steps from CONTEXT_LENGTH to CONTROL_START_IDX
        # For each step, use the previous CONTEXT_LENGTH steps to predict the next steer command
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            seq = features[i - CONTEXT_LENGTH:i]  # Past CONTEXT_LENGTH steps
            target = steer_command[i] / 2.0  # Normalized target steer command
            self.sequences.append(seq)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Items are already tensors (on CPU). DataLoader with pin_memory=True
        # will speed up transfer to CUDA when using non_blocking=True.
        return self.sequences[idx], self.targets[idx]


class SteerTransformer(nn.Module):
    """Transformer model for predicting steering commands."""
    
    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        context_length: int = CONTEXT_LENGTH,
        dropout: float = DROPOUT,
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


class CombinedSteerLoss(nn.Module):
    """Custom loss combining supervised MSE on steer and a steer-jerk penalty.

    This is a differentiable proxy for the simulator's lataccel+jerk total_cost.
    It uses the previous (normalized) steer from the input sequence to compute
    a simple jerk term: ((pred_steer - prev_steer) / DEL_T)^2.
    """
    def __init__(self, mse_weight: float = 1.0, jerk_weight: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.jerk_weight = jerk_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, prev_steer_norm: torch.Tensor):
        # pred, target: (batch,)
        mse_loss = self.mse(pred, target)

        # convert normalized steer back to physical units (steer range is [-2, 2])
        pred_steer = pred * 2.0
        prev_steer = prev_steer_norm * 2.0

        # approximate jerk in steer (delta steer / dt)
        jerk = ((pred_steer - prev_steer) / DEL_T) ** 2
        jerk_loss = jerk.mean() * 100.0

        return (self.mse_weight * mse_loss) + (self.jerk_weight * jerk_loss)


def validate_with_simulator(model, save_path: str, val_files: list, model_path: str, device: torch.device) -> float:
    """
    Validate the model by running the full simulator on held-out files.
    
    Args:
        model: The transformer model to validate
        save_path: Path to temporarily save the model for the controller
        val_files: List of validation file paths
        model_path: Path to tinyphysics.onnx model
        device: torch device
    
    Returns:
        Average total_cost across validation files
    """
    # Save current model state to a temp file so the controller can load it
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': NUM_FEATURES,
            'hidden_dim': HIDDEN_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'context_length': CONTEXT_LENGTH,
            'dropout': DROPOUT,
        }
    }, save_path)
    
    # Import tinyphysics components
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    # Reload the transformer controller module to pick up the new model
    import controllers.transformer
    importlib.reload(controllers.transformer)
    
    # Create the physics model once
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    total_costs = []
    for val_file in val_files:
        # Create a fresh controller for each file
        controller = controllers.transformer.Controller(model_path=save_path)
        
        # Run simulation
        sim = TinyPhysicsSimulator(tinyphysics_model, str(val_file), controller=controller, debug=False)
        cost = sim.rollout()
        total_costs.append(cost['total_cost'])
    
    avg_cost = np.mean(total_costs)
    return avg_cost


def train_model(data_dir: str, save_path: str, max_files: int = None, 
                physics_model_path: str = "./models/tinyphysics.onnx",
                num_val_files: int = 3):
    """Train the transformer model with simulator-based validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Improve performance for small models / repeated transfers
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    
    # Get all data files and split into train/val by file (not by sequence)
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob("*.csv"))
    if max_files:
        all_files = all_files[:max_files]
    
    # Reserve last num_val_files for validation
    val_files = all_files[-num_val_files:]
    train_files = all_files[:-num_val_files]
    
    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files")
    print(f"Validation files: {[f.name for f in val_files]}")
    
    # Create dataset using only training files
    dataset = SteerDataset(data_dir=None, max_files=None)
    dataset.sequences = []
    dataset.targets = []
    
    print(f"Loading {len(train_files)} training files...")
    for file in tqdm(train_files):
        dataset._load_file(file)
    
    # Convert to tensors
    dataset.sequences = torch.from_numpy(np.array(dataset.sequences, dtype=np.float32))
    dataset.targets = torch.from_numpy(np.array(dataset.targets, dtype=np.float32))
    print(f"Loaded {len(dataset.sequences)} training samples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    # Create model
    model = SteerTransformer().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CombinedSteerLoss(mse_weight=1.0, jerk_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Ensure save directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    best_val_cost = float('inf')
    last_val_cost = None
    RERUN_MAX = 3

    for epoch in range(NUM_EPOCHS):
        attempt = 0
        while True:
            attempt += 1
            # Save state before epoch so we can restore if validation spikes
            saved_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            saved_opt_state = copy.deepcopy(optimizer.state_dict())

            # Training
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad()
                pred = model(batch_x)
                # previous steer is the last timestep's steer feature (normalized)
                prev_steer = batch_x[:, -1, 5]
                loss = criterion(pred, batch_y, prev_steer)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # Validate using actual simulator every epoch
            model.eval()
            val_cost = validate_with_simulator(
                model, save_path, val_files, physics_model_path, device
            )

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} (attempt {attempt}): Train Loss: {train_loss:.4f}, Val Cost (sim): {val_cost:.2f}")

            # If this validation cost is catastrophically worse than last epoch's cost,
            # restore and retry the epoch (up to RERUN_MAX attempts).
            if last_val_cost is not None and val_cost > 4.0 * last_val_cost and attempt <= RERUN_MAX:
                print(f"  Val cost {val_cost:.2f} > 10x last val cost {last_val_cost:.2f}, restoring and retrying epoch (attempt {attempt}/{RERUN_MAX})")
                # restore model and optimizer
                model.load_state_dict({k: v.to(device) for k, v in saved_model_state.items()})
                optimizer.load_state_dict(copy.deepcopy(saved_opt_state))
                # small lr backoff to stabilize repeated failures
                for g in optimizer.param_groups:
                    g['lr'] *= 0.9
                continue

            # Accept this epoch's validation cost
            break

        # update last_val_cost and best
        last_val_cost = val_cost

        # Save best model based on simulator cost
        if val_cost < best_val_cost:
            best_val_cost = val_cost
            # Model was already saved in validate_with_simulator, just log it
            print(f"  Saved best model with val_cost: {val_cost:.2f}")
    
    print(f"\nTraining complete! Best validation cost: {best_val_cost:.2f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train steering transformer")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--save_path", type=str, default="./models/steer_transformer.pt", help="Path to save model")
    parser.add_argument("--max_files", type=int, default=None, help="Max number of files to use (for testing)")
    args = parser.parse_args()
    
    train_model(args.data_dir, args.save_path, args.max_files)
