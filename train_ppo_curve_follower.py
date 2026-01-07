"""
PPO Reinforcement Learning Controller

Trains a PPO agent to minimize the total cost function:
- lataccel_cost = mean((actual - target)^2) * 100
- jerk_cost = mean((diff(actual)/dt)^2) * 100  
- total_cost = lataccel_cost * 50 + jerk_cost

The agent learns to output steering commands that minimize this cost.
Uses CUDA-accelerated ONNX for the physics model inference.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import onnxruntime as ort
import os
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Constants from tinyphysics
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

# Directory setup
MODEL_SAVE_DIR = Path("models")
DATA_DIR = Path("data")


class LataccelTokenizer:
    """Tokenizer for lateral acceleration values."""
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[np.clip(token, 0, self.vocab_size - 1)]


def load_data_file(data_path):
    """Load data from CSV file."""
    df = pd.read_csv(data_path)
    return pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values
    })


class TinyPhysicsModel:
    """Physics model wrapper with CUDA support."""
    def __init__(self, model_path: str, use_cuda: bool = True):
        self.tokenizer = LataccelTokenizer()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        
        # Setup providers with CUDA priority
        providers = []
        available = ort.get_available_providers()
        
        if use_cuda and 'CUDAExecutionProvider' in available:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
            print("Using CUDA for physics model inference")
        
        providers.append('CPUExecutionProvider')
        
        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, providers)
        
        print(f"Physics model loaded with providers: {self.ort_session.get_providers()}")

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=1.):
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == VOCAB_SIZE
        sample = np.random.choice(probs.shape[2], p=probs[0, -1])
        return sample

    def get_current_lataccel(self, sim_states, actions, past_preds):
        tokenized_actions = self.tokenizer.encode(np.array(past_preds))
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        input_data = {
            'states': np.expand_dims(states, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
        }
        return self.tokenizer.decode(self.predict(input_data, temperature=0.8))


class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic Network.
    
    Input features:
    - Current lateral acceleration error (target - current)
    - Target lateral acceleration
    - Current lateral acceleration
    - Previous lateral acceleration (for jerk awareness)
    - Vehicle state (roll_lataccel, v_ego, a_ego)
    - Previous action
    - Future targets (next few steps)
    
    Total: 13 input features
    """
    def __init__(self, state_dim=13, action_dim=1, hidden_dim=128):
        super(PPOActorCritic, self).__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (outputs mean and log_std for continuous action)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1], scale to STEER_RANGE
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)
    
    def forward(self, state):
        shared_features = self.shared(state)
        return shared_features
    
    def get_action(self, state, deterministic=False):
        shared_features = self.forward(state)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
        
        # Scale from [-1, 1] to STEER_RANGE
        action_scaled = action * (STEER_RANGE[1] - STEER_RANGE[0]) / 2 + (STEER_RANGE[1] + STEER_RANGE[0]) / 2
        action_scaled = torch.clamp(action_scaled, STEER_RANGE[0], STEER_RANGE[1])
        
        return action_scaled, action_mean, action_std
    
    def get_value(self, state):
        shared_features = self.forward(state)
        return self.critic(shared_features)
    
    def evaluate_actions(self, state, action):
        shared_features = self.forward(state)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        # Unscale action back to [-1, 1]
        action_unscaled = (action - (STEER_RANGE[1] + STEER_RANGE[0]) / 2) / ((STEER_RANGE[1] - STEER_RANGE[0]) / 2)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action_unscaled).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        value = self.critic(shared_features)
        
        return value, log_prob, entropy


class SimpleActor(nn.Module):
    """
    Simple actor-only network for REINFORCE.
    No critic - we just use episode cost directly.
    """
    def __init__(self, state_dim=13, action_dim=1, hidden_dim=128):
        super(SimpleActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize with small weights for stable start
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, 0)
    
    def forward(self, state):
        features = self.net(state)
        action_mean = torch.tanh(self.action_mean(features))
        
        # Scale from [-1, 1] to STEER_RANGE
        action_scaled = action_mean * (STEER_RANGE[1] - STEER_RANGE[0]) / 2 + (STEER_RANGE[1] + STEER_RANGE[0]) / 2
        return action_scaled
    
    def get_action(self, state, deterministic=False):
        features = self.net(state)
        action_mean = torch.tanh(self.action_mean(features))
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
        
        # Scale from [-1, 1] to STEER_RANGE
        action_scaled = action * (STEER_RANGE[1] - STEER_RANGE[0]) / 2 + (STEER_RANGE[1] + STEER_RANGE[0]) / 2
        action_scaled = torch.clamp(action_scaled, STEER_RANGE[0], STEER_RANGE[1])
        
        return action_scaled, action_mean, action_std


class RolloutBuffer:
    """Buffer for storing rollout data."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        advantages = []
        returns = []
        gae = 0
        
        values = self.values + [last_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return returns, advantages


class ControlEnv:
    """
    Environment for training the PPO agent.
    Reward is based directly on the total cost function from the README.
    """
    def __init__(self, physics_model, data_dir, num_files=1000):
        self.physics_model = physics_model
        self.data_dir = Path(data_dir)
        self.num_files = num_files
        
        self.data_files = sorted(self.data_dir.glob("*.csv"))[:num_files]
        self.current_file_idx = 0
        self.current_step = 0
        
        self.data = None
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_lataccel_history = []
        self.current_lataccel = 0
        self.prev_lataccel = 0
        self.prev_action = 0
        
    def reset(self, file_idx=None):
        """Reset environment with a new data file."""
        if file_idx is None:
            file_idx = np.random.randint(0, len(self.data_files))
        
        self.current_file_idx = file_idx
        self.data = load_data_file(self.data_files[file_idx])
        
        # Initialize histories
        self.state_history = [State(
            roll_lataccel=self.data.iloc[i]['roll_lataccel'],
            v_ego=self.data.iloc[i]['v_ego'],
            a_ego=self.data.iloc[i]['a_ego']
        ) for i in range(CONTEXT_LENGTH)]
        
        self.action_history = self.data['steer_command'].values[:CONTEXT_LENGTH].tolist()
        self.lataccel_history = self.data['target_lataccel'].values[:CONTEXT_LENGTH].tolist()
        self.target_lataccel_history = self.data['target_lataccel'].values[:CONTEXT_LENGTH].tolist()
        self.current_lataccel = self.lataccel_history[-1]
        self.prev_lataccel = self.lataccel_history[-2] if len(self.lataccel_history) > 1 else self.current_lataccel
        
        # Run through pre-control period
        for step in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state = State(
                roll_lataccel=self.data.iloc[step]['roll_lataccel'],
                v_ego=self.data.iloc[step]['v_ego'],
                a_ego=self.data.iloc[step]['a_ego']
            )
            self.state_history.append(state)
            self.action_history.append(self.data['steer_command'].values[step])
            self.prev_lataccel = self.current_lataccel
            self.current_lataccel = self.data['target_lataccel'].values[step]
            self.lataccel_history.append(self.current_lataccel)
            self.target_lataccel_history.append(self.data['target_lataccel'].values[step])
        
        self.current_step = 0
        self.prev_action = self.action_history[-1]
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation for the agent."""
        global_step = CONTROL_START_IDX + self.current_step
        if global_step >= len(self.data):
            global_step = len(self.data) - 1
        
        target = self.data.iloc[global_step]['target_lataccel']
        state = self.state_history[-1]
        error = target - self.current_lataccel
        
        # Get future targets (next 4 steps for lookahead)
        future_targets = []
        for i in range(1, 5):
            future_step = min(global_step + i, len(self.data) - 1)
            future_targets.append(self.data.iloc[future_step]['target_lataccel'])
        
        obs = np.array([
            error,                              # Current error
            target,                             # Target lateral acceleration
            self.current_lataccel,              # Current actual acceleration
            self.prev_lataccel,                 # Previous acceleration (for jerk)
            state.roll_lataccel,                # Roll lateral acceleration
            state.v_ego / 30.0,                 # Normalized velocity
            state.a_ego / 5.0,                  # Normalized longitudinal acceleration
            self.prev_action,                   # Previous action
            future_targets[0],                  # Next target
            future_targets[1],                  # Target +2
            future_targets[2],                  # Target +3
            future_targets[3],                  # Target +4
            (target - self.prev_lataccel) / DEL_T / 10.0,  # Approximate desired jerk (normalized)
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Execute one step in the environment."""
        action = float(action)
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        
        # Get current state
        global_step = CONTROL_START_IDX + self.current_step
        if global_step >= len(self.data):
            return self._get_observation(), 0, True, {}
        
        target = self.data.iloc[global_step]['target_lataccel']
        
        state = State(
            roll_lataccel=self.data.iloc[global_step]['roll_lataccel'],
            v_ego=self.data.iloc[global_step]['v_ego'],
            a_ego=self.data.iloc[global_step]['a_ego']
        )
        self.state_history.append(state)
        self.action_history.append(action)
        self.target_lataccel_history.append(target)
        
        # Get physics model prediction
        pred = self.physics_model.get_current_lataccel(
            self.state_history[-CONTEXT_LENGTH:],
            self.action_history[-CONTEXT_LENGTH:],
            self.lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        
        self.prev_lataccel = self.current_lataccel
        self.current_lataccel = pred
        self.lataccel_history.append(self.current_lataccel)
        self.prev_action = action
        
        # Compute reward based on the total cost function
        # lataccel_cost component: (actual - target)^2 * 100
        # jerk_cost component: ((actual_t - actual_{t-1}) / dt)^2 * 100
        # total_cost = lataccel_cost * 50 + jerk_cost
        
        lataccel_error_sq = (self.current_lataccel - target) ** 2
        jerk_sq = ((self.current_lataccel - self.prev_lataccel) / DEL_T) ** 2
        
        # Step cost (not averaged, raw contribution to total)
        step_lataccel_cost = lataccel_error_sq * 100
        step_jerk_cost = jerk_sq * 100
        step_total_cost = step_lataccel_cost * LAT_ACCEL_COST_MULTIPLIER + step_jerk_cost
        
        # Simple, direct reward: negative cost, normalized to reasonable range
        # Typical step costs range from ~10 (good) to ~5000 (bad)
        # Normalize by dividing by 1000 to get rewards in [-5, 0] range
        reward = -step_total_cost / 1000.0
        reward = np.clip(reward, -5.0, 0.0)
        
        # Bonus for good tracking (smaller error = bigger bonus)
        if abs(self.current_lataccel - target) < 0.5:
            reward += 0.1 * (0.5 - abs(self.current_lataccel - target))
        
        self.current_step += 1
        done = self.current_step >= (COST_END_IDX - CONTROL_START_IDX)
        
        # Compute episode cost at end
        info = {}
        if done:
            info = self._compute_episode_cost()
        
        return self._get_observation(), reward, done, info
    
    def _compute_episode_cost(self):
        """Compute the full episode cost as in the README."""
        target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        
        lataccel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = lataccel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        
        return {
            'lataccel_cost': lataccel_cost,
            'jerk_cost': jerk_cost,
            'total_cost': total_cost
        }


class RunningMeanStd:
    """Running mean and std for reward normalization."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class PPOTrainer:
    """PPO Trainer for the control agent."""
    def __init__(
        self,
        env,
        model,
        lr=5e-5,           # Lower actor LR to prevent collapse
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,  # Back to standard clip
        value_coef=0.5,
        entropy_coef=0.02,  # Higher constant entropy - NO DECAY
        max_grad_norm=0.5,
        update_epochs=3,    # Even fewer epochs
        batch_size=256,     # Even larger batch for stability
        device='cuda' if torch.cuda.is_available() else 'cpu',
        live_plot=False
    ):
        self.env = env
        self.model = model.to(device)
        self.device = device
        
        # Separate learning rates: critic can learn faster
        actor_params = list(model.shared.parameters()) + list(model.actor_mean.parameters()) + [model.actor_log_std]
        critic_params = list(model.critic.parameters())
        self.optimizer = optim.Adam([
            {'params': actor_params, 'lr': lr},
            {'params': critic_params, 'lr': lr * 3}  # Critic learns 3x faster
        ], eps=1e-5)
        
        # No LR scheduling - keep it constant
        self.scheduler = None
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()
        self.episode_costs = []
        # Live plotting
        self.live_plot = live_plot
        self.train_losses = []
        self.avg_costs = []
        if self.live_plot:
            try:
                plt.ion()
                self._fig, self._ax1 = plt.subplots(1, 1, figsize=(10, 4))
                
                # Left axis for loss (smaller scale)
                self._ax1.set_xlabel('update')
                self._ax1.set_ylabel('loss', color='tab:blue')
                self._loss_line, = self._ax1.plot([], [], 'b-', alpha=0.5, label='loss')
                self._loss_trend_line, = self._ax1.plot([], [], 'b-', linewidth=2, label='loss trend')
                self._ax1.tick_params(axis='y', labelcolor='tab:blue')
                
                # Right axis for cost (larger scale)
                self._ax2 = self._ax1.twinx()
                self._ax2.set_ylabel('avg_cost', color='tab:orange')
                self._cost_line, = self._ax2.plot([], [], 'o-', color='tab:orange', alpha=0.5, markersize=2, label='avg_cost')
                self._cost_trend_line, = self._ax2.plot([], [], '-', color='tab:orange', linewidth=2, label='cost trend')
                self._ax2.tick_params(axis='y', labelcolor='tab:orange')
                
                # Combined legend
                lines = [self._loss_line, self._loss_trend_line, self._cost_line, self._cost_trend_line]
                labels = ['loss', 'loss trend (50)', 'avg_cost', 'cost trend (50)']
                self._ax1.legend(lines, labels, loc='upper right')
                
                self._fig.tight_layout()
                self._plot_started = True
                plt.show(block=False)
            except Exception:
                self.live_plot = False
                self._plot_started = False
    
    def pretrain_with_pid(self, num_episodes=50, epochs=10):
        """
        Pre-train the policy network using behavior cloning from PID controller.
        This gives the RL a good starting point instead of random initialization.
        """
        print(f"\n=== Phase 1: Behavior Cloning from PID ({num_episodes} episodes, {epochs} epochs) ===")
        
        # Simple PID controller (same gains as controllers/pid.py)
        pid_p, pid_i, pid_d = 0.195, 0.100, -0.053
        
        # Collect demonstrations
        all_states = []
        all_actions = []
        
        for ep in tqdm(range(num_episodes), desc="Collecting PID demonstrations"):
            obs = self.env.reset()
            error_integral = 0
            prev_error = 0
            
            while True:
                # Extract target and current from observation
                # obs[0] = error, obs[1] = target, obs[2] = current
                error = obs[0]
                
                # PID control
                error_integral += error
                error_diff = error - prev_error
                prev_error = error
                
                pid_action = pid_p * error + pid_i * error_integral + pid_d * error_diff
                pid_action = np.clip(pid_action, STEER_RANGE[0], STEER_RANGE[1])
                
                all_states.append(obs.copy())
                all_actions.append(pid_action)
                
                obs, _, done, _ = self.env.step(pid_action)
                if done:
                    break
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions = torch.FloatTensor(np.array(all_actions)).unsqueeze(-1).to(self.device)
        
        # Scale actions to [-1, 1] for network output comparison
        actions_scaled = (actions - (STEER_RANGE[1] + STEER_RANGE[0]) / 2) / ((STEER_RANGE[1] - STEER_RANGE[0]) / 2)
        
        print(f"Collected {len(all_states)} state-action pairs")
        
        # Train with supervised learning
        bc_optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        dataset_size = len(states)
        batch_size = 256
        
        for epoch in range(epochs):
            indices = np.random.permutation(dataset_size)
            total_loss = 0
            num_batches = 0
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions_scaled[batch_idx]
                
                # Get actor's mean output (deterministic action)
                shared = self.model.shared(batch_states)
                pred_actions = self.model.actor_mean(shared)
                
                # MSE loss between predicted and PID actions
                loss = nn.functional.mse_loss(pred_actions, batch_actions)
                
                bc_optimizer.zero_grad()
                loss.backward()
                bc_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1}/{epochs}: BC Loss = {avg_loss:.6f}")
        
        # Evaluate PID baseline cost
        print("\nEvaluating PID baseline...")
        pid_costs = []
        for _ in range(10):
            obs = self.env.reset()
            error_integral = 0
            prev_error = 0
            while True:
                error = obs[0]
                error_integral += error
                error_diff = error - prev_error
                prev_error = error
                pid_action = np.clip(pid_p * error + pid_i * error_integral + pid_d * error_diff, 
                                     STEER_RANGE[0], STEER_RANGE[1])
                obs, _, done, info = self.env.step(pid_action)
                if done:
                    if 'total_cost' in info:
                        pid_costs.append(info['total_cost'])
                    break
        
        print(f"PID baseline cost: {np.mean(pid_costs):.2f} ± {np.std(pid_costs):.2f}")
        print("=== Behavior cloning complete! Starting RL fine-tuning ===\n")
    
    def collect_rollout(self, num_steps):
        """Collect experience by running the policy."""
        self.model.eval()
        
        obs = self.env.reset()
        episode_rewards = []
        
        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, _, _ = self.model.get_action(obs_tensor)
                value = self.model.get_value(obs_tensor)
                
                _, action_mean, action_std = self.model.get_action(obs_tensor)
                action_unscaled = (action - (STEER_RANGE[1] + STEER_RANGE[0]) / 2) / ((STEER_RANGE[1] - STEER_RANGE[0]) / 2)
                dist = Normal(action_mean, action_std)
                log_prob = dist.log_prob(action_unscaled).sum(-1, keepdim=True)
            
            next_obs, reward, done, info = self.env.step(action.cpu().numpy()[0, 0])
            episode_rewards.append(reward)
            
            self.buffer.add(
                obs,
                action.cpu().numpy()[0, 0],
                reward,
                value.cpu().numpy()[0, 0],
                log_prob.cpu().numpy()[0, 0],
                done
            )
            
            obs = next_obs
            
            if done:
                if 'total_cost' in info:
                    self.episode_costs.append(info['total_cost'])
                obs = self.env.reset()
                episode_rewards = []
        
        # Get last value for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.model.get_value(obs_tensor).cpu().numpy()[0, 0]
        
        return last_value
    
    def update(self, last_value):
        """Update policy using collected rollouts."""
        self.model.train()
        
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).unsqueeze(-1).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).unsqueeze(-1).to(self.device)
        old_values = torch.FloatTensor(np.array(self.buffer.values)).unsqueeze(-1).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).unsqueeze(-1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                values, log_probs, entropy = self.model.evaluate_actions(batch_states, batch_actions)
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (helps stability)
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (values - batch_returns) ** 2
                value_loss_clipped = (values_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        self.buffer.clear()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def train(self, total_timesteps, rollout_length=2048, log_interval=10):
        """Train the agent."""
        num_updates = total_timesteps // rollout_length
        
        print(f"Training PPO for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Device: {self.device}")
        
        best_cost = float('inf')
        
        # Use a tqdm iterator so we can set a live postfix with metrics
        progress = tqdm(range(num_updates), desc="Training")
        for update in progress:
            last_value = self.collect_rollout(rollout_length)
            metrics = self.update(last_value)

            # Get average cost from recent episodes
            if self.episode_costs:
                recent_costs = self.episode_costs[-10:]
                avg_cost = np.mean(recent_costs)
            else:
                avg_cost = float('inf')

            # Update tqdm postfix with compact metrics
            postfix = {
                'loss': f"{metrics['loss']:.4f}",
                'policy': f"{metrics['policy_loss']:.4f}",
                'value': f"{metrics['value_loss']:.4f}",
                'ent': f"{metrics['entropy']:.4f}",
                'avg_cost': f"{avg_cost:.2f}"
            }
            progress.set_postfix(postfix)

            # Save best model
            if avg_cost < best_cost:
                best_cost = avg_cost
                self.save_model("ppo_controller_best")
            # Update live plot if enabled
            if self.live_plot:
                self.train_losses.append(metrics['loss'])
                self.avg_costs.append(avg_cost if np.isfinite(avg_cost) else None)
                try:
                    x = list(range(len(self.train_losses)))
                    
                    # Raw loss data
                    self._loss_line.set_data(x, self.train_losses)
                    
                    # Loss trend line (moving average, window=50)
                    window = min(50, len(self.train_losses))
                    if len(self.train_losses) >= window:
                        loss_trend = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                        x_trend = list(range(window - 1, len(self.train_losses)))
                        self._loss_trend_line.set_data(x_trend, loss_trend)
                    
                    # Raw cost data
                    y_cost = [v if v is not None else np.nan for v in self.avg_costs]
                    self._cost_line.set_data(x, y_cost)
                    
                    # Cost trend line (moving average, window=50)
                    if len(self.avg_costs) >= window:
                        valid_costs = [v if v is not None else np.nan for v in self.avg_costs]
                        # Use pandas-style rolling mean that handles NaN
                        cost_arr = np.array(valid_costs)
                        cost_trend = np.convolve(np.nan_to_num(cost_arr, nan=np.nanmean(cost_arr)), 
                                                  np.ones(window)/window, mode='valid')
                        x_trend = list(range(window - 1, len(self.avg_costs)))
                        self._cost_trend_line.set_data(x_trend, cost_trend)
                    
                    self._ax1.relim()
                    self._ax1.autoscale_view()
                    self._ax2.relim()
                    self._ax2.autoscale_view()
                    self._fig.canvas.draw()
                    plt.pause(0.001)  # Small pause to allow GUI to update
                except Exception:
                    pass
        
        print(f"\nTraining complete! Best average cost: {best_cost:.2f}")
        return self.model
    
    def save_model(self, name):
        """Save model checkpoint."""
        MODEL_SAVE_DIR.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, MODEL_SAVE_DIR / f"{name}.pt")


def export_actor_to_onnx(model, save_path, device):
    """Export the actor model to ONNX for the controller."""
    model.eval()
    
    class ActorWrapper(nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.shared = actor_critic.shared
            self.actor_mean = actor_critic.actor_mean
        
        def forward(self, state):
            shared_features = self.shared(state)
            action_mean = self.actor_mean(shared_features)
            action_scaled = action_mean * (STEER_RANGE[1] - STEER_RANGE[0]) / 2 + (STEER_RANGE[1] + STEER_RANGE[0]) / 2
            return action_scaled
    
    wrapper = ActorWrapper(model).to(device)
    wrapper.eval()
    
    dummy_input = torch.randn(1, 13).to(device)
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print(f"Exported actor model to {save_path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO controller')
    parser.add_argument('--num_files', type=int, default=1000, help='Number of data files to use')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models/tinyphysics.onnx', help='Physics model path')
    parser.add_argument('--cpu_only', action='store_true', help='Disable all CUDA and force CPU-only execution')
    parser.add_argument('--onnx_cuda', action='store_true', help='Force ONNX runtime to use CUDAExecutionProvider even if PyTorch is CPU')
    parser.add_argument('--onnx_cpu', action='store_true', help='Force ONNX runtime to use CPUExecutionProvider (overrides --onnx_cuda)')
    parser.add_argument('--live_plot', action='store_true', help='Show live training loss plot')
    parser.add_argument('--pretrain_pid', type=int, default=0, help='Number of episodes to pretrain with PID behavior cloning (0=disabled)')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Number of epochs for PID pretraining')
    args = parser.parse_args()
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.cpu_only:
        device = 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Setup environment
    if args.cpu_only:
        use_onnx_cuda = False
    elif args.onnx_cpu:
        use_onnx_cuda = False
    else:
        use_onnx_cuda = args.onnx_cuda or (device == 'cuda')

    print("Loading physics model... (use_onnx_cuda=", use_onnx_cuda, ")")
    physics_model = TinyPhysicsModel(args.model_path, use_cuda=use_onnx_cuda)
    
    print("Creating environment...")
    env = ControlEnv(physics_model, DATA_DIR, args.num_files)
    
    # Create simple actor-only model
    print("Creating actor model...")
    model = SimpleActor(state_dim=13, action_dim=1, hidden_dim=128).to(device)
    
    # Simple REINFORCE training
    print(f"\n=== REINFORCE Training ({args.episodes} episodes) ===")
    print("No critic - just episode cost as the signal")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Running baseline (average cost so far)
    baseline_cost = None
    baseline_alpha = 0.1  # EMA smoothing factor
    
    episode_costs = []
    best_cost = float('inf')
    
    # Setup live plot
    if args.live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
        cost_line, = ax.plot([], [], 'b-', alpha=0.5, label='episode cost')
        baseline_line, = ax.plot([], [], 'r-', linewidth=2, label='baseline (EMA)')
        ax.set_xlabel('episode')
        ax.set_ylabel('total cost')
        ax.legend()
        plt.show(block=False)
    
    progress = tqdm(range(args.episodes), desc="Training")
    for ep in progress:
        # Run one episode, collecting log probs
        obs = env.reset()
        log_probs = []
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, action_mean, action_std = model.get_action(obs_t)
            
            # Compute log prob for this action
            action_unscaled = (action - (STEER_RANGE[1] + STEER_RANGE[0]) / 2) / ((STEER_RANGE[1] - STEER_RANGE[0]) / 2)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action_unscaled).sum()
            log_probs.append(log_prob)
            
            obs, _, done, info = env.step(action.cpu().numpy()[0, 0])
            if done:
                break
        
        # Get episode cost
        episode_cost = info.get('total_cost', float('inf'))
        episode_costs.append(episode_cost)
        
        # Update baseline
        if baseline_cost is None:
            baseline_cost = episode_cost
        else:
            baseline_cost = baseline_alpha * episode_cost + (1 - baseline_alpha) * baseline_cost
        
        # Advantage: negative because we want to MINIMIZE cost
        # If cost < baseline, advantage is positive (good episode, increase prob)
        # If cost > baseline, advantage is negative (bad episode, decrease prob)
        advantage = -(episode_cost - baseline_cost) / (baseline_cost + 1e-8)
        advantage = np.clip(advantage, -2.0, 2.0)  # Clip for stability
        
        # REINFORCE update: increase prob of actions that led to lower cost
        # Need gradients now
        obs = env.reset()
        log_probs_grad = []
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, action_mean, action_std = model.get_action(obs_t)
            
            action_unscaled = (action - (STEER_RANGE[1] + STEER_RANGE[0]) / 2) / ((STEER_RANGE[1] - STEER_RANGE[0]) / 2)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action_unscaled).sum()
            log_probs_grad.append(log_prob)
            
            obs, _, done, _ = env.step(action.detach().cpu().numpy()[0, 0])
            if done:
                break
        
        # Policy gradient loss
        policy_loss = 0
        for lp in log_probs_grad:
            policy_loss -= lp * advantage  # Negative because we minimize loss
        policy_loss = policy_loss / len(log_probs_grad)
        
        # Add entropy bonus for exploration
        entropy = dist.entropy().mean()
        loss = policy_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Save best
        if episode_cost < best_cost:
            best_cost = episode_cost
            MODEL_SAVE_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_DIR / "reinforce_best.pt")
        
        # Update progress
        progress.set_postfix({
            'cost': f"{episode_cost:.1f}",
            'baseline': f"{baseline_cost:.1f}",
            'best': f"{best_cost:.1f}",
            'adv': f"{advantage:.3f}"
        })
        
        # Update plot
        if args.live_plot and ep % 5 == 0:
            try:
                cost_line.set_data(range(len(episode_costs)), episode_costs)
                # Compute running baseline for plot
                baselines = []
                b = episode_costs[0]
                for c in episode_costs:
                    b = baseline_alpha * c + (1 - baseline_alpha) * b
                    baselines.append(b)
                baseline_line.set_data(range(len(baselines)), baselines)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)
            except:
                pass
    
    print(f"\nTraining complete! Best cost: {best_cost:.2f}")
    
    # Save final model
    torch.save(model.state_dict(), MODEL_SAVE_DIR / "reinforce_final.pt")
    
    # Export to ONNX
    dummy_input = torch.randn(1, 13).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_SAVE_DIR / "reinforce_controller.onnx",
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={'state': {0: 'batch'}, 'action': {0: 'batch'}}
    )
    
    print(f"Model saved to: {MODEL_SAVE_DIR / 'reinforce_controller.onnx'}")


if __name__ == "__main__":
    main()
