"""
Quick gain optimization for the controller.
Uses simple random search / hill climbing to find better gains.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
from pathlib import Path

# Test on a few representative segments
TEST_FILES = [f"data/{i:05d}.csv" for i in range(20)]

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

class TunableController:
    def __init__(self, params):
        self.kp = params[0]
        self.ki = params[1]
        self.kd = params[2]
        self.kff = params[3]
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        fb = self.kp * error + self.ki * self.error_integral + self.kd * error_diff
        ff = self.kff * target_lataccel
        return fb + ff

def evaluate(params, files=TEST_FILES[:10]):
    total_cost = 0
    for f in files:
        controller = TunableController(params)
        sim = TinyPhysicsSimulator(model, f, controller=controller, debug=False)
        cost = sim.rollout()
        total_cost += cost['total_cost']
    return total_cost / len(files)

# Starting point (from simple controller)
best_params = np.array([0.17, 0.10, -0.03, 0.13])
best_cost = evaluate(best_params)
print(f"Initial: params={best_params}, cost={best_cost:.2f}")

# Random search with local refinement
np.random.seed(42)
for iteration in range(200):
    # Random perturbation
    if iteration < 100:
        # Global search
        noise = np.random.randn(4) * np.array([0.05, 0.03, 0.02, 0.05])
    else:
        # Local refinement
        noise = np.random.randn(4) * np.array([0.01, 0.01, 0.005, 0.01])
    
    new_params = best_params + noise
    # Keep params in reasonable range
    new_params = np.clip(new_params, [-0.5, -0.5, -0.2, -0.5], [0.5, 0.5, 0.2, 0.5])
    
    try:
        new_cost = evaluate(new_params)
        if new_cost < best_cost:
            best_params = new_params
            best_cost = new_cost
            print(f"Iter {iteration}: params={best_params}, cost={best_cost:.2f}")
    except Exception as e:
        pass

print(f"\nBest: params={best_params}, cost={best_cost:.2f}")
print(f"kp={best_params[0]:.4f}, ki={best_params[1]:.4f}, kd={best_params[2]:.4f}, kff={best_params[3]:.4f}")
