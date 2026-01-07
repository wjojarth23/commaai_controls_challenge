from . import BaseController
import numpy as np
from collections import deque

class Controller(BaseController):
  """
  PID that compensates for road roll (bank) via `state.roll_lataccel` and
  accounts for two-step actuator/controller lag by predicting the vehicle
  lateral acceleration two steps ahead using a 3rd-degree polynomial fit
  (equivalent to a cubic Taylor predictor when using recent samples).
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    # Use positive kd and compute derivative on measurement for stability
    self.d = 0.053
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.integral_limit = 1.5
    self.dt = 0.1
    self.lat_hist = deque(maxlen=4)
    self.prev_lataccel = None
    self.pred_filtered = None
    self.pred_alpha = 0.4
    self.max_pred_delta = 0.5

  def _predict_two_steps(self):
    if len(self.lat_hist) < 4:
      return self.lat_hist[-1] if self.lat_hist else 0.0
    times = np.array([-(3*self.dt), -(2*self.dt), -self.dt, 0.0])
    vals = np.array(list(self.lat_hist))
    try:
      coeffs = np.polyfit(times, vals, 3)
      poly = np.poly1d(coeffs)
      t_pred = 2 * self.dt
      pred = float(poly(t_pred))
      # Exponential smoothing of prediction to reduce high-frequency jumps
      if self.pred_filtered is None:
        self.pred_filtered = pred
      else:
        self.pred_filtered = self.pred_alpha * pred + (1.0 - self.pred_alpha) * self.pred_filtered

      # Limit how much the predictor can change in one step
      if self.pred_filtered is not None:
        last = vals[-1]
        delta = np.clip(self.pred_filtered - last, -self.max_pred_delta, self.max_pred_delta)
        return float(last + delta)
      return float(pred)
    except Exception:
      return float(vals[-1])

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.lat_hist.append(current_lataccel)
    predicted_lataccel = self._predict_two_steps()

    roll_lataccel = state.roll_lataccel if hasattr(state, 'roll_lataccel') else 0.0

    # Compute error using the smoothed/clipped prediction and roll compensation
    error = target_lataccel - (predicted_lataccel + roll_lataccel)

    # Integral with anti-windup
    self.error_integral += error * self.dt
    self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)

    # Derivative on measurement (lataccel rate) for noise robustness
    if self.prev_lataccel is None:
      lataccel_rate = 0.0
    else:
      lataccel_rate = (current_lataccel - self.prev_lataccel) / self.dt
    self.prev_lataccel = current_lataccel

    p_term = self.p * error
    i_term = self.i * self.error_integral
    d_term = -self.d * lataccel_rate

    return p_term + i_term + d_term
