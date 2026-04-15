import numpy as np

# ==== Профильный R-фактор ====
def profile_R_factor(y_obs, y_calc):
  a1=float(sum(abs(y_obs-y_calc)))
  a2=float(sum(y_obs))
  Rp=a1/a2*100
  return Rp


def profile_R_factor_from_diff(diff, y_obs):
    return float(np.sum(np.abs(diff)) / np.sum(np.abs(y_obs)) * 100)