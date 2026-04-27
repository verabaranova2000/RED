from .geometry import (
    d_hkl_jax,
    stl_hkl_jax,
    two_theta_hkl_jax,
    two_theta_hkl_single,
    build_delta_array,
    build_delta_array_snap,
)


"""
geometry → f_el → F² → I_hkl → profile → model



OLD:
------
phase_profile
  └── phase_profile_hkl
          └── compute_intensity


NEW (jax):
------
phase_profile_jax
  ├── intensity_array_jax   ← сразу все I_hkl
  ├── two_theta_hkl_jax     ← сразу все μ
  └── sum_peak_profiles_jax ← сразу сумма по всем пикам
"""