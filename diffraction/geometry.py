import jax.numpy as jnp
from phases.params import hkl_to_str


"""
Geometry layer of diffraction pipeline.

Contains functions related to:
- d-spacing                     # d
- sin(theta)/lambda             # stl
- 2theta calculation            # 2θ
- peak position corrections     # Δ(2θ)

This module does NOT depend on:
- atomic scattering
- structure factors
- amplitudes
"""


def d_hkl_jax(hkl_array, a, b, c, alpha, beta, gamma):
    """
    Векторизованная версия d_hkl на jax — для произвольной сингонии.
    hkl_array — shape (M, 3)
    Возвращает массив d (M,)
    """
    h = hkl_array[:, 0]
    k = hkl_array[:, 1]
    l = hkl_array[:, 2]

    alpha_r = jnp.deg2rad(alpha)
    beta_r  = jnp.deg2rad(beta)
    gamma_r = jnp.deg2rad(gamma)

    cα, cβ, cγ = jnp.cos(alpha_r), jnp.cos(beta_r), jnp.cos(gamma_r)
    sα, sβ, sγ = jnp.sin(alpha_r), jnp.sin(beta_r), jnp.sin(gamma_r)
    ω          = jnp.sqrt(1 - cα**2 - cβ**2 - cγ**2 + 2 * cα * cβ * cγ)

    C1 = (h/(a*sα))**2 + (k/(b*sβ))**2 + (l/(c*sγ))**2
    C2 = (2*h*k/(a*b)*(cα*cβ - cγ) +
          2*h*l/(a*c)*(cγ*cα - cβ) +
          2*k*l/(b*c)*(cβ*cγ - cα))
    D  = (1/ω**2)*(C1 + C2)

    mask = (h!=0) | (k!=0) | (l!=0)            # jax не поддерживает "маски с присваиванием", используем jnp.where
    d = jnp.where(mask, 1/jnp.sqrt(D), 0.0)
    return d



def stl_hkl_jax(hkl_array, a,b,c,alpha,beta,gamma):
    """
    Векторизованная версия stl_hkl на jax — для произвольной сингонии.
    hkl_array — shape (M, 3)   — numpy or jnp
    Возвращает массив stl (M,) — jnp array
    """
    hkl_j = jnp.array(hkl_array)
    d = d_hkl_jax(hkl_j, a,b,c,alpha,beta,gamma)
    stl = jnp.where(d > 0, 1.0/(2.0 * d), 0.0)
    return stl



#@title Векторный расчёт 2θ для всех hkl (JAX)
def two_theta_hkl_jax(hkl_array, a, b, c, alpha, beta, gamma, wavelength, delta_array):
    """
    hkl_array: jnp.array shape (M,3) or array-like
    delta_array: jnp.array shape (M,) — значения в градусах, которые нужно вычесть из 2θ
    Возвращает jnp.array shape (M,) — 2θ в градусах (с delta вычтенным).
    """
    hkl_j = jnp.array(hkl_array)
    d     = d_hkl_jax(hkl_j, a, b, c, alpha, beta, gamma)
    val = jnp.where(d > 0.0, wavelength / (2.0 * d), 0.0) # безопасный аргумент для asin: val = λ/(2d) — но если d==0 => val=inf -> надо 0
    val = jnp.clip(val, -1.0, 1.0)                        # clamp val в [-1, 1] для безопасности численной
    two_theta = 2.0 * jnp.arcsin(val)                     # в радианах
    two_theta_deg = jnp.degrees(two_theta)

    two_theta_corrected = two_theta_deg - delta_array     # вычитаем положения рефлексов с поправкой
    two_theta_corrected = jnp.where(d > 0.0, two_theta_corrected, 0.0)  # если h=k=l=0 (d==0) хотим ноль:
    return two_theta_corrected

def two_theta_hkl_single(h, k, l, cell, wavelength, delta=0.0):
    hkl = jnp.array([[h, k, l]])
    delta_arr = jnp.array([delta])
    val = two_theta_hkl_jax(hkl, *cell, wavelength, delta_arr)[0]
    return float(val)


def build_delta_array(bragg_positions, prefix, settings, numeric_params):
    """
    bragg_positions: array-like shape (M, >=3) or list of lists; первые 3 колонки h,k,l
    prefix: префикс фазы, например 'Phase1_'
    setting: my_phase.setting (python dict)
    numeric_params: dict с числовыми значениями параметров
    Возвращает jnp.array shape (M,) с delta для каждого рефлекса.
    """
    # преобразуем bragg_positions в numpy-подобный массив на python-уровне
    # предполагаем, что bragg_positions — list/np.array; получаем список кортежей (h,k,l)
    hkls = [(int(line[0]), int(line[1]), int(line[2])) for line in bragg_positions]

    #use_calib = bool(setting.get('calibration mode', False))
    #calibrate_list = setting.get('calibrate', None)  # может быть 'all' или список
    use_calib = bool(settings.calibration_mode)
    calibrate_list = settings.calibrate  # может быть 'all' или список

    deltas = []
    for (h, k, l) in hkls:
        if use_calib and (calibrate_list == 'all' or [h, k, l] in calibrate_list):
            key = f"{prefix}delta_{hkl_to_str(h,k,l)}"
            val = numeric_params.get(key, 0.0)
            deltas.append(float(val))
        else:
            deltas.append(0.0)
    return jnp.array(deltas)


def build_delta_array_snap(bragg_positions, prefix, settings, numeric_params):
    """
    bragg_positions: array-like shape (M, >=3) or list of lists; первые 3 колонки h,k,l
    prefix: префикс фазы, например 'Phase1_'
    setting: my_phase.setting (python dict)
    numeric_params: dict с числовыми значениями параметров
    Возвращает jnp.array shape (M,) с delta для каждого рефлекса.
    """
    # преобразуем bragg_positions в numpy-подобный массив на python-уровне
    # предполагаем, что bragg_positions — list/np.array; получаем список кортежей (h,k,l)
    hkls = [(int(line[0]), int(line[1]), int(line[2])) for line in bragg_positions]

    #use_calib = bool(setting.get('calibration mode', False))
    #calibrate_list = setting.get('calibrate', None)  # может быть 'all' или список
    use_calib = bool(settings["calibration_mode"])
    calibrate_list = settings["calibrate"]  # может быть 'all' или список

    deltas = []
    for (h, k, l) in hkls:
        if use_calib and (calibrate_list == 'all' or [h, k, l] in calibrate_list):
            key = f"{prefix}delta_{hkl_to_str(h,k,l)}"
            val = numeric_params.get(key, 0.0)
            deltas.append(float(val))
        else:
            deltas.append(0.0)
    return jnp.array(deltas)
