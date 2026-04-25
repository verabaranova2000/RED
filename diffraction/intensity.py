import jax.numpy as jnp
from phases.params import hkl_to_str
from diffraction.structure_factor import F2_array_jax, F2_array_jax_snap

# ---- Поправка по Блэкману ----
def blackman_correction_jax(F_mod, A, eps=1e-6):
    """
    JAX-векторизованная поправка по Блэкману.

    Args:
        F_mod : jnp.array, модуль структурного фактора |F|
        A     : float или jnp.array, параметр A
        eps   : малое значение для устойчивости
    """
    x = A*F_mod
    return jnp.where(jnp.abs(x)<eps, 1.0, (1-jnp.exp(-x))/x)




"""
Сначала вычисляется базовый ритвельдовский массив 
    base_riet = internal_scale * scale * phvol * p * F2,
Затем к нему применяется Blackman-коэффициент 
    (в виде массива blackman_corr, где по умолчанию 1.0),
Затем заменяем на соответствующие позиции (mask_le) независимые амплитуды I_hkl 
    (если они есть в pars, иначе 0).
Никаких if mask.any() и никаких умножений всего массива при установке части элементов.
"""


# ---- Ampl (с учетом setting) ----
def intensity_array_jax(phase_object, **pars):
    """
    Векторная версия сборки амплитуд (jax-friendly).
    Возвращает jnp.array shape (N,)
    """
    prefix = phase_object.prefix
    bragg = phase_object.bragg_positions
    N = len(bragg)

    # --- HKL как Python-список для удобства работы со строками параметров ---
    py_hkls = [ (int(line[0]), int(line[1]), int(line[2])) for line in bragg ]

    # --- jnp-массивы для вычислений ---
    # hkl_array   = jnp.array(py_hkls)                        # (N,3)
    mult_array  = jnp.array([float(line[3]) for line in bragg])  # multiplicity p (N,)
    mode_ids    = jnp.array([int(line[9]) for line in bragg])    # 0,1,2 (N,)

    # --- маски для разных режимов расчёта ---
    mask_riet      = (mode_ids == 0)
    mask_le        = (mode_ids == 1)
    mask_blackman  = (mode_ids == 2)

    # --- коэффициенты масштабирования и объёма фаз ---
    scale = float(pars.get(prefix + "scale", 1.0))
    phvol = float(pars.get(prefix + "phvol", 1.0))

    # --- F^2 для всех рефлексов ---
    compute_riet = mask_riet.any()  # True, если есть рефлексы Ритвелда
    F2_all = jnp.where(compute_riet,F2_array_jax(phase_object, **pars), jnp.ones(N)) # заглушка, чтобы не ломался код

    # --- Blackman: коэффициент коррекции ---
    compute_blackman = mask_blackman.any()  # True, если хотя бы один рефлекс с Blackman
    A_val = float(pars.get(prefix + "A", 0.0001))                       #          извлекаем A, если отсутствует — 1
    blackman_all = jnp.where(compute_blackman,blackman_correction_jax(jnp.sqrt(F2_all), A_val),jnp.ones(N))
    blackman_corr = jnp.where(mask_blackman, blackman_all, 1.0)      # (N,)     применяем только для соответствующих рефлексов

    # --- базовая ритвельдовская амплитуда ---
    base_riet = scale * phvol * mult_array * F2_all * blackman_corr  # (N,)

    # --- подготовка массива I_hkl для Le-Beil ---
    I_all = jnp.array([ float(pars.get(f"{prefix}I_{hkl_to_str((h, k, l))}", 0.0)) for (h,k,l) in py_hkls ])   # если соответствующего ключа нет в pars — используем 0.0
    internal_scale = phase_object.settings.internal.internal_scale

    # --- итоговая амплитуда: заменяем для Le-Beil ---
    Ampl = base_riet.at[mask_le].set(internal_scale * I_all[mask_le])

    return Ampl



#@title **4) 🧩. $Ampl$ (с учетом setting)**

def intensity_array_jax_snap(phase_snap, **pars):
    """
    Snapshot-версия сборки амплитуд.

    Векторная версия сборки амплитуд (jax-friendly).
    Возвращает jnp.array shape (N,)
    """
    prefix = phase_snap["prefix"]
    bragg = phase_snap["bragg_positions"]
    N = len(bragg)

    # --- HKL как Python-список для удобства работы со строками параметров ---
    py_hkls = [ (int(line[0]), int(line[1]), int(line[2])) for line in bragg ]

    # --- jnp-массивы для вычислений ---
    # hkl_array   = jnp.array(py_hkls)                        # (N,3)
    mult_array  = jnp.array([float(line[3]) for line in bragg])  # multiplicity p (N,)
    mode_ids    = jnp.array([int(line[9]) for line in bragg])    # 0,1,2 (N,)

    # --- маски для разных режимов расчёта ---
    mask_riet      = (mode_ids == 0)
    mask_le        = (mode_ids == 1)
    mask_blackman  = (mode_ids == 2)

    # --- коэффициенты масштабирования и объёма фаз ---
    scale = float(pars.get(prefix + "scale", 1.0))
    phvol = float(pars.get(prefix + "phvol", 1.0))

    # --- F^2 для всех рефлексов ---
    compute_riet = mask_riet.any()  # True, если есть рефлексы Ритвелда
    F2_all = jnp.where(compute_riet, F2_array_jax_snap(phase_snap, **pars), jnp.ones(N)) # заглушка, чтобы не ломался код

    # --- Blackman: коэффициент коррекции ---
    compute_blackman = mask_blackman.any()  # True, если хотя бы один рефлекс с Blackman
    A_val = float(pars.get(prefix + "A", 0.0001))                       #          извлекаем A, если отсутствует — 1
    blackman_all = jnp.where(compute_blackman,blackman_correction_jax(jnp.sqrt(F2_all), A_val),jnp.ones(N))
    blackman_corr = jnp.where(mask_blackman, blackman_all, 1.0)      # (N,)     применяем только для соответствующих рефлексов

    # --- базовая ритвельдовская амплитуда ---
    base_riet = scale * phvol * mult_array * F2_all * blackman_corr  # (N,)

    # --- подготовка массива I_hkl для Le-Beil ---
    I_all = jnp.array([ float(pars.get(f"{prefix}I_{hkl_to_str((h, k, l))}", 0.0)) for (h,k,l) in py_hkls ])   # если соответствующего ключа нет в pars — используем 0.0
    internal_scale = phase_snap["settings"]["internal_scale"]

    # --- итоговая амплитуда: заменяем для Le-Beil ---
    Ampl = base_riet.at[mask_le].set(internal_scale * I_all[mask_le])

    return Ampl
