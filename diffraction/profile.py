import jax
import jax.numpy as jnp
# from functools import partial
from phases.models import models_dict_jax, par_form_dict
from diffraction.intensity import intensity_array_jax, intensity_array_jax_snap
from diffraction.geometry import build_delta_array, build_delta_array_snap
from diffraction.geometry import two_theta_hkl_jax
from utils.format import get_value


# ---- Стеккер профиля ----
def sum_peak_profiles_jax(axes, amps, mus, shape_params_dict, peak_model):
    """
    Вернёт суммарный профиль по всем рефлексам, shape (N,).
    shape_params_dict: {'σ': scalar or array(M,), 'η': ...}
    """

    # преобразуем словарь в список ключей и значений
    keys = list(shape_params_dict.keys())
    values = [shape_params_dict[k] for k in keys]

    # функция одного пика
    def one_peak(x, A, mu, *shape_values):
        kwargs = {k: v for k, v in zip(keys, shape_values)}
        return peak_model(x, A, mu, **kwargs)  # (N,)

    # собираем in_axes: x → None, A → 0, mu → 0, shape_values → либо 0 либо None
    in_axes = (None, 0, 0) + tuple(0 if jnp.ndim(v) > 0 else None for v in values)

    # векторизация
    one_peak_vmap = jax.vmap(one_peak, in_axes=in_axes)
    peaks = one_peak_vmap(axes, amps, mus, *values)  # (M,N)

    return jnp.sum(peaks, axis=0)  # (N,)



# ---- Суммарный профиль ----
def phase_profile_jax(axes, project_object=None, prefix_KPhase=None, **params):
    """
    Строит суммарный профиль фазы для заданной оси 2θ.

    Args:
        axes : jnp.array (N,) или np.array – сетка по 2θ
        project_object :
        prefix_KPhase : определяет объект фазы
        params : dict – уточняемые параметры
        internal_scale : float – внутренний масштаб (для амплитуд по le Beil)

    Returns:
        profile : jnp.array (N,) – суммарный профиль
    """
    my_phase      = project_object.__dict__.get(prefix_KPhase.replace('_',''))

    # --- 1. Массив амплитуд ---
    amps    = intensity_array_jax(my_phase, **params)  # (M,)
    amps    = jnp.nan_to_num(amps, nan=0)   # тоже заменяет NaN на I_0_0_0


    # --- 2. Позиции пиков (центры, 2θ) ---
    cell_array  = [get_value(params[my_phase.prefix + par]) for par in ['a','b','c','alpha','beta','gamma']]
    hkl_array   = jnp.array([line[:3] for line in my_phase.bragg_positions])
    #delta_array = build_delta_array(my_phase.bragg_positions, my_phase.prefix, my_phase.setting, params)
    delta_array = build_delta_array(my_phase.bragg_positions, my_phase.prefix, my_phase.settings, params)
    mus         = two_theta_hkl_jax(hkl_array, *cell_array, my_phase.wavelength, delta_array)

    # --- 3. Определение модели профиля ---
    model_name = my_phase.settings.form
    peak_model = models_dict_jax[model_name]

    # --- 4. Сбор shape-параметров для модели ---
    shape_params_dict = {}
    for par_def in par_form_dict[model_name]:
        name = par_def['name']
        if name in ['A', 'μ']:
            continue                                                            # амплитуда и центр обрабатываются отдельно
        full_name = my_phase.prefix + model_name + '_' + name
        shape_params_dict[name] = get_value(params[full_name])

    # --- 5. Суммирование профилей всех рефлексов ---
    profile = sum_peak_profiles_jax(jnp.array(axes), amps, mus, shape_params_dict, peak_model)

    L_of_ring = jnp.sin(jnp.deg2rad(axes) / 2.0) / my_phase.wavelength * (2.0 * jnp.pi)
    # защитим от деления на ноль — если L_of_ring == 0 (в начале оси),
    # заменим на 1.0 чтобы не получить inf; это аналогично исключению нулевого рефлекса
    L_safe = jnp.where(L_of_ring > 0.0, L_of_ring, 1.0)

    profile_normed = profile / L_safe
    return profile_normed


# ---- Суммарный профиль (по снимку) ----
def phase_profile_jax_snap(axes, project_snap=None, phase_name=None, **params):
    """
    Snapshot-версия построения профиля.
    Строит суммарный профиль фазы для заданной оси 2θ.

    Args:
        axes : jnp.array (N,) или np.array – сетка по 2θ
        project_object :
        prefix_KPhase : определяет объект фазы
        params : dict – уточняемые параметры
        internal_scale : float – внутренний масштаб (для амплитуд по le Beil)

    Returns:
        profile : jnp.array (N,) – суммарный профиль
    """
    phase_snap = project_snap["phases"][phase_name]

    # --- 1. Массив амплитуд ---
    amps    = intensity_array_jax_snap(phase_snap, **params)  # (M,)
    amps    = jnp.nan_to_num(amps, nan=0)   # тоже заменяет NaN на I_0_0_0


    # --- 2. Позиции пиков (центры, 2θ) ---
    cell_array  = [get_value(params[phase_snap["prefix"] + par]) for par in ['a','b','c','alpha','beta','gamma']]
    hkl_array   = jnp.array([line[:3] for line in phase_snap["bragg_positions"]])
    #delta_array = build_delta_array(my_phase.bragg_positions, my_phase.prefix, my_phase.setting, params)
    delta_array = build_delta_array_snap(phase_snap["bragg_positions"], phase_snap["prefix"], phase_snap["settings"], params)
    mus         = two_theta_hkl_jax(hkl_array, *cell_array, phase_snap["wavelength"], delta_array)

    # --- 3. Определение модели профиля ---
    model_name = phase_snap["settings"]["form"]
    peak_model = models_dict_jax[model_name]

    # --- 4. Сбор shape-параметров для модели ---
    shape_params_dict = {}
    for par_def in par_form_dict[model_name]:
        name = par_def['name']
        if name in ['A', 'μ']:
            continue                                                            # амплитуда и центр обрабатываются отдельно
        full_name = phase_snap["prefix"] + model_name + '_' + name
        shape_params_dict[name] = get_value(params[full_name])

    # --- 5. Суммирование профилей всех рефлексов ---
    profile = sum_peak_profiles_jax(jnp.array(axes), amps, mus, shape_params_dict, peak_model)

    L_of_ring = jnp.sin(jnp.deg2rad(axes) / 2.0) / phase_snap["wavelength"] * (2.0 * jnp.pi)
    # защитим от деления на ноль — если L_of_ring == 0 (в начале оси),
    # заменим на 1.0 чтобы не получить inf; это аналогично исключению нулевого рефлекса
    L_safe = jnp.where(L_of_ring > 0.0, L_of_ring, 1.0)

    profile_normed = profile / L_safe
    return profile_normed



# 🧪 Обязательно: тест сравнения

#profile_old = phase_profile_jax(pr.Profile_points.two_theta, pr, pr.Phase1.prefix, **pr.params)

#profile_new = phase_profile_jax_snap(
#    pr.Profile_points.two_theta,
#    project_to_snapshot(pr),
#    phase_name=pr.Phase1.name,
#    **pr.params
#)

#print(jnp.max(jnp.abs(profile_old - profile_new)))