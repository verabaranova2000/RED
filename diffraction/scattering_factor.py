import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from utils.format import get_value

# ---- IT4322 fₑₗ ----
def f_el_matrix_jax(s, PARAM_A_at, PARAM_B_at):
    """
    Электронный фактор рассеяния для одного атома (JAX-версия).

    s        : jnp.ndarray, shape (M,)
    return   : jnp.ndarray, shape (M,)
    """
    s2 = s**2                                               # (M,)
    exp_term = jnp.exp(-PARAM_B_at[:, None] * s2[None, :])  # broadcasting: (5, M)
    fe = jnp.sum(PARAM_A_at[:, None] * exp_term, axis=0)    # (M,)
    return fe
f_el_matrix_jax_jit = jax.jit(f_el_matrix_jax)



# ---- Каппа-модель fₑₗ ----
def f_el_kmodel_jax_preinterp(stl_arr, phase_prefix, atom_name, atom_Z, curves, **pars):
    full_prefix = phase_prefix+atom_name+'_'

    # --- 1. precompute arrays (numpy) on stl_arr ---
    interp_core = interp1d(curves['core']['x'], curves['core']['y'], kind='quadratic', fill_value='extrapolate')
    fe_core_np  = interp_core(stl_arr)
    names_curves_valence = [k for k,v in curves.items() if k not in ['neutral atom', 'core']]
    valence_np_list = []
    P_list = []
    for shell in names_curves_valence:
        interp_v = interp1d(curves[shell]['x'], curves[shell]['y'], kind='quadratic', fill_value='extrapolate')
        kappa_v  = float(get_value(pars[full_prefix + shell + '_kappa']))
        P_v      = float(get_value(pars[full_prefix + shell + '_P']))
        valence_np_list.append(interp_v(stl_arr / kappa_v))
        P_list.append(P_v)

    valence_stack_np = np.stack(valence_np_list, axis=0)   # (N_valence, K)
    P_np = np.array(P_list)                                # (N_valence,)

    # --- 2. call JAX formula but with precomputed arrays cast to jnp.float64 ---
    fe_core_j  = jnp.array(fe_core_np, dtype=jnp.float64)
    valence_j  = jnp.array(valence_stack_np, dtype=jnp.float64)
    P_j        = jnp.array(P_np, dtype=jnp.float64)
    stl_j      = jnp.array(stl_arr, dtype=jnp.float64)

    # JAX formula that takes precomputed arrays:
    def fe_el_from_precomputed(stl, fe_core, valence_stack, P_v, Z):
        factor       = 1.0 / (8 * jnp.pi**2 * 0.529177210544)
        P_fe_valence = jnp.dot(P_v, valence_stack)  # (K,)
        fe_el        = jnp.where(stl == 0, 0.0, factor / (stl**2) * (Z - fe_core - P_fe_valence))
        #fe_el        = factor / (stl**2) * (Z - fe_core - P_fe_valence)
        return fe_el

    fe_jax_res2 = fe_el_from_precomputed(stl_j, fe_core_j, valence_j, P_j, atom_Z)
    return fe_jax_res2



# ---- Обертка для fₑₗ (выбор модели) ----
def f_el_jax_wrapper(stl, atom, phase_prefix, **pars):
    """
    Универсальная обёртка для вычисления fₑₗ атома.

    stl     : jnp.ndarray, shape (M,)
    atom    : Atom object
    pars    : dict параметров (для каппа-модели)

    return  : jnp.ndarray, shape (M,)
    """
    model = atom.settings.fe_from
    if model == 'it4322':
      return f_el_matrix_jax_jit(stl, 
                                atom.info['it4322']['A'], 
                                atom.info['it4322']['B'])
    elif model == 'Mott-Bethe':
      return f_el_kmodel_jax_preinterp(stl, phase_prefix,
                                         atom.name, 
                                         atom.Z, 
                                         atom.info['curves'],**pars)
    else:
      raise ValueError(f"Unknown model for atom {atom.name}: {model}")



# ---- Обертка для fₑₗ (выбор модели) по снимку ----
def f_el_jax_wrapper_snap(stl, atom_snap, phase_prefix, **pars):
    """
    Snapshot-версия функции f_el_jax_wrapper.
    Универсальная обёртка для вычисления f_e атома.

    stl     : jnp.ndarray, shape (M,)
    atom    : Atom object
    pars    : dict параметров (для каппа-модели)

    return  : jnp.ndarray, shape (M,)
    """
    model = atom_snap["fe_from"]
    if model == 'it4322':
      return f_el_matrix_jax_jit(stl,
                                atom_snap['it4322']['A'],
                                atom_snap['it4322']['B'])
    elif model == 'Mott-Bethe':
      return f_el_kmodel_jax_preinterp(stl, phase_prefix,
                                         atom_snap["name"],
                                         atom_snap["Z"],
                                         atom_snap["curves"],**pars)
    else:
      raise ValueError(f"Unknown model for atom {atom_snap['name']}: {model}")
