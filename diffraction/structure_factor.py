import numpy as np
import jax.numpy as jnp
from utils.format import get_value
from diffraction.geometry import stl_hkl_jax
from diffraction.scattering_factor import fe_el_jax_wrapper, fe_el_jax_wrapper_snap
from atoms.generate import get_all_positions_in_cell_for_atom


# ---- F² (с atom_map) ----
def F2_jax(hkl_array, x, y, z, occ, fe_el, t_at, t_overall, atom_map):
    """
    Вычисляет интенсивности |F|² для набора отражений.

    Формула:
        F(hkl) = sum_j [ f_j * T_j * occ_j * exp(2πi (h x_j + k y_j + l z_j)) ] * T_overall
        |F|² = |F(hkl)|^2

    Parameters
    ----------
    hkl_array : ndarray, shape (M, 3)       ← Индексы отражений (h, k, l).
    x, y, z :   ndarray, shape (N_sites,)   ← Координаты всех атомных позиций (с учётом симметрий).
    occ :       ndarray, shape (N_sites,)   ← Заселённости позиций.
    fe_el :     ndarray, shape (M, N_atoms) ← Атомные факторы рассеяния для уникальных атомов.
    t_at :      ndarray, shape (M, N_atoms) ← Температурные поправки для уникальных атомов.
    t_overall : ndarray, shape (M,)         ← Общая температурная поправка.
    atom_map :  ndarray, shape (N_sites,)   ← Отображение: позиция → индекс уникального атома.

    Returns
    -------
    F2 :        ndarray, shape (M,)          ← Квадраты структурных амплитуд для отражений |F|^2.
    """
    h, k, l = hkl_array[:,0,None], hkl_array[:,1,None], hkl_array[:,2,None]   # (M,1)
    phase = jnp.exp(2j * jnp.pi * (h*x + k*y + l*z))    # (M, N_sites)

    fe_sites = fe_el[:, atom_map]   # (M, N_sites)
    t_sites  = t_at[:, atom_map]    # (M, N_sites)

    F = jnp.sum(fe_sites * t_sites * occ * phase, axis=1) * t_overall
    return jnp.abs(F)**2



# ---- Посчитать F² на jax для фазы (с atom_map) ----
def F2_array_jax(phase_object,**params):
    prefix = phase_object.prefix

    # --- 1. Параметры ячейки ---
    cell_array = [get_value(params[prefix + par]) for par in ['a','b','c','alpha','beta','gamma']]

    # --- 2. HKL и stl ---
    hkl_array = np.array([line[:3] for line in phase_object.bragg_positions])
    stl_array = stl_hkl_jax(hkl_array, *cell_array)
    stl_sq    = stl_array**2

    # --- 3. Общая температурная поправка ---
    Biso_overall = get_value(params[prefix + 'Biso_overall'])
    t_overall    = jnp.exp(-Biso_overall * stl_sq)

    # --- 4. Сбор данных по атомам ---
    all_sites, all_occ, atom_map = [], [], []
    fe_el_list, t_at_list        = [], []

    for atom_idx, atom in enumerate(phase_object.atoms):
        # координаты, заселённость, Biso
        xa      = get_value(params[prefix + atom.name + '_x'])
        ya      = get_value(params[prefix + atom.name + '_y'])
        za      = get_value(params[prefix + atom.name + '_z'])
        occ     = get_value(params[prefix + atom.name + '_occ'])
        Biso_at = get_value(params[prefix + atom.name + '_Biso'])
        sites   = get_all_positions_in_cell_for_atom(xa, ya, za, phase_object.symmetry_operations)

        # атомный фактор рассеяния (универсальная обёртка)
        fe_el = fe_el_jax_wrapper(stl_array, atom, prefix, **params)

        # атомная температурная поправка
        t_at  = jnp.exp(-Biso_at * stl_sq)

        fe_el_list.append(fe_el)   # (M,)
        t_at_list.append(t_at)     # (M,)

        # позиции и отображение site → atom
        for pos in sites:
            all_sites.append(pos)
            all_occ.append(occ)
            atom_map.append(atom_idx)

    # --- 5. Приведение к массивам ---
    all_sites = jnp.array(all_sites)                  # (N_sites, 3)
    all_occ   = jnp.array(all_occ)                    # (N_sites,)
    atom_map  = jnp.array(atom_map)                   # (N_sites,)
    fe_el_all = jnp.stack(fe_el_list, axis=1)         # (M_stl, N_atoms)
    t_at_all  = jnp.stack(t_at_list, axis=1)          # (M_stl, N_atoms)

    # --- 6. Структурные факторы ---
    F2 = F2_jax(hkl_array,
                all_sites[:,0], all_sites[:,1], all_sites[:,2],
                all_occ, fe_el_all, t_at_all, t_overall, atom_map)

    return F2



# ---- Посчитать F² на jax для фазы (с atom_map) по снимку ----
def F2_array_jax_snap(phase_snap,**params):
    prefix = phase_snap["prefix"]

    # --- 1. Параметры ячейки ---
    cell_array = [get_value(params[prefix + par]) for par in ['a','b','c','alpha','beta','gamma']]

    # --- 2. HKL и stl ---
    hkl_array = np.array([line[:3] for line in phase_snap["bragg_positions"]])
    stl_array = stl_hkl_jax(hkl_array, *cell_array)
    stl_sq    = stl_array**2

    # --- 3. Общая температурная поправка ---
    Biso_overall = get_value(params[prefix + 'Biso_overall'])
    t_overall    = jnp.exp(-Biso_overall * stl_sq)

    # --- 4. Сбор данных по атомам ---
    all_sites, all_occ, atom_map = [], [], []
    fe_el_list, t_at_list        = [], []

    for atom_idx, atom_snap in enumerate(phase_snap["atoms"]):
        # координаты, заселённость, Biso
        xa      = get_value(params[prefix + atom_snap["name"] + '_x'])
        ya      = get_value(params[prefix + atom_snap["name"] + '_y'])
        za      = get_value(params[prefix + atom_snap["name"] + '_z'])
        occ     = get_value(params[prefix + atom_snap["name"] + '_occ'])
        Biso_at = get_value(params[prefix + atom_snap["name"] + '_Biso'])
        sites   = get_all_positions_in_cell_for_atom(xa, ya, za, phase_snap["symmetry_operations"])

        # атомный фактор рассеяния (универсальная обёртка)
        fe_el = fe_el_jax_wrapper_snap(stl_array, atom_snap, prefix, **params)

        # атомная температурная поправка
        t_at  = jnp.exp(-Biso_at * stl_sq)

        fe_el_list.append(fe_el)   # (M,)
        t_at_list.append(t_at)     # (M,)

        # позиции и отображение site → atom
        for pos in sites:
            all_sites.append(pos)
            all_occ.append(occ)
            atom_map.append(atom_idx)

    # --- 5. Приведение к массивам ---
    all_sites = jnp.array(all_sites)                  # (N_sites, 3)
    all_occ   = jnp.array(all_occ)                    # (N_sites,)
    atom_map  = jnp.array(atom_map)                   # (N_sites,)
    fe_el_all = jnp.stack(fe_el_list, axis=1)         # (M_stl, N_atoms)
    t_at_all  = jnp.stack(t_at_list, axis=1)          # (M_stl, N_atoms)

    # --- 6. Структурные факторы ---
    F2 = F2_jax(hkl_array,
                all_sites[:,0], all_sites[:,1], all_sites[:,2],
                all_occ, fe_el_all, t_at_all, t_overall, atom_map)

    return F2


# 🧪 Как тестировать (очень важно)

# F2_old = F2_array_jax(pr.Phase1, **pr.params)
# F2_new = F2_array_jax_snap(phase_to_snapshot(pr.Phase1), **pr.params)

# print(jnp.max(jnp.abs(F2_old - F2_new)))