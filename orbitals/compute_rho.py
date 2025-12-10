from scipy.integrate import simpson
import numpy as np


# ====== Плотность одной орбитали ========
def compute_rho_orbital(r, P, Q):
    """Электронная плотность одной орбитали."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = (P**2 + Q**2) / (4*np.pi*r**2)
    return np.nan_to_num(rho)

# ====== Нормировка ========
def normalize_rho_1e(r, rho):
    """Нормирует плотность на 1 электрон."""
    norm = simpson(4*np.pi * r**2 * rho, r)
    return rho / norm


def build_rho_total(rho_orbs, occupations):
    """ Сборка полной плотности по заселённостям """
    rho_total = np.zeros_like(next(iter(rho_orbs.values())))
    for nm, rho in rho_orbs.items():
        rho_total += occupations[nm] * rho
    return rho_total


def check_electrons(r, rho):
    """ Контроль электронов """
    return simpson(4*np.pi * r**2 * rho, r)


def build_rho_core(rho_orbs, occupations, valence_list):
    """
    Строит электронную плотность ядра (core), вычитая из полной плотности
    вклад заданных валентных оболочек.

    Parameters
    ----------
    rho_orbs : dict[str, ndarray]
        Плотности орбиталей, нормированные на 1 электрон.
    occupations : dict[str, float]
        Заселённости орбиталей.
    valence_list : list[str]
        Список валентных оболочек (например: ['4f', '5d', '6s']).

    Returns
    -------
    rho_core : ndarray
        Электронная плотность ядра.
    rho_val_dict : dict[str, ndarray]
        Отдельные вклады каждой валентной оболочки (по одной).
    """

    rho_core = np.zeros_like(next(iter(rho_orbs.values())))
    rho_val_dict = {}

    for nm, rho in rho_orbs.items():
        base = nm.replace('-', '').replace('+', '')

        if any(base.startswith(v) for v in valence_list):
            rho_val_dict[nm] = occupations[nm] * rho
        else:
            rho_core += occupations[nm] * rho

    return rho_core, rho_val_dict

