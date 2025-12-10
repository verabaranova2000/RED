# ===== Атомный фактор рассеяния ======
import numpy as np
from scipy.integrate import simpson

def safe_sinc(x):
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out
    

def compute_fq(r, rho, qgrid_angstrom):
    """
    Вычисляет атомный фактор рассеяния f(q) из сферически симметричной
    электронной плотности ρ(r) с помощью преобразования Фурье–Бесселя:

        f(q) = ∫ 4π r² ρ(r) sin(qr)/(qr) dr

    Параметры
    ---------
    r : ndarray
        Радиальная сетка в боровских радиусах (Bohr).
    rho : ndarray
        Электронная плотность, нормированная на один электрон.
    qgrid_angstrom : ndarray
        Вектор рассеяния q в единицах Å⁻¹.

    Возвращает
    ----------
    f : ndarray
        Атомный фактор рассеяния f(q).
    """
    # --- константа: 1 Bohr = 0.529177210903 Å ---
    a0_in_A = 0.529177210903

    f = np.zeros_like(qgrid_angstrom)
    qgrid_au = qgrid_angstrom * a0_in_A      # переводим q из 1/Å -> 1/Bohr
    for i, q_au in enumerate(qgrid_au):
        integrand = 4*np.pi * r**2 * rho * safe_sinc(q_au * r)
        f[i]      = simpson(integrand, r)    # или np.trapezoid
    return f
