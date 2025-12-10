import numpy as np
import math
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



# ========= Векторы решетки ===============
def cell_to_lattice_vectors(a, b, c, alpha, beta, gamma, to_Bohr=False, verbose = False):
    """
    Вычисляет прямые (декартовы) векторы решётки по параметрам элементарной ячейки.

    Параметры
    ----------
    a, b, c : float
        Длины векторов элементарной ячейки (в ангстремах).
    alpha, beta, gamma : float
        Межвекторные углы (в градусах):
        alpha = ∠(b, c), beta = ∠(a, c), gamma = ∠(a, b).
    to_Bohr : bool, optional
        Если True, все длины (a, b, c) перед вычислением векторов
        переводятся из ангстремов в боры.  
        Используется для совместимости с квантово-химическими и
        электронно-структурными кодами, работающими в атомных единицах.
        Константа пересчёта: 1 Bohr = 0.529177210903 Å.
    verbose : bool, optional
        Если True, печатает полученные векторы решётки в удобочитаемой форме.

    Возвращает
    ----------
    np.ndarray shape (3, 3)
        Матрица, содержащая три прямых вектора решётки в декартовой системе координат.
        Формат:
            [[ax, ay, az],
             [bx, by, bz],
             [cx, cy, cz]]
    """
    # --- Перевод результата Å в боры ---
    if to_Bohr: 
      a0_in_A = 0.529177210903
      a, b, c = [ai/a0_in_A for ai in [a,b,c]]
    # --- Перевод углов в радианы ---
    alpha = np.radians(alpha)
    beta  = np.radians(beta)
    gamma = np.radians(gamma)
    # --- Координаты a1 ---
    a1 = np.array([a, 0.0, 0.0])
    # --- Координаты a2 ---
    a2 = np.array([b * np.cos(gamma),
                   b * np.sin(gamma),
                   0.0])
    # --- Вычисление компонента a3_z ---
    # Формула из International Tables:
    cz = c * np.sqrt(1- np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)
    # --- Координаты a3 ---
    a3 = np.array([c * np.cos(beta),
                   c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                   cz])
    vecs = np.vstack([a1, a2, a3])

    if verbose: 
      for line in vecs: print([round(float(ci), 3) for ci in line])
    return vecs


# =========== Символы Wyckoff ====================
def compute_wyckoffs(lattice_vectors, coords, symbols, symprec=1e-2):
    """
    Возвращает список Вайкоффских обозначений для атомов структуры.

    Параметры:
        lattice_vectors : array (3×3)
            Прямые векторы решётки в Å.
        coords : list of [x, y, z]
            Фракционные координаты всех атомов.
        symbols : list of str
            Химические символы.
        symprec : float
            Чувствительность при определении симметрии.

    Возвращает:
        list[str]:
            Список букв Вайкоффа для каждого атома.
    """
    s = Structure(lattice=lattice_vectors, species=symbols, coords=coords)
    sg = SpacegroupAnalyzer(s, symprec=symprec)
    dataset = sg.get_symmetry_dataset()
    return dataset["wyckoffs"]



## ====== Межплоскостные расстояния ======
def d_hkl(h,k,l, a,b,c,alpha,beta,gamma):
  """
  Межплоскостное расстояние d(hkl) для произвольной решётки.
  Углы в градусах.
  """
  c_α, c_β, c_γ = math.cos(alpha/180*math.pi), math.cos(beta/180*math.pi), math.cos(gamma/180*math.pi)
  s_α, s_β, s_γ = math.sin(alpha/180*math.pi), math.sin(beta/180*math.pi), math.sin(gamma/180*math.pi)
  ω=(1-c_α**2-c_β**2-c_γ**2+2*c_α*c_β*c_γ)**0.5
  C1=(h/(a/s_α))**2+(k/(b/s_β))**2+(l/(c/s_γ))**2                                               # Первые три слагаемых
  C2=2*h*k/(a*b)*(c_α*c_β-c_γ)+2*h*l/(a*c)*(c_γ*c_α-c_β)+2*k*l/(b*c)*(c_β*c_γ-c_α)              # Первые три слагаемых
  D=(1/ω**2)*(C1+C2)                                                                            # 1/d^2
  d=1/D**0.5 if not h==k==l==0 else 0
  return d

## ====== Объем элементарной ячейки ======
def volume_cell(a, b, c, alpha_deg, beta_deg, gamma_deg):
  """
  Объём элементарной ячейки: V = a*b*c*sqrt(1 - cos^2α - cos^2β - cos^2γ + 2 cosα cosβ cosγ)
  Углы задаются в градусах. Параметры могут быть скалярами или array-like (будут broadcast'иться).
  Возвращает numpy scalar или ndarray (в Å^3).
  """
  alpha    = np.deg2rad(alpha_deg)                          ## Перевод в радианы
  beta     = np.deg2rad(beta_deg)
  gamma    = np.deg2rad(gamma_deg)
  ca,cb,cg = np.cos(alpha), np.cos(beta), np.cos(gamma)     ## Косинусы углов

  term   = 1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg
  volume = a*b*c*np.sqrt(term)
  return volume

# Пример
#V = volume_cell(a, b, c, alpha, beta, gamma)
#print(f"Объём элементарной ячейки: {V:.4f} Å³")