# функции для чтения выходных файлов GRASP
import numpy as np
from scipy.interpolate import interp1d

# ==== Чтение заселенностей ====
def read_occupations_from_grasp_sum(filename):
    occupations = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    # --- 1. Ищем строку с заголовком 'occupation' ---
    start_idx = None
    for i, line in enumerate(lines):
        if "occupation" in line.lower():
            start_idx = i + 1  # данные начинаются со следующей строки
            break
    if start_idx is None:
        raise RuntimeError("Таблица occupation не найдена!")
    # --- 2. Читаем строки таблицы ---
    for line in lines[start_idx:]:
        line = line.strip()
        # Конец таблицы
        if not line:
            continue
        if line.startswith("Eigenenergies"):
            break
        parts = line.split()
        # Ожидаем минимум: subshell + 5 чисел
        if len(parts) < 6:
            continue
        subshell = parts[0]
        occ_str = parts[-1].replace("D", "E")  # Fortran → Python
        try:
            occupation = float(occ_str)
            occupations[subshell] = occupation
        except ValueError:
            pass
    return occupations




# ==== Чтение орбиталей ====
def read_rwfn_plot(filename = 'rwfn.plot', verbose = True):
    """
    Читает файл rwfn.plot.

    Returns
    -------
    r : ndarray
        Радиальная сетка (Bohr).
    orbital_names : list[str]
        Имена орбиталей.
    P_dict, Q_dict : dict[str, ndarray]
        Радиальные большие и малые компоненты.
    """

    with open(filename, 'r') as f:
        header = f.readline().strip()

    data = np.loadtxt(filename, skiprows=1)
    r = data[:, 0]
    cols = header.split()
    n_orbitals = (data.shape[1] - 1) // 2

    orbital_names = []
    P_dict = {}
    Q_dict = {}

    for i in range(n_orbitals):
        token = cols[1 + 2*i]
        if '(' in token and ')' in token:
            nm = token[token.find('(')+1:token.find(')')]
        else:
            nm = token.strip()

        orbital_names.append(nm)
        P_dict[nm] = data[:, 1 + 2*i]
        Q_dict[nm] = data[:, 1 + 2*i + 1]

    if verbose:
      print("Найденные орбитали (в порядке колонок):")
      print(orbital_names)
    return r, orbital_names, P_dict, Q_dict



# ===== Улучшение сетки r из 'rwfn.plot' =====
def refine_rwfn_data(r, P_dict, Q_dict, factor=4, r_max_mult=1.5,
                     interp_kind='linear', keep_original=True, verbose=False):
    """
    Сгущает и расширяет r-сетку, интерполируя P и Q, но при этом
    НЕ УДАЛЯЕТ исходные точки r (чтобы сохранить высокую плотность у ядра).

    Подход: объединяем исходную сетку r и более плотную дополнительную сетку,
    затем интерполируем P и Q на объединённой сетке.

    Параметры
    ----------
    r : ndarray
        исходная r-сетка (Bohr), строго неубывающая.
    P_dict, Q_dict : dict[str, ndarray]
        радиальные компоненты, ключи = имена орбиталей.
    factor : int
        во сколько раз примерно увеличить число точек (приблизительно).
    r_max_mult : float
        множитель для r_max (новая максимальная точка будет r[-1] * r_max_mult).
    interp_kind : str
        тип интерполятора ('linear' рекомендован).
    keep_original : bool
        если True — сохраняем все исходные точки r в итоговой сетке.
    verbose : bool
        печатать предупреждения/информацию.

    Возвращает
    -------
    r_new : ndarray
        объединённая упорядоченная сетка (Bohr).
    P_new : dict
        интерполированные P на r_new.
    Q_new : dict
        интерполированные Q на r_new.
    """
    # ---- валидация ----
    r = np.asarray(r)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("r должен быть одномерным массивом длины >= 2")
    if np.any(np.diff(r) < 0):
        raise ValueError("r должен быть монотонно неубывающим")
    r_min = r[0]
    r_max = r[-1]
    r_max_new = float(r_max) * float(r_max_mult)

    # ---- создаём дополнительную сетку (плотную) ----
    n_new = int(len(r) * factor)
    if n_new < len(r):
        n_new = len(r)
    # плоская плотная сетка (линейно); можно заменить на логspace, но union сохраняет малые r
    r_dense = np.linspace(r_min, r_max_new, n_new)

    # ---- объединяем исходную и дополнительную сетки (сохраняем исходные точки) ----
    if keep_original:
        r_new = np.union1d(r, r_dense)   # упорядоченная уникальная сетка
    else:
        r_new = r_dense

    # ---- интерполируем P и Q на новую сетку ----
    P_new = {}
    Q_new = {}
    for nm in P_dict.keys():
        # интерполяторы: fill_value=0 для точек за пределами исходного р (tail)
        Pi = interp1d(r, P_dict[nm], kind=interp_kind,
                      bounds_error=False, fill_value=0.0, assume_sorted=True)
        Qi = interp1d(r, Q_dict[nm], kind=interp_kind,
                      bounds_error=False, fill_value=0.0, assume_sorted=True)
        Pn = Pi(r_new)
        Qn = Qi(r_new)
        P_new[nm] = Pn
        Q_new[nm] = Qn

    # ---- диагностика: предупреждение, если исходный хвост был заметен ----
    if verbose:
        tail_vals = [abs(P_dict[nm][-1]) + abs(Q_dict[nm][-1]) for nm in P_dict.keys()]
        max_tail = max(tail_vals)
        if max_tail > 1e-6:
            print(f"[refine_rwfn_data] предупреждение: max |P|+|Q| at r_max = {max_tail:.2e}. "
                  "Если это >1e-6, лучше запросить rwfn.plot с большим r_max или увеличить r_max_mult.")
        print(f"[refine_rwfn_data] исходная len(r)={len(r)}, новая len(r_new)={len(r_new)}")
    return r_new, P_new, Q_new
