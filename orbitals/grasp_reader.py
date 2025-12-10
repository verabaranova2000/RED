# функции для чтения выходных файлов GRASP
import numpy as np


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

def c(filename, verbose = True):
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
