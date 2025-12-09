"""
Модуль: cif_atoms.py
Назначение:
    Извлечение атомных координат из CIF-файлов, включая:
    - учёт симметрий кристаллической решётки;
    - фильтрацию эквивалентных атомов;
    - подготовку атомных координат к расчётам структурных амплитуд.

Содержит:
    - XYZ_all_atoms(CIF_file)
    - find_in_list_of_list(mylist, char)
    - (опционально) get_symmetry_matrix_of_crystal_lattice(CIF_file)
"""
import re 
import numpy as np
from .cif_symmetry import get_symmetry_matrix_of_crystal_lattice

# Функция для поиска индексов элемента во вложенном списке
def find_in_list_of_list(mylist, char):
  """
  Поиск индексов элемента во вложенном списке.
  Аргументы:
      mylist (list of list): Двумерный список (например, таблица CIF).
      char (str): Элемент (например, название колонки), который нужно найти.
  Возвращает:
      tuple: (i, j), где i — индекс подсписка, j — индекс элемента в подсписке.
  Исключения:
      ValueError: Если элемент не найден.
  """
  for sub_list in mylist:
      if char in sub_list:
        return (mylist.index(sub_list), sub_list.index(char))
  raise ValueError("'{char}' is not in list".format(char = char))



import numpy as np

def allclose_PBC(pos1, pos2, atol=1e-5, eps_mult=100):
    """
    Проверка эквивалентности двух fractional координат с учётом PBC, с учётом
    тонкостей представления чисел с плавающей точкой.

    Parameters
    ----------
    pos1, pos2 : array-like, shape (3,)
        Fractional coordinates (x, y, z). Могут быть вне [0,1) — функция приводит %1.0.
    atol : float, optional
        Абсолютная погрешность (fractional units). По умолчанию 1e-5.
    eps_mult : int, optional
        Множитель для машинного эпсилон (np.finfo(float).eps). Значение eps_mult=100
        обычно достаточно, чтобы избежать ложных отрицательных срабатываний из-за
        бинарной арифметики (можно уменьшить/увеличить при необходимости).

    Returns
    -------
    bool
        True, если позиции эквивалентны по PBC в пределах порога atol (с «подушкой» eps_mult*eps), иначе False.

    Notes
    -----
    - Используется минимальный образ вдоль каждой координаты: d = min(|dx|, 1-|dx|).
    - Параметр eps_mult нужен из-за того, что операции вида 1.0 - 0.98 могут вернуть
      значение немного > 0.02, и простая проверка <= atol даст ложный False.

    Example:
    -----
    >>> a = [0.99, 0.99, 0.5]
    >>> b = [0.01, 0.01, 0.5]
    >>> print(allclose_PBC(a, b, atol=0.02))   # -> True (ожидаемо)
    >>> print(allclose_PBC(a, b, atol=0.005))  # -> False (ожидаемо)
    """
    pos1 = np.asarray(pos1, dtype=float) % 1.0
    pos2 = np.asarray(pos2, dtype=float) % 1.0

    delta = np.abs(pos1 - pos2)
    delta = np.minimum(delta, 1.0 - delta)

    # "Подушка" для чисел с плавающей точкой
    eps = np.finfo(float).eps
    thresh = float(atol) + eps_mult * eps

    return np.all(delta <= thresh)



# Функция для поиска неэквивалентных атомных позиций 
# ---------- ФУНКЦИЯ (с минимальными правками) ----------
def XYZ_all_atoms(CIF_file, return_inequiv=False):
  """
  Теперь удаление дубликатов выполняется только внутри каждой группы All_atoms[N], 
  то есть атомы разных исходных позиций (и разных элементов) не смешиваются.

  Example
  ------
  
  XYZ, ineq = XYZ_all_atoms(data, return_inequiv=True)
  for i, start in enumerate(ineq):
      end = ineq[i+1] if i+1 < len(ineq) else len(XYZ)
      print(f"rep #{i} at idx {start}: multiplicity = {end-start}, rep = {XYZ[start]}")
  """
  CIF_file=CIF_file
  Operations_symmetry=get_symmetry_matrix_of_crystal_lattice(CIF_file)
  ################# Извлечение координат атомов из CIF
  error0=0.005
  atom_site=[]
  i0=min(CIF_file.index(' _atom_site_label\n'), CIF_file.index(' _atom_site_type_symbol\n'))
  i=0
  while CIF_file[i0+i].split()[0][0]=='_' and i<20:
    atom_site.append([CIF_file[i0+i]])
    i=i+1

  while len(atom_site)==len(CIF_file[i0+i].split()):
    for k in range(len(atom_site)):
      atom_site[k].append(CIF_file[i0+i].split()[k])
    i=i+1

  # Формирование исходного массива атомных позиций (не размноженных элементами симметрии)
  Atom_names=[]            # Названия атомов
  Atoms=[]                 # Координаты атомов
  Error=[]                 # Точность задания координат атомов
  N_atom=find_in_list_of_list(atom_site,' _atom_site_type_symbol\n')[0]
  N_x=find_in_list_of_list(atom_site, ' _atom_site_fract_x\n')[0]
  N_y=find_in_list_of_list(atom_site, ' _atom_site_fract_y\n')[0]
  N_z=find_in_list_of_list(atom_site, ' _atom_site_fract_z\n')[0]
  for N in range(1, len(atom_site[N_atom])):
    name=atom_site[N_atom][N]
    name=re.sub("[^A-Za-z]", "", name)
    Atom_names.append(name)
    Atoms.append(np.array([float(atom_site[N_x][N]),float(atom_site[N_y][N]),float(atom_site[N_z][N])]))
    Error.append(np.array([error0,error0,error0]))

  All_atoms=[list(t) for t in zip(Atom_names, Atoms)]

  # Размножение атомных позиций
  for N in range(len(Atoms)):                                                 # Пробегаемся каждому атому
    for i in range(len(Operations_symmetry)):                                 # Размножаем атомы всеми операциями симметрии
      new_xyz=Operations_symmetry[i][1].dot(Atoms[N][..., None])              # Умножение столбца (атомную позицию) на оператор поворота G
      new_xyz=new_xyz.transpose()[0]                                          # Превращаем столбец в строку
      new_xyz=new_xyz+Operations_symmetry[i][0]                               # Прибавляем вектор трансляции
      All_atoms[N].append(new_xyz)

  # Приводим все сгенерированные позиции в [0,1)
  for N in range(len(All_atoms)):
    for M in range(1, len(All_atoms[N])):
      # заменяем while на быстрый модуль
      All_atoms[N][M] = All_atoms[N][M] % 1.0

  # -------- исключение эквивалентных в пределах каждой группы (НЕ смешивая неэквивалентные исходные атомы) --------
  Unequal_atoms=[]

  for N in range(len(All_atoms)):
    # minimal change: не предзаполняем второй позицией, оставляем только имя в [0]
    Unequal_atoms.append([All_atoms[N][0]])   # [element_name, pos1?, pos2?, ...]
    
    for M in range(1, len(All_atoms[N])):    # пробегаем по всем сгенерированным позициям текущего исходного атома
      posM = np.array(All_atoms[N][M], dtype=float)
      coincidence = 0
      # --- Перебираем уже накопленные уникальные позиции (они лежат в Unequal_atoms[N][1:])
      for K in range(1, len(Unequal_atoms[N])):
        posK = np.array(Unequal_atoms[N][K], dtype=float)
        #if np.allclose(posK, posM, atol=Error[N].max(), rtol=0):                # сравнение через allclose с абсолютной погрешностью Error[N]    
        if allclose_PBC(posK, posM, atol=Error[N].max()):
          coincidence += 1
          break
      if coincidence == 0: 
        Unequal_atoms[N].append(posM)                                           # добавляем новую уникальную позицию прямо в группу N


  ######################################################
  # Нужен массив следующего формата:                      (Для последующего расчета структурных амплитуд)
  # Координаты атомов
  #XYZ=[[0,0,0,"Te"],
  #     [0.333333,0.666667,0.790108, "Te"],
  #     [0.666667,0.333333,-0.790108, "Te"],
  #     [0.333333,0.666667,0.411045,"Te"],
  #     [0.666667,0.333333,-0.411045, "Te"],
  #     [0.333333,0.666667,0.112658,"Sb"],
  #     [0.666667,0.333333,0.112658,"Sb"],
  #     [0,0,0.332252,"Ge"],
  #     [0,0,-0.332252,"Ge"]]
  #######################################################

  
  # ---------- формирование XYZ и индексов неэквивалентных ----------
  XYZ=[]
  inequivalent_indices = []
  for N in range(len(Unequal_atoms)):
    inequivalent_indices.append(len(XYZ))                                       # индекс первой неэквивалентной позиции данного атома
    for M in range(1, len(Unequal_atoms[N])):
      p = Unequal_atoms[N][M]
      XYZ.append([float(p[0]), float(p[1]), float(p[2]), Unequal_atoms[N][0]])

  if return_inequiv:
    return XYZ, inequivalent_indices
  else:
    return XYZ