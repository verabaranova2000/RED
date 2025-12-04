"""
cif_symmetry.py

Функции для обработки симметрии кристаллов на основе CIF-файлов:
- получение матриц симметрии;
- работа с точечной и пространственной группами;
- удаление дубликатов операций симметрии.
"""
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from .cif_extract import keyword_value

 
def unique_op_matrix(ops, tol=1e-8):
    """
    Удаление дубликатов операций симметрии [r0, G] с учетом погрешности.
    
    Parameters
    ----------
    ops : list of [numpy.ndarray, numpy.ndarray]        ← Список операций симметрии [r0, G].
    tol : float                                         ← Допуск для сравнения элементов матрицы.

    Returns
    -------
    unique_ops : list of [numpy.ndarray, numpy.ndarray]  ← Список уникальных операций.
    """
    seen = set()
    unique_ops = []
    for r0, G in ops:
        key = tuple(np.round(G.flatten(), decimals=int(-np.log10(tol))))   ## округляем для стабильного сравнения
        if key not in seen:
            seen.add(key)
            unique_ops.append([r0.copy(), G.copy()])
    return unique_ops


# Функция для получения матриц операций симметрии пространственной группы кристалла
def get_symmetry_matrix_of_crystal_lattice(CIF_file):
  """
  Извлекает операции симметрии из CIF-файла кристаллической структуры. 
    
  Если CIF-файл содержит явный список симметрий (обычно под ключом '_space_group_symop_operation_xyz'),
  функция считывает их напрямую. Если список отсутствует, функция автоматически определяет
  номер пространственной группы и запрашивает симметрии из базы данных через pymatgen.

  Возвращает список операций в виде пар [r0, G], где:
    - r0 (numpy.ndarray, shape (3,)) — вектор трансляции (сдвиг);
    - G (numpy.ndarray, shape (3, 3)) — матрица поворота.
  Порядок:
    r1 = G @ r + r0, где r — исходный вектор координат атома, r1 — преобразованный.
  Аргументы:
  ----------
  CIF_file : list of str
      CIF-файл, прочитанный как список строк.
  Возвращает:
  -----------
  List[[numpy.ndarray, numpy.ndarray]]
      Список операций симметрии в формате [r0, G].
  """
  # 1. Извлекаем операции симметрии
  for line in CIF_file:                                                           # Поиск начала списка операций симметрии (строка типа ' 1   x,y,z\n')
    if line.replace(' ', '').replace('\t','').replace(',', '')=='1xyz\n':
      i0=CIF_file.index(line)
      break
    else: i0 = None  # Если не найдено

  symmetry=[]
  if i0 is not None:
    i=0
    if i0+i+1<len(CIF_file):
      while (i0 + i) < len(CIF_file) and len(CIF_file[i0+i].split())==2:
        if CIF_file[i0+i].split()[0]==str(i+1):
          symmetry.append(CIF_file[i0+i].split()[1:][0].split(','))
          i=i+1
  # 1b. Если не нашли операций — получаем их из pymatgen по номеру группы
  if not symmetry:
    spg_number = keyword_value(CIF_file, "_symmetry_Int_Tables_number")
    if spg_number is None:
      raise ValueError("Не удалось найти операции симметрии и номер группы в CIF")
    spg = SpaceGroup.from_int_number(int(spg_number))
    symmetry = [op.as_xyz_str().replace(' ', '').split(',') for op in spg.symmetry_ops]

  # 2. Преобразование к формату [r0, G]
  # (Матричное представление операций симметрии  (https://mypresentation.ru/presentation/1549492142_matricy))
  # Операция симметрии: r1=G*r+r0, где G - матрица поворота, r0 - вектор трансляции.
  # Выделим G и r0 для каждой операции симметрии.
  # a) Найтем векторы трансляции r0. Для этого занулим матрицу G:
  x=0
  y=0
  z=0
  Operations_symmetry=[]              # Список операций симметрии (набор матриц)
  for sym in symmetry:
    x0 = eval(sym[0])
    y0 = eval(sym[1])
    z0 = eval(sym[2])
    r0 = np.array([x0, y0, z0])
    Operations_symmetry.append([r0])

  # b) Найдем матрицы поворота. Для этого обнулим векторы трансляций r0:
  x=np.array([1, 0, 0])
  y=np.array([0, 1, 0])
  z=np.array([0, 0, 1])

  for i, sym in enumerate(symmetry):
    x1 = eval(sym[0])
    y1 = eval(sym[1])
    z1 = eval(sym[2])
    Gx = x1 - Operations_symmetry[i][0][0]
    Gy = y1 - Operations_symmetry[i][0][1]
    Gz = z1 - Operations_symmetry[i][0][2]
    G = np.array([Gx, Gy, Gz])
    Operations_symmetry[i].append(G)

  return Operations_symmetry                     # Получили вложенный список, где каждый подсписок соответствует определенной операции симметрии и включает в себя вектор трансляции r0 и матрицу поворота G


# Функция для получения матриц операций симметрии решетки в обратном пространстве (только точечная группа без векторов трансляций)
def get_symmetry_matrix_of_reciprocal_lattice(Operations_symmetry):
    """
    Получение операций симметрии обратной решётки (точечная группа).
    """
    ops = copy.deepcopy(Operations_symmetry)   # Делаем глубокую копию, чтобы не менять исходный список
    # 1. Зануляем трансляции
    for op in Operations_symmetry:
        op[0] = np.zeros(3)

    # 2. Удаляем дубликаты
    Operations_symmetry = unique_op_matrix(Operations_symmetry)

    # 3. Добавляем инверсию
    I = -np.eye(3)
    Operations_symmetry_I = [[op[0], op[1] @ I] for op in Operations_symmetry]
    Operations_symmetry += Operations_symmetry_I

    # 4. Снова удаляем дубликаты
    Operations_symmetry = unique_op_matrix(Operations_symmetry)
    return Operations_symmetry
