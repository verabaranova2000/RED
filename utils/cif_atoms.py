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

# Функция для поиска неэквивалентных атомных позиций 
def XYZ_all_atoms(CIF_file):                                                                     # Для удаления всех символов из строки, кроме букв  
  """
  Генерация списка всех неэквивалентных атомных координат из CIF-файла с учётом симметрии.
  Аргументы:
      CIF_file (list of str): Содержимое CIF-файла, представленное как список строк.
  Возвращает:
      list of list: Список атомных координат в формате:
          [[x1, y1, z1, 'Element1'],
           [x2, y2, z2, 'Element2'],
           ...]
          где (x, y, z) — дробные координаты в пределах элементарной ячейки,
          'Element' — обозначение химического элемента.
  Описание:
      Функция выполняет следующие шаги:
          1. Извлекает атомные координаты из блока loop_.
          2. Применяет все операции симметрии, заданные в CIF, к каждой атомной позиции.
          3. Приводит координаты к первой элементарной ячейке (значения от 0 до 1).
          4. Удаляет эквивалентные атомы на основе заданной точности сравнения координат.
  """

  Operations_symmetry=get_symmetry_matrix_of_crystal_lattice(CIF_file)                               # Получаем матрицы операций симметрии
  ################# Извлечение координат атомов из CIF
  error0=0.005                     # Точность задания координат
  atom_site=[]
  i0=min(CIF_file.index(' _atom_site_label\n'), CIF_file.index(' _atom_site_type_symbol\n'))
  i=0
  while CIF_file[i0+i].split()[0][0]=='_' and i<20:                               # Заголовки таблицы loop_ _atom_site_
    atom_site.append([CIF_file[i0+i]])
    i=i+1

  while len(atom_site)==len(CIF_file[i0+i].split()):                              # Заполнение таблицы loop_ _atom_site_
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
  for N in range(1, len(atom_site[N_atom])):                             # Находим координаты каждого атома
    name=atom_site[N_atom][N]
    name=re.sub("[^A-Za-z]", "", name)                # Убираем степень окисления, если она указана, напр., Ca2+ (Удаляем все символы из строки, кроме букв)
    Atom_names.append(name)
    Atoms.append(np.array([float(atom_site[N_x][N]),float(atom_site[N_y][N]),float(atom_site[N_z][N])]))
    Error.append(np.array([error0,error0,error0]))

  All_atoms=[list(t) for t in zip(Atom_names, Atoms)]             # Массив с координатами атомов, в который будем добавлять новые координаты, полученные операциями симметрии

  # Размножение атомных позиций
  for N in range(len(Atoms)):                                                 # Пробегаемся каждому атому
    for i in range(len(Operations_symmetry)):                                 # Размножаем атомы всеми операциями симметрии
      new_xyz=Operations_symmetry[i][1].dot(Atoms[N][..., None])              # Умножение столбца (атомную позицию) на оператор поворота G
      new_xyz=new_xyz.transpose()[0]                                          # Превращаем столбец в строку
      new_xyz=new_xyz+Operations_symmetry[i][0]                               # Прибавляем вектор трансляции
      All_atoms[N].append(new_xyz)


  # Исключение эквивалентных атомных позиций (точек (x,y,z),

        # Сначала вычтем векторы постоянный решетки, чтобы получить координаты в пределах одной элементарной ячейки
  for N in range(len(All_atoms)):                                             # Пробегаемся по каждому типу атомов
    for M in range(1,len(All_atoms[N])):                                      # Пробегаемся по всем позициям данного типа атомов
      for i in range(3):                                                      # Пробегаемся по каждой координате x,y,z
        if All_atoms[N][M][i]<0:
          while All_atoms[N][M][i]<0:
            All_atoms[N][M][i]=All_atoms[N][M][i]+1
        if All_atoms[N][M][i]>=1:
          while All_atoms[N][M][i]>=1:
            All_atoms[N][M][i]=All_atoms[N][M][i]-1

      # Удалим совпадающие позиции (точки (x,y,z), совпадающие в пределах точности)
  Unequal_atoms=[]

  for N in range(len(All_atoms)):                                             # Пробегаемся по каждому типу атомов
    Unequal_atoms.append([All_atoms[N][0], All_atoms[N][1]])

    for M in range(1,len(All_atoms[N])):                                      # Пробегаемся по всем позициям данного типа атомов
      coincidence=0                                                           # Количество совпадений M-позиции с набором из Unequal_atoms
      for K in range(1,len(Unequal_atoms[N])):
        if (Unequal_atoms[N][K]-Error[N]<=All_atoms[N][M]).all() and (All_atoms[N][M]<=Unequal_atoms[N][K]+Error[N]).all():
          coincidence=coincidence+1
      if coincidence==0:
        Unequal_atoms[N].append(All_atoms[N][M])


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


  # Создание массива требуемого формата:
  XYZ=[]
  for N in range(len(Unequal_atoms)):
    for M in range(1,len(Unequal_atoms[N])):
      XYZ.append([Unequal_atoms[N][M][0],Unequal_atoms[N][M][1],Unequal_atoms[N][M][2],Unequal_atoms[N][0]])

  return XYZ