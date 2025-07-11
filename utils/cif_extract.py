"""
cif_extract.py

Функции для извлечения данных из CIF-файла, представленного как список строк.
Включает доступ к отдельным ключам, чтение таблиц loop_ и получение параметров атомов.
"""

from typing import List, Optional, Dict


# === ОБРАБОТКА ОДИНОЧНЫХ КЛЮЧЕЙ ===
def keyword_value(CIF_data: List[str], key_word: str) -> Optional[str]:
  """
  Возвращает значение для ключа key_word (например, '_cell_length_a') из CIF-данных.
  """
  for line in CIF_data:
    if line.strip().startswith(key_word):            # Строгое соответствие ключа — строка должна начинаться с него
      parts = line.strip().split(maxsplit=1)         # Убираем перевод строки и лишние пробелы
      if len(parts) == 2:
        raw_value = parts[1].strip().strip("'\"")    # Убираем кавычки
        if raw_value not in ['.', '?', '']:
          return raw_value
    return None                                      # Если не найдено или значение пустое


# === РАБОТА С ТАБЛИЦАМИ loop_ ===

def find_loop_by_row(CIF_file: List[str], name_row: str) -> int:                 # На вход - название какого-то столбца интересующей таблицы в формате 'atom_site_disorder_assembly' (т.е. без лишних символов вначале и в конце)
  """
  Находит индекс начала таблицы loop_, содержащей указанный столбец (name_row).  (Функция для поиска таблицы loop_ (по названию одного из столбцов))
  """
  name_row=' _'+name_row+'\n'
  i0=CIF_file.index(name_row)
  while CIF_file[i0]!='loop_\n':
    i0=i0-1
  return i0                              # На вход: i0 - номер строки loop_


# Функция для чтения таблицы loop_
def loop_(CIF_file: List[str], index_of_loop: int) -> List[List[str]]:            # На вход: i0 - номер строки loop_
  """
  Возвращает таблицу loop_ как список списков (один список на столбец).
  """
  i0=index_of_loop+1
  k=0
  table=[]
  # 1. Чтение заголовков таблицы
  while i0+k<len(CIF_file) and '_' in CIF_file[i0+k]:
    table.append([CIF_file[i0+k].replace('\n', '').replace(' ','')])
    k=k+1
  # 2. Чтение данных (точек)
  i0=i0+k
  k=0
  while i0+k<len(CIF_file) and len(table)==len(CIF_file[i0+k].split()):
    for c in range(len(table)):                                            # Добавляем значение в каждую колонку для текущей точки
      table[c].append(CIF_file[i0+k].split()[c])
    k=k+1
  return table


# Функция для получения таблицы (по названию одного из столбцов)
def get_table(CIF_file: List[str], name_row: str) -> Dict[str, List[str]]:                                          # На вход - название одного из столбцов типа 'atom_site_label'
  """
  Возвращает таблицу loop_ в виде словаря: ключ — имя столбца, значение — список строк.
  """
  i0=find_loop_by_row(CIF_file, name_row)
  table=loop_(CIF_file, i0)
  table_dict = {key: value for key, value in map(lambda x: (x[0][1:],x[1:]), table)}
  return table_dict

### Получаем таблицу типа словарь:
#{'_atom_site_label': ['Ca1', 'F1'],
# '_atom_site_type_symbol': ['Ca', 'F'],
# '_atom_site_fract_x': ['0', '0.25'],
# '_atom_site_fract_y': ['0', '0.25'],
# '_atom_site_fract_z': ['0', '0.25'],
# '_atom_site_adp_type': ['Uiso', 'Uiso'],
# '_atom_site_U_iso_or_equiv': ['-0.0001', '0.0001'],
# '_atom_site_site_symmetry_multiplicity': ['4', '8'],
# '_atom_site_occupancy': ['1', '1'],
# '_atom_site_calc_flag': ['d', 'd'],
#'_atom_site_refinement_flags': ['.', '.'],
# '_atom_site_disorder_assembly': ['.', '.'],
# '_atom_site_disorder_group': ['.', '.']}


# === ОБОБЩЁННАЯ ФУНКЦИЯ ДЛЯ ДОСТУПА К ЯЧЕЙКЕ ===

### Функция для получения значения параметра из ячейки таблицы (напр., occ для атома Ca)
def get_value_for_atom(CIF_file: List[str], parametr: str, atom: str) -> Optional[str]:
  """
  Возвращает значение параметра (parametr) для атома (atom) из таблицы CIF.
  Например: parametr='atom_site_occupancy', atom='Ca1'
  """
  table=get_table(CIF_file,parametr)
  for key,value in table.items():
    if atom in value:
      id=value.index(atom)
  x=table.get(parametr)[id] if table.get(parametr)[id] not in ['.', '?'] else None
  return x