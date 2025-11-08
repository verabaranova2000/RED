import numpy as np
import os

## ============= Сохранение ============= 
def save_bragg_positions(array, filename=None, phase_object=None):
    """
    Сохраняет массив Bragg-позиций в файл.
    
    Параметры
    ----------
    array : list of list            ← Массив Bragg-позиций.
    filename : str | None           ← Имя файла для сохранения (используется, если phase_object не указан).
    phase_object : Phase | None     ← Объект фазы, из которого автоматически берётся имя файла и путь.

    """

    ## --- Если передан phase_object — формируем путь автоматически  ---
    if phase_object is not None:
      filename = f"{phase_object.prefix}bragg_positions.txt"
    else:
      if filename is None: raise ValueError("Не указан ни filename, ни phase_object.")
    # Сохраняем
    with open(filename, 'w') as f:
      for row in array: ## Форматируем: числа с 6 знаками после точки, строки как есть
        line = '\t'.join(f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x) for x in row)
        f.write(line + '\n')
    print(f"[INFO] Bragg positions saved as '{filename}'")



## ============= Загрузка ============= 
def load_bragg_positions(filename):
    """
    Загружает массив Bragg-позиций из текстового файла.
    
    Параметры
    ----------
    filename : str   ← Имя файла для чтения.
    
    Возвращает
    -------
    list of list     ← Восстановленный массив с сохранением формата чисел и строк.
    """
    array = []
    with open(filename, 'r') as f:
      for line in f:
        elems = line.strip().split('\t')
        row = []
        for e in elems:       # Пробуем преобразовать к float, если не получилось — оставляем строку
          try:                row.append(float(e))
          except ValueError:  row.append(e)
        array.append(row)
    for row in array:    ## Преобразуем первые 5 элементов в int (h, k, l, multiplicity, KPhase)
      for i in range(5): row[i] = int(round(row[i]))
    return array


## ============= Путь к файлу ============= 
def get_bragg_file(project_name, phase_prefix, data_root="RED/examples"):
    """
    Возвращает путь к файлу bragg_positions для заданного проекта и фазы,
    если файл существует.
    """
    project_dir = os.path.join(data_root, project_name)
    filename = f"{phase_prefix}bragg_positions.txt"
    full_path = os.path.join(project_dir, filename)
    return full_path if os.path.exists(full_path) else None


# Сохранили
# save_bragg_positions('bragg_positions.txt', bp_new)

# Загрузили обратно
# bp_new_loaded = load_bragg_positions('bragg_positions.txt')

# Проверка
# print(len(bp_new_loaded) == len(bp_new))  # должно быть True