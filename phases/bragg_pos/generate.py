from .extinction import extinction_rules_by_number
from phases.utils_cryst.lattice import d_hkl
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict


## ===== ✪ (i) Звезда hkl (с учетом правил погасания) =====
def get_star_hkl(hkl, operations_symmetry, spacegroup_number=None, add_inversion=True, verbose=False):
    """
    Получение звезды для данного hkl по операциям симметрии.

    Параметры
    ----------
    hkl : array-like длины 3                    Индексы отражения (h, k, l).
    operations_symmetry : list of [t, R]        Список операций симметрии в формате [вектор трансляции t, матрица R].
                                                  t здесь хранится для совместимости, но в звезде hkl не используется.
    add_inversion : bool                        Добавить ли инверсию (если её нет в списке).
    verbose : bool                              Печатать промежуточный результат (для отладки).

    Возвращает
    -------
    list of dict
        Каждый элемент: {
            'hkl': np.array([h, k, l], dtype=int),
            'phase': комплексный фазовый множитель,
            'op_index': индекс операции в списке (для трассировки),
            'R': матрица преобразования,
            't': вектор трансляции (как есть из входа)
        }
        Дубликаты НЕ удаляются (для отладки).
    """
    hkl = np.asarray(hkl, dtype=int)
    if hkl.shape != (3,):    raise ValueError("hkl должен быть массивом длины 3")

    ops = list(operations_symmetry)  ## Берём копию операций
    if add_inversion:                ## Проверяем наличие инверсии
        I = -np.eye(3, dtype=int)
        has_inversion = any(np.array_equal(R, I) for _, R in operations_symmetry)

        if not has_inversion:
          if verbose:  print("Инверсия не найдена — добавляем её.")
          for t, R in operations_symmetry: ops.append([t.copy(), R @ I])  ## добавляем новые операции вида R*I
        else: 
           if verbose:  print("Инверсия уже есть — не добавляем.")
    
    rule = None   # Правило экстинкции
    if spacegroup_number is not None:  rule = extinction_rules_by_number.get(spacegroup_number, None)
    ## ---- Применяем все операции ------
    hkl_star = []
    for iop, (t, R) in enumerate(ops):
        hkl_new = R @ hkl                            ## Применение матрицы к hkl
        hkl_new = np.rint(hkl_new).astype(int)       ## На случай численных артефактов
        if rule is not None:                         ## Применяем правило, если оно есть
            try:
                if not rule(*hkl_new):
                    if verbose:       print(f"op {iop+1:2d}: {hkl_new} запрещено   ⨯")
                    continue
            except Exception as e:
                if verbose:           print(f"⚠️ ошибка при проверке {hkl_new}: {e}")
                continue
        phase = np.exp(2j * np.pi * np.dot(hkl, t))  ## Фазовый множитель exp(2πi * hkl·t)
        hkl_star.append({'hkl': hkl_new,  'phase': phase,
                         'op_index': iop, 'R': R, 't': t})
        if verbose:
            print(f"op {iop+1:2d}: hkl -> {hkl_new}, "
                  f"phase = {phase.real:+.3f}{phase.imag:+.3f}j")
    return hkl_star



# ===== ✪ (ii) Вспомогательные для get_vals_for_hkl ======
# ---------------------------
def get_unique_hkl(hkl_list):
    """
    Возвращает список уникальных hkl (по tuple(hkl)).
    """
    unique = []
    seen = set()
    for r in hkl_list:
        key = tuple(r['hkl'])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# ---------------------------
def print_hkl_table(hkl_list, hkl_label=None, multiplicity=None, unique_only=False, title=None):
    """
    Печать таблицы эквивалентных hkl с дополнительной колонкой F^2.

    Параметры
    ----------
    hkl_list : list of dict                  Список словарей с ключами 'hkl', 'd', 'two_theta', 'phase', 'op_index' и опционально 'F2'.
    hkl_label : tuple or array, optional     Метка representative hkl. Если None — выбирается первый уникальный hkl в списке.
    multiplicity : int, optional             Количество уникальных hkl. Если None — вычисляется автоматически.
    unique_only : bool                       True — печатать только уникальные hkl (первое вхождение каждого tuple(h,k,l)).
                                             False — печатать все записи (включая дубликаты).
    title : str, optional                    Заголовок таблицы.
    """
    ## ----- Фильтрация уникальных при необходимости -------
    if unique_only:  table_rows = get_unique_hkl(hkl_list)
    else:            table_rows = list(hkl_list)

    ## ------ Multiplicity: посчитаем, если не передали (по уникальным hkl в исходном списке)
    if multiplicity is None:
        uniq_keys    = {tuple(int(xi) for xi in r['hkl']) for r in hkl_list}
        multiplicity = len(uniq_keys)

    ## ------ hkl_label:  если не передали, возьмём первый уникальный
    if hkl_label is None:  hkl_label = '--'
        #if len(table_rows)>0:  hkl_label = tuple(int(x) for x in table_rows[0]['hkl'])
        #else:                  hkl_label = None

    ## ------- Заголовок и шапка ---------
    if title is None: title = "Все эквивалентные hkl" + (" (без дубликатов):" if unique_only else " (с дубликатами):")
    print(title)
    hdr = f"{'op':>3} {'h':>4} {'k':>4} {'l':>4} {'d':>10} {'2θ':>10} {'Re(phase)':>10} {'Im(phase)':>10} {'F^2':>12}"
    print(hdr)
    print("-" * len(hdr))

    ## ------- Печать строк  ---------
    for r in table_rows:
        h, k, l = (int(r['hkl'][i]) for i in range(3))     # гарантируем целые индексы
        re_ph = float(np.real(r.get('phase', 1.0+0.0j)))
        im_ph = float(np.imag(r.get('phase', 1.0+0.0j)))

        ## d и two_theta — предполагаем, что есть, но на всякий случай fallback
        d = float(r.get('d', float('nan')))
        two_theta = float(r.get('two_theta', float('nan')))

        ## F^2 — может отсутствовать (в этом случае вывожу '--')
        F2_val = r.get('F2', None)
        try:               F2s = f"{float(F2_val):12.4f}" if F2_val is not None and not (isinstance(F2_val, float) and math.isnan(F2_val)) else f"{'--':>12}"
        except Exception:  F2s = f"{'--':>12}"

        op_idx = int(r.get('op_index', -1))
        print(f"{op_idx:3d} {h:4d} {k:4d} {l:4d} {d:10.5f} {two_theta:10.4f}"
              f" {re_ph:10.4f} {im_ph:10.4f} {F2s}")
    print("-" * len(hdr))
    print("hkl_label: ", hkl_label)
    print("multiplicity:", multiplicity)

# ---------------------------
def group_by_d(rows, tol=1e-5):
    """
    Разбивает список словарей rows на группы по одинаковому d (с учётом tol).

    Параметры
    ----------
    rows : list of dict                Каждый dict содержит 'hkl', 'd', 'two_theta', 'phase', 'op_index'.
    tol : float                        Допуск для сравнения d (обычно маленький, например 1e-5 Å).

    Возвращает
    -------
    grouped : list of list of dict     Список групп, каждая группа — список рефлексов с одинаковым d.
    """
    grouped_dict = defaultdict(list)
    for r in rows:
        key = round(r['d']/tol)*tol       ## округляем d с точностью tol для ключа
        grouped_dict[key].append(r)       ## возвращаем список групп, отсортированных по d
    grouped = [grouped_dict[k] for k in sorted(grouped_dict.keys())]
    return grouped


# ---------------------------
def canonical_hkl(hkl_group):
    """
    Выбирает канонического представителя для группы эквивалентных hkl.
    Правила:
    1. Если есть отрицательные знаки — инвертируем так, чтобы первый ненулевой индекс был >= 0.
    2. Из полученных вариантов выбираем лексикографически минимальный.

    Параметры
    ----------
    hkl_group : list   Список троек hkl (напр., [np.array([1,0,1]), np.array([-1,0,-1])])
                       или список словарей {'hkl': np.array([...]), ...}
    Возвращает
    ----------
    tuple              Уникальный представитель группы (h, k, l)
    """
    normalized = []
    for item in hkl_group:
        if isinstance(item, dict): hkl = item["hkl"]       ## поддержка словарей
        else:                      hkl = item
        h, k, l = map(int, hkl)
        ## если первый ненулевой индекс отрицательный — меняем знак
        if h<0 or (h==0 and k<0) or (h==0 and k==0 and l<0):
            h,k,l = -h,-k,-l
        normalized.append((h, k, l))
    return min(normalized)       ## выбираем "минимальный" по лексикографическому порядку



## ========== Генерация hkl_data ==========
def generate_hkl_array(hkl_max, a,b,c,alpha,beta,gamma, two_theta_max, λ,
                       spacegroup_number, forbidden=False, verbose=True, include_hkl000=False):
    extinction_rule = extinction_rules_by_number.get(spacegroup_number, lambda h, k, l: True)
    hkl_list = []
    d_list = []
    two_theta_list = []
    stl_list = []
    # Если include_hkl000=True, то добавляем (0,0,0) вручную в начало списка
    if include_hkl000:
        hkl_list.append((0, 0, 0))
        d_list.append(np.inf)          # или np.nan, тк d для (0,0,0) физически не определено
        two_theta_list.append(0.0)
        stl_list.append(0.0)

    for h in range(-hkl_max, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if (h, k, l) == (0, 0, 0): continue
                # Разрешённые или запрещённые в зависимости от параметра
                allowed = extinction_rule(h, k, l)
                if (allowed and not forbidden) or ((not allowed) and forbidden):
                    d = d_hkl(h, k, l, a,b,c,alpha,beta,gamma)
                    try:
                        theta = math.asin(λ / (2 * d))
                    except ValueError:
                        continue
                    two_theta = 2 * theta * 180 / math.pi
                    if (two_theta_max is None) or (two_theta <= two_theta_max):
                        sin_theta_over_lambda = math.sin(theta) / λ
                        hkl_list.append((h, k, l))
                        d_list.append(d)
                        two_theta_list.append(two_theta)
                        stl_list.append(sin_theta_over_lambda)
    result = {'hkl': np.array(hkl_list),            # shape: (H, 3)
              'd': np.array(d_list),                # shape: (H,)
              '2theta': np.array(two_theta_list),   # shape: (H,)
              'stl': np.array(stl_list)}            # shape: (H,)
    if verbose and not forbidden: print(f"Количество отражений: {len(result['hkl'])}")
    if verbose and forbidden: print(f"Количество запрещенных отражений: {len(result['hkl'])}")
    return result



## ====== Генерация hkl-групп ======
def build_hkl_groups(phase_object, mode='allowed', individual=False,
                        hkl_max=10, two_theta_max=None, include_hkl000=False, verbose=True):
    """
    Генерирует hkl-группы по данным из объекта Phase.

    Параметры
    ----------
    phase_object : Phase                       ← Объект класса Phase, содержащий параметры ячейки, симметрию и длину волны.
    mode : {'allowed', 'forbidden', 'all'}     ← Режим генерации hkl:
          - 'allowed'   — разрешённые отражения (учитывает правила погасания)
          - 'forbidden' — только запрещённые отражения
          - 'all'       — все возможные hkl без проверки на правила погасания
    individual : bool, по умолчанию False      ← Если True — минует объединение симметрически эквивалентных рефлексов;
                                                 каждая группа будет состоять из одного hkl.
    hkl_max : int, по умолчанию 10             ← Максимальное значение индексов h, k, l.
    two_theta_max : float, optional            ← Максимальный угол 2θ (если требуется ограничение по углу дифракции).
    include_hkl000 : bool, по умолчанию False  ← Включать ли (0,0,0).
    verbose : bool                             ← Выводить ли отладочную информацию.

    Возвращает
    ----------
    list[list[dict]    ← Список групп, где каждая группа — это список словарей с параметрами рефлексов.
    
    Примеры
    ----------
    Режим a) разрешённые
             groups_allowed = build_hkl_groups(pr.Phase1, mode='allowed', individual=False)
    Режим b) запрещённые
             groups_forbidden = build_hkl_groups(pr.Phase1, mode='forbidden', individual=False)
    Режим c) все (без проверки симметрии)
             groups_all = build_hkl_groups(pr.Phase1, mode='all', individual=False)
    Режим d) индивидуальные — каждая группа = один hkl
             groups_individual = build_hkl_groups(pr.Phase1, mode='allowed', individual=True)
    """

    # --- Извлекаем параметры ячейки ---
    a, b, c, alpha, beta, gamma = [v.value for k, v in phase_object.param_cell.items()]
    λ = phase_object.wavelength

    # --- Настраиваем режим ---
    if mode == 'allowed':
      spacegroup_number = phase_object.spacegroup_number
      forbidden = False
    elif mode == 'forbidden':
      spacegroup_number = phase_object.spacegroup_number
      forbidden = True
    elif mode == 'all':
      spacegroup_number = None
      forbidden = False
    else:  raise ValueError(f"Неверный режим '{mode}'. Допустимо: 'allowed', 'forbidden', 'all'.")

    if verbose: 
      print(f"Генерация hkl_data в режиме: {mode.upper()}")
      print(f"  {'spacegroup_number':<18} = {spacegroup_number}")
      print(f"  {'forbidden':<18} = {forbidden}")
      print(f"  {'hklₘₐₓ':<18} = {hkl_max}")
      print(f"  {'2θₘₐₓ':<18} = {two_theta_max}")
      print(f"  {'individual':<18} = {individual}")
    # --- Генерация hkl_data ---
    hkl_data = generate_hkl_array(hkl_max=hkl_max, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                                  two_theta_max=two_theta_max, λ=λ, spacegroup_number=spacegroup_number,
                                  forbidden=forbidden, verbose=verbose, include_hkl000=include_hkl000)

    # --- Если individual=True → каждая группа состоит из одного hkl: генерируем звезду только по тождественной операции ---
    if individual:
        hkl_groups  = []
        identity_op = [np.zeros(3), np.eye(3)]    ## тождественная операция: нулевая трансляция и единичная матрица 3x3
        sym_ops_identity = [identity_op]
        for hkl in tqdm(hkl_data['hkl'], desc="Generating individual groups"):
          star_hkl = get_star_hkl(hkl, sym_ops_identity, spacegroup_number=None, add_inversion=False, verbose=False)  # Передаём в get_star_hkl только единичную операцию, чтобы не размножать hkl; убираем инверсию
          groups = get_vals_for_hkl(star_hkl, phase_object, verbose=False, unique_only=True)       # Вызов get_vals_for_hkl заполнит все необходимые поля, включая 'F2'
          # groups — список групп (обычно одна группа), каждый элемент grp — список словарей
          # Разбиваем каждую возвращённую группу на одиночные группы, сохраняя формат dict'ов
          for grp in groups:
            for member in grp: hkl_groups.append([member.copy()])  # оборачиваем в список, как в обычном формате
        if verbose:  print(f"\nСформировано {len(hkl_groups)} индивидуальных групп (по одному hkl в каждой).\n")
        return hkl_groups

    # --- Иначе: обычный режим с объединением эквивалентов ---
    all_groups_dict = {}
    for hkl in tqdm(hkl_data['hkl'], desc="Grouping hkl"):
      star_hkl = get_star_hkl(hkl, phase_object.symmetry_operations, spacegroup_number, verbose=False)
      groups = get_vals_for_hkl(star_hkl, phase_object, verbose=False, unique_only=True)
      for grp in groups:
        hkl_label = grp[0]['hkl_label']
        key = (hkl_label,)
        if key not in all_groups_dict: all_groups_dict[key] = grp
    hkl_groups = list(all_groups_dict.values())
    if verbose:  print(f"\nСформировано {len(hkl_groups)} групп после объединения по симметрии.\n")
    return hkl_groups




## ======== Генерация массива bragg_positions ============= 
def build_bragg_positions_from_groups(groups, KPhase, two_theta_max=5.5):
    """
    Собирает итоговый массив Bragg-позиций из групп эквивалентных hkl.

    Параметры
    ----------
    groups : list of list of dict   ← Каждая группа = список словарей, каждый dict содержит:
                                      'hkl', 'd', 'two_theta', 'hkl_label', 'multiplicity', 'F2', ...
    KPhase : int                    ← Номер фазы.
    two_theta_max : float           ← Максимальный угол 2θ рефлексов для включения в массив.

    Возвращает
    -------
    Result : list of list        ← Каждый элемент:
                                 [h, k, l, multiplicity, KPhase, two_theta, 'shift', 'FWHM', F2_mult, 0, 0, 0]
    """
    Result = []
    for grp in groups:
      if not grp:
        continue  # защита от пустых групп
      hkl_label = grp[0]['hkl_label']                                ## Берём канонический hkl и multiplicity 
      multiplicity = grp[0]['multiplicity']                          ## (они одинаковы внутри группы)
      two_theta_mean = float(np.mean([e['two_theta'] for e in grp])) ## Средние значения d, two_theta, F2 по группе
      F2_mean        = float(np.mean([e['F2'] for e in grp]))
      F2_mult = F2_mean * multiplicity  # Учитываем multiplicity для интенсивности
      Result.append([*hkl_label,        # h, k, l
                     multiplicity,      # multiplicity
                     KPhase,            # номер фазы
                     two_theta_mean,    # среднее 2θ
                     'shift',           # заглушка
                     'FWHM',            # заглушка
                     F2_mult,           # I = multiplicity * F2_mean
                     0, 0, 0])          # дополнительные заглушки
    Result.sort(key=lambda x: x[5])     # Сортировка по возрастанию two_theta
    if two_theta_max!=None:
      Result = [r for r in Result if r[5]<=two_theta_max]  # Обрезка по максимальному углу
    return Result
