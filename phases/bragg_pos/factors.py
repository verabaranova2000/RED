#@title ✪ (iv) Расчет $F^2$ для набора (h,k,l)

####   Извлечь параметры из фазы для calc_structural_factors
def extract_pars_for_F2(phase_object):
  a, b, c, alpha, beta, gamma  = [v.value for k,v in phase_object.param_cell.items()]
  wavelength  = phase_object.wavelength
  ops_sym     = phase_object.symmetry_operations
  KPhase      = phase_object.KPhase
  XYZ=[]                                                                    # Предварительно формируем массив XYZ - координаты неразмноженных атомов. Получаем [[0.0, 0.0, 0.0, 'Ca'], [0.25, 0.25, 0.25, 'F']]
  for atom in phase_object.atoms:
    name=re.sub("[^A-Za-z]", "", atom.name)                                      # Убираем число или степень окисления, если они есть, напр., Ca2+ (Удаляем все символы из строки, кроме букв)
    x,y,z=[v.value for k,v in atom.__dict__.get('param_positions').items()][:3]
    XYZ.append([x,y,z,name])
  return XYZ, a, b, c, alpha, beta, gamma, wavelength, ops_sym, KPhase


def calc_structural_factors(hkl_list, XYZ, a, b, c, alpha, beta, gamma, wavelength,
                            Operations_symmetry, KPhase, two_theta_max=5.5, Biso_overall=0):
    """
    Рассчитывает структурные факторы для списка hkl.

    Параметры
    ----------
    hkl_list : list of [h,k,l]          Список рефлексов
    XYZ : list of lists                 [x,y,z, ..., name] для каждой позиции атома.
    a,b,c,alpha,beta,gamma : float      Параметры ячейки.
    wavelength : float                  Длина волны.
    Operations_symmetry : list          Симметричные операции.
    KPhase : int                        Идентификатор фазы.
    Biso_overall : float                Значение Biso для всех атомов (по умолчанию 0).

    Возвращает
    -------
    dataBp : list of lists
        Каждая строка: [h,k,l, multiplicity, KPhase, two_theta, shift, FWHM, F_hkl, 0,0,0]
    """
    # Размножаем атомные позиции по симметриям
    XYZoccBiso_all=[]
    for line in XYZ:
        name = line[-1]
        x,y,z = line[:3]
        positions = get_all_positions_in_cell_for_atom(x,y,z,Operations_symmetry)
        # пока Biso = 1, fe = '?'
        all_info_for_atom = [list(pos)+[1,1,'?',name] for pos in positions]
        XYZoccBiso_all.extend(all_info_for_atom)

    F2_list=[]
    for i in range(len(hkl_list)):
      h,k,l     = hkl_list[i]
      #h,k,l     = hkl_list[i]['hkl']
      stl=1/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)) if not h==k==l==0 else 0
      two_theta=2*math.asin(wavelength/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)))*180/math.pi     if not h==k==l==0 else 0
      #if two_theta<=two_theta_max:                                             # рассчитываем структурные амплитуды и заносим их в ячейку line[5]
      for j in range(len(XYZoccBiso_all)):
        XYZoccBiso_all[j][5]=fe(stl,re.sub("[^A-Za-z]", "", XYZoccBiso_all[j][-1]))
      FF_hkl = FF(h,k,l,  a,b,c,alpha,beta,gamma, XYZoccBiso_all, Biso_overall=0)      # Квадрат структурной амплитуды
      F2_list.append(FF_hkl)
    return F2_list


#@title ✪ (v) Функция **get_vals_for_hkl**
import math
import numpy as np

def get_vals_for_hkl(hkl_star, phase_object, verbose=True, unique_only=False):
    """
    Для звезды hkl считает d, 2θ и печатает таблицу.

    Параметры
    ----------
    hkl_star : list of dict        Результат функции get_star_hkl (с ключами 'hkl', 'phase', 'R', 't').
    phase_object :                 объект класса Phase
    verbose : bool                 Печатать ли таблицу.
    unique_only : bool             True — печатать hkl без дубликатов, отсортированные по возрастанию 2θ.
                                   Если False — печатать все hkl, включая дубликаты.

    Возвращает
    -------
    Список массивов с dict - элемент списка соответствует определенному d
    list of dict        { 'hkl': np.array([h,k,l]),
                          'd': float,
                          'two_theta': float,
                          'phase': complex,
                          'F2' - добавляется позже как результат функции calc_structural_factors}
    """
    wavelength = phase_object.wavelength
    cell_array = [v.value for k, v in phase_object.param_cell.items()]
    pars_for_F2 = extract_pars_for_F2(phase_object)                             ## Извлекаем XYZ, a, b, c, alpha, beta, gamma, wavelength, ops_sym, KPhase

    rows = []                                                                   ## посчитаем все значения
    for entry in hkl_star:
        h, k, l = entry['hkl']
        d = d_hkl(h, k, l, *cell_array)
        theta_rad = math.asin(wavelength / (2 * d))
        two_theta = 2 * theta_rad * 180 / math.pi
        rows.append({'hkl': np.array([h, k, l]),
                     'd': d,
                     'two_theta': two_theta,
                     'phase': entry['phase'],
                     'op_index': entry['op_index']})

    rows_sorted   = sorted(rows, key=lambda r: r['two_theta'])                  ## сортировка по 2θ
    unique_sorted = sorted(get_unique_hkl(rows), key=lambda r: r['two_theta'])  ## Берем уникальные

    multiplicity = len(unique_sorted)

    if verbose:
      if unique_only: print_hkl_table(unique_sorted, hkl_label=None, multiplicity=multiplicity, unique_only=unique_only)
      else:           print_hkl_table(rows_sorted, hkl_label=None, multiplicity=multiplicity, unique_only=unique_only)

    ###### Разбиение на группы по d
    groups = group_by_d(unique_sorted) if unique_only else group_by_d(rows_sorted)
    for i, grp in enumerate(groups):
      # Выбор представителя группы (например, первый уникальный hkl в группе)
      grp_unique   = get_unique_hkl(grp) if not unique_only else grp
      hkl_label    = canonical_hkl(grp_unique)  if grp_unique else None
      multiplicity = len(grp_unique)
      hkl_list=[grp[i1]['hkl'] for i1 in range(len(grp))]
      F2_list      = calc_structural_factors(hkl_list, *pars_for_F2)
      for i2, grp_i in enumerate(grp):
          grp_i['hkl_label']    = hkl_label
          grp_i['multiplicity'] = multiplicity
          grp_i['F2']           = F2_list[i2]
      if verbose:                                    # Печатаем таблицы для каждой группы
        print(f"\n--- Группа {i+1}, d ≈ {grp[0]['d']:.5f} Å ---")
        print_hkl_table(grp, hkl_label, multiplicity, unique_only)
    return groups