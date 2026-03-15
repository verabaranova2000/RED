from __future__ import annotations       # делает все аннотации ленивыми (строками)
from typing import Optional, List, TYPE_CHECKING
#if TYPE_CHECKING:
#    # импорт только для type checkers, не выполняется во время рантайма
#    from refinement.project import Project

import re
from lmfit import Parameters, Parameter 
from phases.params import hkl_to_str
from phases.models import model_list
from refinement.segment import check_hkl_in_segment



# ==== Объединение параметров (параметры всех фаз в проекте и фона) с учетом setting ====
def prepare_params(project_object=None, phase_object=None, profile=True, background=True):
    """
    Формирование общего набора параметров lmfit.Parameters для уточнения.

    Функция объединяет параметры фаз, атомов, интенсивностей и фона
    в единый объект Parameters с учётом настроек фаз (setting).

    Возможны два режима работы:
    - подготовка параметров для всего проекта
    - подготовка параметров только для одной фазы

    Parameters
    ----------
    project_object : Project, optional
        Объект проекта, содержащий все фазы и параметры профиля.
        Используется для формирования набора параметров всех фаз.

    phase_object : Phase, optional
        Объект одной фазы. Если указан, формируется набор параметров
        только для данной фазы.

    profile : bool, optional
        Если True, в набор параметров включаются параметры формы
        профиля (profile parameters).

    background : bool, optional
        Если True, добавляются параметры фона из объекта проекта.

    Returns
    -------
    lmfit.Parameters
        Объект Parameters, содержащий объединённый набор параметров
        для уточнения.
    
    Notes
    -----
    При наличии нескольких фаз автоматически накладываются связи
    между параметрами scale и shift разных фаз.
    """
    if project_object is not None and phase_object is not None:
        raise ValueError("Необходимо указать либо project_object, либо phase_object, но не оба одновременно.")

    dict_all_pars = {}                       # словарь, куда будем заносить параметры всех фаз
    set_of_Phases = []                       # cписок, куда поместим все фазы
    if project_object is not None: 
      set_of_Phases=[project_object.__dict__.get('Phase'+str(KPhase)) for KPhase in range(1, project_object.NPhases+1)]
    elif phase_object is not None: 
      set_of_Phases = [phase_object]
    
    # --- 1. Цикл по фазам из набора
    for Phasei in set_of_Phases:                       
      typeref          = Phasei.setting['typeref']
      corrections      = Phasei.setting['corrections']
      calibrate        = Phasei.setting['calibrate']
      calibration_mode = Phasei.setting['calibration mode']

      # --- a-c. Объединение словарей с парметрами в один словарь ---
      # --- a. (Rietveld) ---
      if typeref=='Rietveld':
        for pars_block_ph in [Phasei.param_cell, Phasei.param_scale_shift_phvol_Biso_overall, Phasei.param_profile]:                        # Добавляем параметры текущей фазы Phasei
          if profile:                                          
            dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))
          elif not profile and pars_block_ph!=Phasei.param_profile: 
            dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))
        
        for atom in Phasei.atoms:                                                                                                            # Добавляем параметры атомов в текущей фазе Phasei:
          dict_all_pars=dict(list(dict_all_pars.items()) + list(atom.params.items()))
        if len(corrections)!=0:                                                                                                              # Добавляем интенсивности, которые нужно уточнить
          for h_,k_,l_ in corrections:
            I_name=Phasei.prefix + 'I_' + hkl_to_str([h_,k_,l_])
            dict_all_pars[I_name]=Phasei.param_intensity[I_name]

      # --- b. (le Beil) ---
      if typeref=='le Beil':
        for pars_block_ph in [Phasei.param_cell, Phasei.param_scale_shift_phvol_Biso_overall, Phasei.param_profile, Phasei.param_intensity]:  # Добавляем параметры текущей фазы Phasei
          if profile:                                          
            dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))
          elif not profile and pars_block_ph!=Phasei.param_profile: 
            dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))

      # --- c. (calibrate) ---
      if   calibration_mode and (calibrate=='all'): 
        dict_all_pars=dict(list(dict_all_pars.items()) + list(Phasei.param_delta.items()))
      elif calibration_mode and (calibrate!=[]):
        delta_for_ref = [f"{Phasei.prefix}delta_{hkl_to_str(hkl)}" for hkl in Phasei.setting['calibrate']]
        dict_all_pars.update({k: Phasei.param_delta[k] for k in delta_for_ref if k in Phasei.param_delta})

    # --- 2. Создание объекта типа Parameters ---
    params_ph = Parameters()
    for k,v in dict_all_pars.items():
      if profile:                            
        params_ph.add(v)
      elif not profile and k.split('_')[-1] not in ['phvol','shift']: 
        params_ph.add(v)

    # --- 3. Добавление параметров фона, если ключевое слово background=True ---
    if (project_object is not None) and background and profile:
      for k_bckg, v_bckg in project_object.Profile_points.params.items():
        params_ph.add(v_bckg)

    # --- 4. Наложение условий на параметры, если количество фаз > 1 ---
    if profile and (project_object is not None) and (project_object.NPhases>1):
      for KPhase in range(2,project_object.NPhases+1):
        params_ph['Phase'+str(KPhase)+'_scale'].expr  ='Phase1_scale'
        params_ph['Phase'+str(KPhase)+'_shift'].expr  ='Phase1_shift'
        #params_ph['Phase'+str(KPhase)+'_phvol'].value = 0.5
        #params_ph['Phase'+str(KPhase)+'_phvol'].init_value = 0.5
  
    return params_ph



# ====== Глубокая копия набора параметров (объекта типа Parameters) ========
def deepcopy_params(params):
    """
    Создание глубокой копии объекта lmfit.Parameters.

    Функция формирует новый объект Parameters и копирует в него
    все параметры из исходного набора, включая их атрибуты
    (value, vary, min, max, expr и др.).

    Используется вместо стандартного deepcopy, поскольку
    объекты lmfit.Parameters и Parameter могут некорректно
    копироваться стандартными средствами Python.

    Parameters
    ----------
    params : lmfit.Parameters
        Исходный набор параметров.

    Returns
    -------
    lmfit.Parameters
        Новый объект Parameters, содержащий копии всех параметров.
    """
    params_new_dict={}
    for k,v in params.items():             # Пробегаемся по каждому параметру (здесь параметр - это значение в словаре)
      pari={}                              # Создадим словарь. В него будем копировать параметр v
      for k1,v1 in v.__dict__.items():     # Пробегаемся по парам ключ-значение, которые входят в параметр
        if k1!='from_internal':
          pari[k1]=v1               ##### Т.о., скопировали информацию о параметре в словарь
      par_new=Parameter(name=pari.get('name'))   # Это - копия параметра. Его нужно наполнить содержанием из словаря pari
      for k2,v2 in pari.items():
         par_new.__dict__[k2]=v2           # Заполним словарь значениями по ключам
      params_new_dict[k]=par_new           # Добавим полученный параметр в словарь параметров

    params_new=Parameters()
    for k,v in params_new_dict.items():
      params_new.add(v)
    return params_new


# ============================================================
# Подготовка набора параметров к новому уточнению 
# ============================================================

def params_for_next(project_object,     #: Project, 
                    model_result, 
                    canсel_lastref: Optional[List[str]] = None, 
                    undate_init_val='all', 
                    fix: bool = True, 
                    refonly: Optional[List[str]] = None, 
                    segment=None): 
    """
    Подготовка набора параметров для следующего шага уточнения.

    Функция формирует новый объект Parameters на основе результатов
    предыдущего уточнения и выполняет необходимые операции:

    - создание глубокой копии набора параметров
    - отмену результатов последнего уточнения для выбранных параметров
    - обновление начальных значений параметров
    - добавление параметров интенсивностей I_hkl при необходимости
    - фиксацию всех параметров
    - открытие выбранных параметров для уточнения

    Parameters
    ----------
    project : Project
        Объект проекта, содержащий информацию о фазах,
        брэгговских позициях и настройках уточнения.

    model_result : lmfit.model.ModelResult
        Результат предыдущего шага уточнения.

    cancel_lastref : list of str, optional
        Список параметров, для которых отменяется результат
        последнего уточнения (значение возвращается к init_value).

    update_init_val : {"all"} or list of str, optional
        Обновление начальных значений параметров (init_value) (заменить init_value на уточненные значения).
        Если указано "all", обновляются все параметры.

    fix : bool, optional
        Если True, все параметры фиксируются перед открытием
        выбранных параметров для уточнения.

    refonly : list of str, optional
        Список параметров или групп параметров, которые
        должны быть открыты для уточнения.

    segment : tuple or None, optional
        Диапазон уточнения (например, по 2θ), используемый
        для выбора параметров внутри сегмента.

    Returns
    -------
    lmfit.Parameters
        Новый набор параметров, подготовленный для следующего
        шага уточнения.
    """
    out=model_result
    pars_new=deepcopy_params(out.params)                                            # создаем глубокую копию, чтобы не изменялось состояние объекта

    # --- 1. Отмена результатов последнего уточнения для перечисленных в квадратных скобках параметров
    if canсel_lastref is not None and isinstance(canсel_lastref, list):             # чтобы отменить результаты последнего уточнения некоторых параметров и вернуться к начальным значениям
      for par in canсel_lastref:
        assert par in [k for k,v in pars_new.items()]                               # прерываем, если названия параметра нет в списке
        pars_new.get(par).value = out.init_params.get(par).value                    # открываем параметр для уточнения

    # --- 2. Обновляем начальные значения (init_value = value) либо для всех параметров, либо для перечисленных в квадратных скобках параметров. Теперь уточненные на предыдущем шаге величины будут отправными точками в следующем уточнении
    if undate_init_val=='all':                                                      # Обновляем начальные значения (init_value = value)
      for v,k in pars_new.items():
        k.init_value=k.value
    if undate_init_val!='all' and isinstance(undate_init_val, list):
      for par in undate_init_val:
        assert par in [k for k,v in pars_new.items()]
        pars_new.get(par).init_value=out.params.get(par).value

    # --- 3. Обновляем набор параметров I_hkl ---
    for KPhase in range(1, project_object.NPhases+1):             # пробегаемся по фазам в проекте
      Phasei      = project_object.__dict__.get('Phase'+str(KPhase))
      corrections = Phasei.setting['corrections']
      if len(corrections)!=0:
        for h,k,l in Phasei.setting['corrections']:
          I_name=f"{Phasei.prefix}I_{hkl_to_str([int(h), int(k), int(l)])}"
          if I_name not in pars_new: 
            pars_new.add(Phasei.param_intensity[I_name])


    # --- 4. Фиксируем все параметры ---
    if fix:                                                         
      for v,k in pars_new.items():
        k.vary=False

    # --- 5. Открываем для уточнения параметры, перечисленные в квадратных скобках ---
    if refonly is not None and isinstance(refonly, list):
      if 'I_hkl' in refonly:                                        # Если хотим уточнить все интенсивности в методе le Beil
        for par in [k for k,v in pars_new.items() if '_I_' in k]: 
          pars_new.get(par).vary=True                               # Открываем все интенсивности для уточнения

      if 'delta_hkl' in refonly:                                    # Если хотим уточнить сдвиги всех пиков
        for par in [k for k,v in pars_new.items() if '_delta_' in k]: 
          pars_new.get(par).vary=True                               # Открываем все сдвиги пиков для уточнения
 
#      if 's_all' in refonly:                                    # Если хотим уточнить все параметры фона
#        s_list = expand_background_params(['s_all'], pars_new)
#        for par in s_list:
#          pars_new.get(par).vary=True                          # Открываем все узлы сплайна для уточнения


      for par in refonly:                                           # Если подаем список параметров для уточнения
        assert (par in [k for k,v in pars_new.items()]+['I_hkl', 'delta_hkl', 's_all']) or ('_I_hkl' in par) or ('_I_inside' in par) or ('_profile' in par) # прерываем, если названия параметра нет в списке
        if par in [k for k,v in pars_new.items()] and (pars_new.get(par).expr is None):
          pars_new.get(par).vary=True                               # открываем параметр для уточнения

        # --- 5.1. Уточнение группы параметров для какой-то фазы ---
        if '_I_hkl' in par:                                         # Если хотим уточнить все интенсивности только одной фазы
          prefix_KPhase=par.split('_')[0]                           # Префикс этой фазы, интенсивности которой хотим уточнить
          for parI in [k for k,v in pars_new.items() if prefix_KPhase+'_I_' in k]: 
            pars_new.get(parI).vary=True

        if '_I_inside' in par:                                      # Если хотим уточнить все интенсивности только одной фазы
          prefix_KPhase=par.split('_')[0]                           # Префикс этой фазы, интенсивности которой хотим уточнить
          segm = out.userkws['axes'] if segment is None else segment
          my_phase = project_object.__dict__.get(prefix_KPhase.replace('_',''))
          for line in my_phase.bragg_positions:
            hi,ki,li=line[:3]
            if check_hkl_in_segment(segm, my_phase,h=hi,k=ki,l=li):
              I_name=f"{my_phase.prefix}I_{hkl_to_str([int(hi), int(ki), int(li)])}"        # если рефлекс лежит внутри уточняемого диапазона, он нам нужен
              if I_name not in pars_new: pars_new.add(my_phase.param_intensity[I_name])     # Добавляем интенсивность рефлекса в набор параметров для уточнения, если её там нет
              pars_new.get(I_name).vary=True                                                # Открываем её для уточнения
              if project_object.__dict__.get(prefix_KPhase.replace('_','')).setting['typeref']!='le Beil' and [hi,ki,li] not in project_object.__dict__.get(prefix_KPhase.replace('_','')).setting['corrections']: 
                project_object.__dict__.get(prefix_KPhase.replace('_','')).setting['corrections'].append([hi,ki,li])  # Добавляем индексы этих рефлексов в sitting для фазы

        # --- 5.2. Уточнение группы параметров для какой-то фазы ---
        if '_delta_hkl' in par:                                                   # Если хотим уточнить сдвиги всех пиков только одной фазы
          prefix_KPhase=par.split('_')[0]                                         # Префикс этой фазы, сдвиги пиков которой хотим уточнить
          for pardelta in [k for k,v in pars_new.items() if prefix_KPhase+'_delta_' in k]: 
            pars_new.get(pardelta).vary=True

        if '_delta_inside' in par:                                                # Если хотим уточнить сдвиги всех пиков только одной фазы
          prefix_KPhase=par.split('_')[0]                                         # Префикс этой фазы, сдвиги пиков которой хотим уточнить
          segm = out.userkws['axes'] if segment is None else segment
          my_phase = project_object.__dict__.get(prefix_KPhase.replace('_',''))
          for line in my_phase.bragg_positions:
            hi,ki,li=line[:3]
            if check_hkl_in_segment(segm, my_phase,h=hi,k=ki,l=li):
              delta_name=f"{my_phase.prefix}delta_{hkl_to_str([int(hi), int(ki), int(li)])}"                          # если рефлекс лежит внутри уточняемого диапазона, он нам нужен
              if delta_name not in pars_new: 
                 pars_new.add(my_phase.param_delta[delta_name])                           # Добавляем интенсивность рефлекса в набор параметров для уточнения, если её там нет
              pars_new.get(delta_name).vary=True                                                                      # Открываем её для уточнения
              if [hi,ki,li] not in project_object.__dict__.get(prefix_KPhase.replace('_','')).setting['calibrate']: 
                project_object.__dict__.get(prefix_KPhase.replace('_','')).setting['calibrate'].append([hi,ki,li])    # Добавляем индексы этих рефлексов в setting для фазы

        if '_profile' in par:                                                   # Если хотим уточнить все параметры формы пиков для какой-то фазы
          prefix_KPhase=par.split('_')[0]
          pars_form = [k for k,v in pars_new.items() if (len(set(model_list).intersection(k.split('_'))) > 0) and (prefix_KPhase in k)] ### Извлекли названия параметров формы
          for par_form in pars_form:
            if pars_new.get(par).expr is None: 
              pars_new.get(par_form).vary = True     


    ## Фиксация параметров, у которых есть expr
    #for k,v in pars_new.items():
    # if v.expr!=None: pars_new[k].vary=False
    return pars_new




# ============================================================
# Background parameter utilities
# ============================================================

BACKGROUND_PARAM_PATTERN = re.compile(r"^(bckg|s)\d+$")   # допустимые параметры фона:  bckg0, bckg1, ...; s0, s1, ...
# (bckg|s) - префикс
# \d+      - одно или больше чисел
# $	       - конец строки

def is_background_param(name: str) -> bool:
    """
    Проверить, является ли параметр параметром фона.

    Поддерживаемые форматы:
    - bckg0, bckg1, ...
    - s0, s1, ...
    """
    return bool(BACKGROUND_PARAM_PATTERN.match(name))

def parse_background_param(name: str):
    """
    Разобрать имя параметра фона.
    
    Превращает: 
        bckg12 → ("bckg", 12)
        s7     → ("s", 7)

    Возвращает
    ----------
    (prefix, index)
    """
    m = BACKGROUND_PARAM_PATTERN.match(name)
    if not m:
        return None, None
    prefix = m.group(1)
    idx = int(name[len(prefix):])
    return prefix, idx



# ============================================================
# Intensity parameter utilities
# ============================================================

# ==== Разворачивание маркеров типа "Phase1_I_inside" в реальные параметры ====
INTENSITY_PARAM_PATTERN = re.compile(r".+_I_\d+_\d+_\d+$")
def is_intensity_param(name: str) -> bool:
    """
    Проверить, является ли параметр параметром интенсивности.
    Шаблон: 
      Phase1_I_1_1_1
      Phase1_I_2_0_0
    """
    return "_I_" in name

def extract_intensity_params(pars, prefix=None):
    """
    Извлечение параметров из набора: какие параметры интенсивностей есть в наборе?
    Возвращает список параметров интенсивностей I_hkl из lmfit.Parameters.

    Parameters
    ----------
    pars : lmfit.Parameters
        Набор параметров модели.

    prefix : str, optional
        Префикс фазы (например 'Phase1_').
        Если указан — выбираются только интенсивности этой фазы.

    Returns
    -------
    list[str]
        Список имён параметров интенсивностей.
    """
    names = []
    for name in pars.keys():
        if is_intensity_param(name):
            if prefix is None or name.startswith(prefix):
                names.append(name)
    return names



# ============================================================
# YAML marker expanders
# ============================================================

BACKGROUND_ALL_PATTERN = re.compile(r"^(bckg|s)_all$")
def expand_background_params(params_list, pars):
    """
    Разворачивание YAML-маркеров фоновых параметров.

    Поддерживаемые маркеры:
        bckg_all → bckg0, bckg1, ...
        s_all    → s0, s1, ...

    Parameters
    ----------
    params_list : list[str]
        Список параметров из YAML.

    pars : lmfit.Parameters
        Набор параметров модели.

    Returns
    -------
    list[str]
        Развёрнутый список параметров.

    Baжно
    ------
    Если параметров s* / bckg* нет, то функция вернет [], т.е. маркер просто пропадет.
    """
    expanded = []
    for p in params_list:
        m = BACKGROUND_ALL_PATTERN.match(p)
        if m:                   # маркер разворачивается в список пар-в; список пар-ров добавляется в набор вместо маркера
            prefix = m.group(1)
            expanded.extend(name for name in pars.keys()
                            if name.startswith(prefix) and is_background_param(name))
        else:
            expanded.append(p)   # обысный параметр добавляется в набор
    return expanded


def expand_intensity_params(params_list, pars):
    """
    Разворачивание YAML-маркеров.
    Заменяет маркеры типа 'Phase1_I_inside' на реальные параметры интенсивностей.
    """
    expanded = []
    for p in params_list:
        if p.endswith("_I_inside"):
            prefix = p.replace("I_inside", "")
            expanded.extend(extract_intensity_params(pars, prefix))
        else:
            expanded.append(p)
    return expanded


PARAM_EXPANDERS = [expand_intensity_params,
                   expand_background_params]
def expand_param_markers(params_list, pars):
    """
    Развернуть все YAML-маркеры параметров.

    Поддерживаемые маркеры:
        PhaseX_I_inside
        bckg_all
        s_all

    Expansion pipeline
    ------------------
    expand_param_markers
        -> expand_intensity_params
        -> expand_background_params

    Parameters
    ----------
    params_list : list[str]
        Список параметров из YAML.

    pars : lmfit.Parameters
        Набор параметров модели.

    Returns
    -------
    list[str]
        Развёрнутый список параметров.
    """
    params = expand_intensity_params(params_list, pars)
    params = expand_background_params(params, pars)
    return params



# ============================================================
# Parameter grouping for reports
# ============================================================

def split_param_groups(param_data):
    """
    Разделить параметры на группы:
    - background
    - intensity
    - normal
    """
    background = {}
    intensity = {}
    normal = {}
    for p, val in param_data.items():
        if is_background_param(p):
            background[p] = val
        elif is_intensity_param(p):
            intensity[p] = val
        else:
            normal[p] = val
    return background, intensity, normal








# ====== Постобработка параметров  =====
#         - извлекает значение параметра
#         - считает изменение %

def val_delta_percent(pars, param_name):
    """
    Возвращает значение параметра и его относительное изменение (%).
    Parameters
    ----------
    pars : lmfit.Parameters
        Набор параметров.

    param_name: str
        Имя параметра.

    Returns
    -------
    tuple
        (value, delta_percent)
    """
    val = pars[param_name].value
    init_val = pars[param_name].init_value

    if init_val == 0:
        dperc = None
    else:
        dperc = (init_val - val) / init_val * 100
    return val, dperc

def format_value(v, fmt=".6f"):
    """ Утилита для value из val_delta_percent. """
    if v is None:
        return "—"
    return format(v, fmt)

def format_dperc(d, fmt=".3f"):
    """ Утилита для delta_percent из val_delta_percent. """
    if d is None:
        return "∞"
    return format(d, fmt)   #f"{d:.3f}"


# Возможно, не пригодится. Убрать
def relative_change(pars, param_name):
    """
    Формирует строку с текущим значением параметра и
    его относительным изменением (%).

    Parameters
    ----------
    pars : lmfit.Parameters
        Набор параметров.

    param_name : str
        Имя параметра.

    Returns
    -------
    str
        Строка вида: "param=value (Δ%)".
    """
    val, delta = val_delta_percent(pars, param_name)
    return f"{param_name}={round(val, 6):<12} ({delta:>7} %)"
