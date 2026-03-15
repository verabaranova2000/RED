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

def parse_marker(par):
    """
    Разбирает маркер refonly на префикс и тип.

    Возвращает:
        prefix : str | None    # префикс фазы или None
        kind   : str           # тип маркера
    """
    # глобальные маркеры (все фазы, все параметры ∃-е в pars)
    if par in ["I_hkl", "delta_hkl", "s_all", "bckg_all"]:
        return None, par
    # фазовые маркеры с сегментом (одна фаза, все параметры внутри сегмента: если они !∃ в pars, добавляем в pars)
    for suffix in ["_I_inside", "_delta_inside"]:
        if par.endswith(suffix):
            prefix = par[:-len(suffix)]
            return prefix, suffix
    # фазовые маркеры без сегмента (одна фаза, все параметры ∃-е в pars)
    for suffix in ["_I_hkl", "_delta_hkl", "_profile"]:
        if par.endswith(suffix):
            prefix = par[:-len(suffix)]
            return prefix, suffix
    # обычный параметр
    return None, par


def resolve_refonly(refonly, pars_new, project_object, out, segment=None):
    """
    Преобразует список refonly в полный набор параметров, которые нужно открыть.
    """
    resolved = set()
    for par in refonly:
        prefix, kind = parse_marker(par)
        # --- фоновые параметры (∃-ие в pars) ---
        if kind == "s_all":
            resolved.update(p for p in pars_new if p.startswith("s") and is_background_param(p))
        elif kind == "bckg_all":
            resolved.update(p for p in pars_new if p.startswith("bckg") and is_background_param(p))

        # --- все интенсивности (∃-ие в pars) ---
        elif kind == "I_hkl":
            resolved.update(p for p in pars_new if "_I_" in p)

        # --- все дельты (∃-ие в pars) ---
        elif kind == "delta_hkl":
            resolved.update(p for p in pars_new if "_delta_" in p)

        # --- все интенсивности фазы (∃-ие в pars) ---
        elif kind == "_I_hkl":
            resolved.update(p for p in pars_new if prefix+"_I_" in p)

        # --- все дельты фазы (∃-ие в pars) ---
        elif kind == "_delta_hkl":
            resolved.update(p for p in pars_new if prefix+"_delta_" in p)

        # --- все интенсивности фазы внутри сегмента ---
        elif kind == "_I_inside":
            segm = out.userkws['axes'] if segment is None else segment
            phase = project_object.__dict__[prefix]
            for line in phase.bragg_positions:
                h,k,l = line[:3]
                if check_hkl_in_segment(segm, phase, h, k, l):
                    name = f"{phase.prefix}I_{hkl_to_str([int(h), int(k), int(l)])}"
                    if name not in pars_new:
                        pars_new.add(phase.param_intensity[name])
                    resolved.add(name)
                    if phase.setting['typeref'] != 'le Beil':
                        if [h,k,l] not in phase.setting['corrections']:
                            phase.setting['corrections'].append([h,k,l])
       
        # ---- все дельты фазы внутри сегмента ----
        elif kind == "_delta_inside":
            segm = out.userkws['axes'] if segment is None else segment
            phase = project_object.__dict__[prefix]
            for line in phase.bragg_positions:
                h,k,l = line[:3]
                if check_hkl_in_segment(segm, phase, h, k, l):
                    name = f"{phase.prefix}delta_{hkl_to_str([int(h), int(k), int(l)])}"
                    if name not in pars_new:
                        pars_new.add(phase.param_delta[name])
                    resolved.add(name)
                    if [h,k,l] not in phase.setting['calibrate']:
                        phase.setting['calibrate'].append([h,k,l])

        # --- параметры формы пиков фазы ---
        elif kind == "_profile":
            phase = project_object.__dict__[prefix]
            pars_form = [k for k in pars_new
                         if (len(set(model_list).intersection(k.split('_'))) > 0) and prefix in k]
            resolved.update(pars_form)
  
        # --- обычный параметр ---
        else:
          if par not in pars_new:
            raise ValueError(f"Неизвестный параметр '{par}'")
          resolved.add(par)
    return [p for p in pars_new if p in resolved]


def apply_refonly(pars, pars_new):
    """
    Фиксирует vary=True для всех параметров, которые нужно уточнять,
    кроме тех, у которых expr.
    """
    for p in pars:
        if pars_new[p].expr is None:
            pars_new[p].vary = True


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

    Обработка аргумента ``refonly`` выполняется в два этапа:

    refonly (маркеры и/или реальные параметры)
    ↓
    resolve_refonly
    ├─ раскрывает маркеры (I_hkl, Phase_I_inside, s_all и т.д.)
    ├─ добавляет отсутствующие параметры в pars_new
    ├─ обновляет настройки фаз (phase.setting)
    └─ формирует список реальных параметров → resolved
    ↓
    apply_refonly
    └─ открывает параметры для уточнения (vary=True)
    
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
        Список параметров и/или маркеров, которые
        должны быть открыты для уточнения.

    segment : tuple or None, optional
        Диапазон уточнения (например, по 2θ), используемый
        для выбора параметров внутри сегмента.

    Returns
    -------
    pars_new : lmfit.Parameters
        Новый набор параметров, подготовленный для следующего
        шага уточнения.

    resolved : list of str
        Список реальных параметров, полученных после раскрытия
        маркеров refonly. Используется для логирования и
        отображения параметров шага уточнения.
    """
    out = model_result
    pars_new = deepcopy_params(out.params)                                            # создаем глубокую копию, чтобы не изменялось состояние объекта

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

    # --- 5. Обрабатываем refonly (добавляет параметры, меняет setting); возвр. реальные параметры
    if refonly:
        resolved = resolve_refonly(refonly, pars_new, project_object, out, segment)
        apply_refonly(resolved, pars_new)

    return pars_new, resolved




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

INTENSITY_PARAM_PATTERN = re.compile(r".+_I_\d+_\d+_\d+$")
def is_intensity_param(name: str) -> bool:
    """
    Проверить, является ли параметр параметром интенсивности.
    Шаблон: 
      Phase1_I_1_1_1
      Phase1_I_2_0_0
    """
    return "_I_" in name



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
