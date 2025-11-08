from lmfit import Parameter
from typing import Literal
from utils.cif_extract import keyword_value
from models_peak.models_info import par_form_dict
## ========= Набор параметров ===========

## ----- Глобальные параметры фазы ------
def create_param_global(prefix_KPhase: str) -> dict[str, Parameter]:
  """
  Создаёт глобальные параметры фазы:
  - scale, shift — масштаб и смещение
  - Biso_overall — изотропный температурный фактор
  - phvol — объемная доля фазы

  Parameters:
  - prefix_KPhase: str — префикс фазы, например 'Phase1_'

  Returns:
  - dict[str, Parameter] — словарь параметров
  """
  prefix_KPhase = prefix_KPhase
  objects = {}
  object_name = prefix_KPhase + 'scale'
  objects[object_name] = Parameter(name=object_name, value=1, min=0, vary=False)

  object_name = prefix_KPhase + 'Biso_overall'
  objects[object_name] = Parameter(name=object_name, value=0.1, min=0, vary=False)

  object_name = prefix_KPhase + 'shift'
  objects[object_name] = Parameter(name=object_name, value=0, vary=False)

  object_name = prefix_KPhase + 'phvol'  ## fractional volume                     (Объемное содержание дополнительной фазы по отношению к основной)
  objects[object_name] = Parameter(name=object_name, value=1, min=0, vary=False) if prefix_KPhase.replace('Phase','').replace('_','')=='1' else Parameter(name=object_name, value=0.5, min=0, vary=False)
  return objects



## ----- Интенсивности пиков (уточняются в методе Ле Бейля) ------
def create_par_intensity(data_hkl: list[list[float]], prefix_KPhase: str) -> dict[str, Parameter]:                              # data_hkl - формат Bragg Positions
  """
  Создаёт параметры интенсивностей I_hkl для метода Ле Бейля.

  Parameters:
  - data_hkl: list[tuple[int, int, int, ...]] — список индексов (h,k,l) и др.
  - prefix_KPhase: str — префикс фазы (например, 'Phase1_')

  Returns:
  - dict[str, Parameter] — словарь параметров интенсивности
  """
  prefix_KPhase = prefix_KPhase
  data_hkl      = data_hkl
  objects = {}
  for line in data_hkl:
    object_name = prefix_KPhase + 'I_'+str(line[0])+'_'+str(line[1])+'_'+str(line[2])
    objects[object_name] = Parameter(object_name)

    objects.get(object_name).value = float(line[8])
    objects.get(object_name).min = 0
    objects.get(object_name)._vary = False
  return objects



## ----- Параметры решетки (a,b,c, alpha,beta,gamma) ------
def create_par_cell(data: list[str], prefix_KPhase: str) -> dict:                                       # data - массив строк из файла cif,    prefix_KPhase = 'Phase1_', например
  """
  Создаёт словарь параметров ячейки на основе CIF-данных.

  Parameters:
  - data: list[str] — строки CIF-файла
  - prefix_KPhase: str — префикс имени фазы, например 'Phase1_'

  Returns:
  - dict[str, Parameter] — параметры a, b, c, alpha, beta, gamma
  """
  data          = data
  prefix_KPhase = prefix_KPhase
  objects = {}
  i=0
  for keyword in ('_cell_length_a','_cell_length_b','_cell_length_c','_cell_angle_alpha','_cell_angle_beta','_cell_angle_gamma'):
    object_name = prefix_KPhase + keyword.split('_')[-1]
    objects[object_name] = Parameter(object_name)

    val = float(keyword_value(data,keyword))
    objects.get(object_name).value = val
    objects.get(object_name)._vary = False
    objects.get(object_name).min   = 0
    # ----- Restrictions (за счет симметрии) ----
    # if i<3:     #(для группы a,b,c)
    #  objects.get(object_name).expr = prefix_KPhase+'a' if (keyword.split('_')[-1]!='a') and (keyword.split('_')[-1] in ['b','c']) else None
    # if i>=3:    #(для группы aplha,beta,gamma)
    #  objects.get(object_name).expr = prefix_KPhase+'alpha' if (keyword.split('_')[-1]!='alpha') and (keyword.split('_')[-1] in ['beta','gamma']) else None
    i=i+1
  return objects


## ----- Параметры формы пиков ------- 
# Определение типа модели профиля:
FORM = Literal['Gaussian', 'Lorentzian', 'SplitLorentzian',
               'Voigt', 'PseudoVoigt', 'Moffat',
               'Pearson4','Pearson7', 'StudentsT','BreitWigner',
               'Lognormal','DampedOscillator', 'DampedHarmonicOscillator',
               'ExponentialGaussian','SkewedGaussian', 'SkewedVoigt']
def create_par_profile(prefix_KPhase: str, form: FORM, ampl: bool = False, center: bool = False) -> dict[str, Parameter]:
  """
  Генерация словаря с параметрами lmfit.Parameter для выбранной формы пиков.

  Параметры:
      prefix_KPhase (str): Префикс фазы (например, 'Phase1_', 'Phase2_')
      form (FORM): Название функции формы пика (из списка поддерживаемых)
      ampl (bool): Включить амплитуду A (по умолчанию False)
      center (bool): Включить центр μ (по умолчанию False)
  Возвращает:
      dict[str, Parameter]: Словарь с параметрами lmfit.Parameter, ключ — имя параметра
  """
  prefix_KPhase = prefix_KPhase
  model_name = form                            # Имя функции
  par_list = par_form_dict[model_name]         # Список параметров-словарей
  objects = {}                                 # Словарь с объектами Parameter
  for par in par_list:
    name,value,min,max = par['name'],par['value'],par['min'],par['max']
    object_name = prefix_KPhase + form + '_' + name          # Имя переменной с необходимыми префиксами
    objects[object_name] = Parameter(object_name)

    objects.get(object_name).value = value
    objects.get(object_name).min   = min
    objects.get(object_name).max   = max
    objects.get(object_name)._vary = False
  if ampl==False: del objects[prefix_KPhase + form + '_A']   # Удаляем амплитуду из параметров, если ampl=False
  if ampl==False: del objects[prefix_KPhase + form + '_μ']   # Удаляем положение центра из параметров, если center=False
  return objects


## ----- Поправки к положеням пиков ------- 
def hkl_to_str(hkl):
    return '_'.join(f"m{-i}" if i<0 else str(i) for i in hkl)

def create_par_delta(data_hkl: list[list[float]], prefix_KPhase: str) -> dict[str, Parameter]:                              # data_hkl - формат Bragg Positions
  """
  Создаёт параметры сдвига положений пиков δ_hkl для калибровки шкалы.

  Parameters:
  - data_hkl: list[tuple[int, int, int, ...]] — список индексов (h,k,l) и др.
  - prefix_KPhase: str — префикс фазы, например 'Phase1_'

  Returns:
  - dict[str, Parameter] — параметры shift_hkl. (!)
  """
  prefix_KPhase = prefix_KPhase
  data_hkl      = data_hkl
  objects = {}
  for line in data_hkl:
    object_name = prefix_KPhase + 'I_' + hkl_to_str(line[:3])
    objects[object_name] = Parameter(object_name)

    objects.get(object_name).value = 0
    objects.get(object_name)._vary = False
  return objects