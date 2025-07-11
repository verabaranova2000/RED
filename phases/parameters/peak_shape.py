# Генерация словаря с параметрами
from typing import Literal
from typing import Literal, Dict
from lmfit import Parameter
from models_peak.models_info import par_form_dict

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
    object_name = prefix_KPhase + form + '_' + name                             # Имя переменной с необходимыми префиксами
    objects[object_name] = Parameter(object_name)

    objects.get(object_name).value = value
    objects.get(object_name).min   = min
    objects.get(object_name).max   = max
    objects.get(object_name)._vary = False

  if ampl==False: del objects[prefix_KPhase + form + '_A']                      # Удаляем амплитуду из параметров, если ampl=False
  if ampl==False: del objects[prefix_KPhase + form + '_μ']                      # Удаляем положение центра из параметров, если center=False
  return objects