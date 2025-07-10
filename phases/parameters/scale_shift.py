from lmfit import Parameter

def create_param_scale_shift_phvol_Biso_overall(prefix_KPhase: str) -> dict[str, Parameter]:
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