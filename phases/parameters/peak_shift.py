from lmfit import Parameter

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
    object_name = prefix_KPhase + 'delta_'+str(line[0])+'_'+str(line[1])+'_'+str(line[2])
    objects[object_name] = Parameter(object_name)

    objects.get(object_name).value = 0
    objects.get(object_name)._vary = False

  return objects