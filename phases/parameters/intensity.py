from lmfit import Parameter

## Набор параметров: интенсивности пиков (уточняются в методе Ле Бейля)
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