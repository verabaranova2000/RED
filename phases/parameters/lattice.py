from lmfit import Parameter
from utils.cif_extract import keyword_value


# Набор параметров: параметры решетки (a,b,c, alpha,beta,gamma) - *атрибут класса Phase*
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

    ### Restrictions (за счет симметрии)
    if i<3:     #(для группы a,b,c)
      objects.get(object_name).expr = prefix_KPhase+'a' if (keyword.split('_')[-1]!='a') and (keyword.split('_')[-1] in ['b','c']) else None
    if i>=3:    #(для группы aplha,beta,gamma)
      objects.get(object_name).expr = prefix_KPhase+'alpha' if (keyword.split('_')[-1]!='alpha') and (keyword.split('_')[-1] in ['beta','gamma']) else None
    i=i+1
  return objects