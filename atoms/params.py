from lmfit import Parameter
import re, math
from utils.cif_extract import get_value_for_atom
## ========= Набор параметров ===========


## ===== Координаты атома (x,y,z) заселенность occ ===
def create_par_positions(data, prefix_KPhase, atom_name, positions=None):       #Например, для атома Ca1, принадлежащего 1-й фазе:   prefix_KPhase='Phase1_', atom_name='Ca1'
  """
  Создание параметров координат атома (x, y, z) и заселённости (occ).
  Если задан 'data' — координаты извлекаются из CIF.
  Если задан 'positions' — берутся из списка или кортежа [x, y, z].
  """
  objects = {}
  if data is not None:
    for keyword in ('atom_site_fract_x','atom_site_fract_y','atom_site_fract_z'):
      object_name = f"{prefix_KPhase}{atom_name}_{keyword[-1]}"
      objects[object_name] = Parameter(object_name)
      objects.get(object_name).value = float(get_value_for_atom(data,keyword,atom_name))
      objects.get(object_name)._vary = False
  # ---- Если заданы координаты вручную ----
  elif data==None and positions is not None:
    for key, val in zip(('x','y','z'), positions):
      object_name = f"{prefix_KPhase}{atom_name}_{key}"
      objects[object_name] = Parameter(object_name)
      objects.get(object_name).value = val
      objects.get(object_name)._vary = False
  else:
      raise ValueError("Нужно указать либо 'data', либо 'positions'.")
  object_name_occ = f"{prefix_KPhase}{atom_name}_occ"
  objects[object_name_occ]=Parameter(object_name_occ)
  objects[object_name_occ].value=1
  objects[object_name_occ]._vary=False
  objects[object_name_occ].min, objects[object_name_occ].max = 0, 1
  return objects


## ===== Тепловые параметры атома (Biso и т.д.) =====                       Например, для атома Ca1, принадлежащего 1-й фазе:   prefix_KPhase='Phase1_', atom_name='Ca1'
def create_par_ADP(prefix_KPhase, atom_name, ADP_type):
  """
  Создаёт набор параметров атомных тепловых колебаний (ADP) заданного типа (Biso, B, C, D, E, F) 
  для выбранного атома и фазы.
  """
  prefix_KPhase = prefix_KPhase
  atom_name     = atom_name
  ADP_type      = ADP_type
  dict_order_index={'Biso': [''],
                    'B': [str(x) for x in [11,22,33,12,13,23]],
                    'C': [str(x) for x in [111,112,113,122,123,133, 222,223,233,333]],
                    'D': [str(x) for x in [1111,1112,1113,1122,1123,1133, 1222,1223,1233,1333,2222,2223, 2233,2333,3333]],
                    'E': [str(x) for x in [11111,11112,11113,11122,11123,11133, 11222,11223,11233,11333,12222,12223, 12233,12333,13333,22222,22223,22233, 22333,23333,33333]],
                    'F': [str(x) for x in [111111,111112,111113,111122,111123,111133, 111222,111223,111233,111333,112222,112223, 112233,112333,113333,122222,122223,122233, 122333,123333,133333,222222,222223,222233,222333,223333,233333,333333]] }
  indexes = dict_order_index.get(ADP_type)
  objects = {}
  for ind in indexes:
    object_name          = f"{prefix_KPhase}{atom_name}_{ADP_type}{ind}"
    objects[object_name] = Parameter(object_name)
    objects.get(object_name).value = 0    if ADP_type!='Biso' else 0.01        # Все тепловые параметры по умолчанию = 0, кроме Biso
    objects.get(object_name)._vary = False                                     # Все тепловые параметры по умолчанию фиксированы
  return objects                                                               # Создали словарь с объектами типа Parameter. Здесь ключ - имя параметра, значение - сам объект Parameter


### =====  Каппа-модель для атома (P и kappa) без начальных значений из kappa ===
def create_par_kmodel(atom_info, prefix_KPhase, atom_name, ashperical=False, **pars):                     #Например, для атома Ca1, принадлежащего 1-й фазе:   prefix_KPhase='Phase1_', atom_name='Ca1'
  names_curves_valence = [k for k,v in atom_info['curves'].items() if k not in ['neutral atom', 'core']]
  dict_P_max={'s': 2,
              'p': 6, 'p_sph': 2, 'p_asph': 4,
              'd': 10,
              'f': 14}
  objects = {}
  for shell in names_curves_valence:
    for par_name in ['P', 'kappa']:
      object_name = f"{prefix_KPhase}{atom_name}_{shell}_{par_name}"
      objects[object_name] = Parameter(object_name)
      if par_name=='kappa':
        objects[object_name].value=1
        objects[object_name].min=0
      if par_name=='P':
        objects[object_name].value = sum([float(v['P']) for k,v in atom_info['valence'].items() if k==shell or k==shell+'-'])
        objects[object_name].min   = 0
        objects[object_name].max   = dict_P_max[re.sub("[^A-Za-z]", "", shell)]
      objects[object_name]._vary = False
  ## -----------------
  if ashperical==True:
    aspherical_shell=atom_info['parametrisations']['aspherical shell']
    full_prefix_shell=f"{prefix_KPhase}{atom_name}_{aspherical_shell}_"
    if re.sub("[^A-Za-z]", "", aspherical_shell)=='p':
      for part in ['sph', 'asph']:
        object_name = full_prefix_shell + part+'_P'
        objects[object_name] = Parameter(object_name)
        objects[object_name].min   = 0
        if part=='sph':
          objects[object_name].value = dict_P_max[re.sub("[^A-Za-z]", "", aspherical_shell)+'_'+part] if pars[full_prefix_shell+'P'].value>=dict_P_max[re.sub("[^A-Za-z]", "", aspherical_shell)+'_'+part] else pars[full_prefix_shell+'P'].value
        elif part=='asph':
          objects[object_name].value = 0 if pars[full_prefix_shell+'P'].value<=dict_P_max[re.sub("[^A-Za-z]", "", aspherical_shell)+'_sph'] else pars[full_prefix_shell+'P'].value - dict_P_max[re.sub("[^A-Za-z]", "", aspherical_shell)+'_sph']
        objects[object_name].max   = dict_P_max[re.sub("[^A-Za-z]", "", aspherical_shell)+'_'+part]
        objects[object_name].vary  = False
      del objects[full_prefix_shell+'P']                                        ## Удаляем старый параметр общей заселенности
    ## --- Добавляем параметр: ориентацию асферической оболочки ---
    par_angle                = full_prefix_shell + 'angle'
    objects[par_angle]       = Parameter(par_angle)
    objects[par_angle].value = math.atan(2**0.5)*180/math.pi
    objects[par_angle].vary  = False
    objects[par_angle].min, objects[par_angle].max = 0, 180
  return objects