import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import numpy as np



## ============ Чтение Coppens ============

## ====== Проверка/исправление разделительных строк ======
def block_format(data):
  """
  Приводит текстовый массив данных атома к стандартному формату.

  Обеспечивает корректное наличие и отсутствие пустых строк вокруг заголовков:
  'Core subshells', 'Valence subshells' и 'Populations', удаляя лишние пустые строки 
  и вставляя отсутствующие, чтобы каждый блок был однозначно разделён.

  Parameters
  ----------
  data : list of str   ← Список строк файла с параметрами атома.

  Returns
  -------
  list of str          ← Список строк с исправленным форматированием заголовков и блоков.
  """
  titles_idx = [i for i, line in enumerate(data) 
              if ("Core subshells" in line) 
              or ("Valence subshells" in line) 
              or ("Populations" in line)]
  ## Уберем лишние пропуски после заголовков "Core subshells", Valence subshells""
  while data[titles_idx[0]+1].strip() == '':
    del data[titles_idx[0]+1]
    titles_idx[1] -=1
    titles_idx[2] -=1
  while data[titles_idx[1]+1].strip() == '':
    del data[titles_idx[1]+1]
    titles_idx[2] -=1
  ## Проверка пропусков перед и после Valence subshells
  if data[titles_idx[1]-1].strip() != '':
    #print('Перед Valence subshells нет enter! Надо добавить!')
    data.insert(titles_idx[1], '\n')
    titles_idx[1] +=1
    titles_idx[2] +=1
  else: 
    i=1
    while data[titles_idx[1]-1-1].strip() == '':
      del data[titles_idx[1]-1-1]
      titles_idx[1] -=1
      titles_idx[2] -=1
  ## Проверка пропусков перед и после Populations
  if data[titles_idx[2]-1].strip() != '':
    #print('Перед Populations нет enter! Надо добавить!')
    data.insert(titles_idx[2], '\n')
    titles_idx[2] +=1
  else: 
    i=1
    while data[titles_idx[2]-1-1].strip() == '':
      del data[titles_idx[2]-1-1]
      titles_idx[2] -=1
  if data[titles_idx[2]+1].strip() != '':
    #print('После Populations нет enter! Надо добавить!')
    data.insert(titles_idx[2]+1, '\n')
  return data


## ====== Чтение информации об атоме ======
def read_scatfile(txt):                  
  """
  Считывает файл с параметрами атома, содержащий электронную структуру и амплитуды 
  рассеяния по оболочкам (ядро, валентная зона, нейтральный атом).

  Функция предназначена для работы с текстовыми файлами базы Коппенса, 
  расположенными в каталоге `atoms/scattering_factors/data`. 
  Возвращает структуру данных, включающую информацию о заселённостях 
  электронных оболочек и числовые кривые рассеяния.

  Parameters
  ----------
  txt : str       ← Путь к файлу или имя файла с данными атома (формат *.txt*).

  Returns
  -------
  dict            ← Словарь с ключами:
      - 'core'    — данные по остовным (core) оболочкам и их заселённостям;
      - 'valence' — данные по валентным оболочкам;
      - 'curves'  — набор кривых рассеяния (нейтральный атом, ядро, отдельные оболочки).
  """
  ## --- 1. Чтение файла ---
  file_name=txt
  data=[]
  with open (file_name) as file:
    for line in file:
      data.append(line)
  # data = block_format(data)     ## НЕУДАЧНЫЙ ХОД 
  ## --- 2. Разбиение файла на блоки ---
  indexes=[]
  for i in range(len(data)):
    if len(data[i].split())==0: indexes.append(i)
  check_end=[]
  for i in range(indexes[-1],len(data)): check_end.append(len(data[i].split()))
  if sum(check_end)==0: indexes=indexes[:(len(indexes)-1)]

  core_subshells, valence_subshells, populations =[],[],[]
  for line in [linei.split() for linei in data[:indexes[0]] if  'Core subshells:' not in linei]: core_subshells+=line
  for line in [linei.split() for linei in data[(indexes[0]+1):indexes[1]] if  'Valence subshells:' not in linei]: valence_subshells+=line
  for line in [linei.split() for linei in data[(indexes[2]+1):indexes[3]] if  'Populations:' not in linei]: populations+=line

  subshells, core, valence = {}, {}, {}
  for i in range(len(core_subshells)): core[core_subshells[i]]={'P': populations[i]}
  for i in range(len(valence_subshells)): valence[valence_subshells[i]]={'P': populations[len(core_subshells)+i]}
  subshells={'core': core, 'valence': valence}

  ## --- 3. Кривые ---
  names_curves_valence=[]                                                       ## Объединяем оболочки типа p и p-
  for shell in valence_subshells:
    if shell.replace('-','') not in names_curves_valence: names_curves_valence.append(shell.replace('-',''))
  points=np.arange(0,len(names_curves_valence)+2,0.4)/(len(names_curves_valence)+2)
  my_color_scale=sample_colorscale(colorscale='RdBu', samplepoints=points, low=0.0, high=1.0, colortype='rgb')
  subshells['curves']={}
  subshells['curves']['neutral atom']=get_curve(index1=indexes[3],index2=indexes[4],data=data,name_curve='Neutral atom',color= my_color_scale[0])
  subshells['curves']['core']        =get_curve(index1=indexes[4],index2=indexes[5],data=data,name_curve='Core',color=my_color_scale[1])

  for i in range(5,len(indexes)):
    index1=indexes[i]
    index2=indexes[i+1] if (i+1)!=len(indexes) else None
    subshells['curves'][names_curves_valence[i-5]]=get_curve(index1=index1,index2=index2,data=data,name_curve='Valence: '+names_curves_valence[i-5],color=my_color_scale[i-3])
  return subshells


## ===== Получение кривых рассеяния данной оболочки (dict: x, y, trace) =======
def get_curve(index1,index2,data,name_curve,color,dash=None): #dash='dash'
  """
  Извлекает числовые данные кривой рассеяния из текстового блока 
  и формирует объект визуализации Plotly.

  Parameters
  ----------
  index1, index2 : int    ← Индексы начала и конца блока данных в файле.
  data : list of str      ← Содержимое исходного файла, построчно.
  name_curve : str        ← Имя (легенда) для отображаемой кривой.
  color : str             ← Цвет линии на графике (формат RGB или HTML).
  dash : str, optional    ← Стиль линии (например, ``'dash'``).

  Returns
  -------
  dict  ← Словарь с ключами:
        - 'x'     — массив значений sin(θ)/λ;
        - 'y'     — амплитуды рассеяния;
        - 'trace' — объект ``plotly.graph_objects.Scatter`` для визуализации.
  """
  curve=[]
  if index2==None: index2=len(data)
  for i in range(index1+1,index2):
    curve = curve+data[i].replace('\n','').split('  ')
  curve = np.array([float(i) for i in curve if i not in ['']])
  stl   = np.array([0.05*i for i in range(len(curve))])
  tr    = go.Scatter(x=stl, y=curve, name=name_curve, line=dict(color=color, dash=dash),xaxis="x", yaxis="y")
  return {'x':stl,'y':curve, 'trace': tr}




# ============ Чтение Aspherical ============

## ===== Чтение параметризации кривых рассеяния для электронов (aspherical)
def read_aspher_scatfile(txt):                   
  """
  Считывает параметры аппроксимации кривых электронного рассеяния 
  для асферических атомных моделей из текстового файла.

  Parameters
  ----------
  txt : str    Имя текстового файла (формат .txt), содержащего коэффициенты 
               параметризации в виде рядов a_i и b_i.

  Returns
  -------
  dict        Словарь с параметрами вида {'shell': {'a': [...], 'b': [...]}}, 
              включающий данные для нейтрального атома, ядра и валентных оболочек.
  """
  file_name=txt
  data=[]
  with open (file_name) as file:
    for line in file:
      if len(line.split())>0: data.append(line)
  parametrizations={'neutral atom': {'a': [], 'b': []},
                    'core':         {'a': [], 'b': []}}
  parametrizations['neutral atom']['a']=[float(a) for a in data[1].split()[1:]]
  parametrizations['neutral atom']['b']=[float(a) for a in data[2].split()]

  parametrizations['core']['a']=[float(a) for a in data[3].split()[1:]]
  parametrizations['core']['b']=[float(a) for a in data[4].split()]

  for i in range(2,int((len(data)-1)/2)):
    name_of_shell=data[2*i+1].split()[0][2:-1]
    parametrizations[name_of_shell]={}
    parametrizations[name_of_shell]['a']=[float(a) for a in data[2*i+1].split()[1:]]
    parametrizations[name_of_shell]['b']=[float(a) for a in data[2*i+2].split()]
  return parametrizations



__all__ = ["block_format", "read_scatfile", "get_curve"]