import numpy as np
import math
from math import prod
import re
from scipy.interpolate import interp1d
from atoms.scattering_factors.parametric_factors import PARAM

## ===== Модель анизотропных/ангармонических ADP =====

def convol_ADP_h(h,k,l, prefix_KPhase, atom_name, ADP_type, **pars):             ## Напр., C_ijk*h_i*h_j*h_k, умноженное на коэффициент в разложении и на порядок малости
  """
  Вычисляет свёртку тензорных коэффициентов ангармонических параметров атома
  с показателями Миллера (h, k, l) для заданного типа ADP.

  Parameters
  ----------
  h, k, l : int        ← Индексы отражения.
  prefix_KPhase : str  ← Префикс фазы (например, 'P1_').
  atom_name : str      ← Имя атома в структуре.
  ADP_type : str       ← Тип параметров ('B', 'C', 'D', 'E', 'F').
  **pars : dict        ← Словарь параметров уточнения

  Returns
  -------
  complex              ← Суммарный вклад данного типа ADP в экспоненту структурного фактора.
  """
  
  dict_type_coeff={'B': -1,
                   'C': -1j * 1e-3,
                   'D':  1  * 1e-4,
                   'E':  1j * 1e-5,
                   'F': -1  * 1e-6}
  full_prefix = f"{prefix_KPhase}{atom_name}_{ADP_type}"
  pars_ADP    = dict([(k,v) for k,v in pars.items() if (full_prefix in k) and ('Biso' not in k)])
  hlk    = {'1':h, '2':k, '3':l}
  convol = 0
  for k_par_ADP, v_par_ADP in pars_ADP.items():
    indexes = [int(i) for i in k_par_ADP.split('_')[-1][1:]]
    list_hi = [hlk[str(i)] for i in indexes]
    convol  += pars_ADP[k_par_ADP]*prod(list_hi)
  return convol*dict_type_coeff[ADP_type]



def ADPanharmonic(h,k,l, prefix_KPhase, atom_name, order, **pars):
  """
  Рассчитывает ангармоническую поправку атомных смещений (ADP) заданного порядка.

  Parameters
  ----------
  h, k, l : int        ← Индексы отражения.
  prefix_KPhase : str  ← Префикс фазы (например, 'Phase1_').
  atom_name : str      ← Имя атома.
  order : int          ← Порядок ангармонизма (3–6).
  **pars : dict        ← Словарь параметров уточнения.

  Returns
  -------
  complex              ← Суммарная ангармоническая поправка для данного отражения.
  """
  dict_order_type={'3': ['C'],
                   '4': ['C', 'D'],
                   '5': ['C', 'D', 'E'],
                   '6': ['C', 'D', 'E', 'F']}
  ADP = 1
  if order!=0:
    for ADP_typei in dict_order_type[str(order)]:
      ADP += convol_ADP_h(h,k,l,prefix_KPhase, atom_name, ADP_type=ADP_typei, **pars)
  return ADP




## ===== Кривая рассеяния электронов (IT-параметризация) =======
def fe(stl: float, element_ID: str) -> float:
  """
  Рассчитывает функцию рассеяния электронов для заданного элемента и значения (sin θ)/λ.
  
  Parameters
  ----------
  stl (float): Угловой параметр (sin θ)/λ в обратных ангстремах (Å⁻¹).
  element_ID (str): Идентификатор элемента, например, "Na", "Ca", "F".
  
  Returns
  -------
  float: Значение функции рассеяния электронов f_e в ангстремах (Å).
  
  Examples
  -----
  print("Функция рассеяния электронов fe(1.5 Å⁻¹, Na) = %f Å"%(fe(1.5, "Na")))
  """
  element=PARAM["elements"][element_ID]

  f=0
  f+=element["a1"]*math.exp(-element["b1"]*stl*stl)
  f+=element["a2"]*math.exp(-element["b2"]*stl*stl)
  f+=element["a3"]*math.exp(-element["b3"]*stl*stl)
  f+=element["a4"]*math.exp(-element["b4"]*stl*stl)
  f+=element["a5"]*math.exp(-element["b5"]*stl*stl)
  return f


## ========== Каппа-модель ==========

def fe_el_kmodel(stl, prefix_KPhase, atom_name, curves, **pars):
  """
  Вычисляет электронный атомный фактор рассеяния в рамках κ-модели
  по формуле Мотта–Бете на основе параметров уточнения и табулированных
  рентгеновских факторов для остова и валентных оболочек.

  Parameters
  ----------
  stl : float or ndarray  ← Обратное расстояние (sinθ/λ) в Å⁻¹.
  prefix_KPhase : str     ← Префикс, идентифицирующий фазу (например, 'Phase1_').
  atom_name : str         ← Имя атома в структуре.
  curves : dict           ← Словарь с интерполяционными данными для остова ('core') и валентных оболочек 
                            (содержится в атрибуте объекта Atoms)
  **pars : dict           ← Параметры сжатия κ и коэффициенты заселенности P для каждой оболочки
                            (например, 'Phase1_Ca_4s_kappa', 'Phase1_Ca_4s_P').

  Returns
  -------
  float or ndarray    ← Электронный атомный фактор рассеяния fₑ(k) в единицах атомных единиц.
  """
  full_prefix=prefix_KPhase+atom_name+'_'
  a0     = 0.529177210544                           ## Боровский радиус a0 = 5.29177210544(82)⋅10^(−11) м = 0.529177210544(82) Å
  factor = 1/(8*math.pi**2*a0)                      ## В единицах Å^(-1). Это коэффициент перед скобкой
  element_ID = re.sub("[^A-Za-z]", "", atom_name)
  Z = PARAM["elements"][element_ID]['Z']            ## Порядковый номер элемента

  ## --- Остов ----
  interpol_core = interp1d(curves['core']['x'],curves['core']['y'], kind='quadratic', fill_value="extrapolate")   # quadratic - сплайновая интерполяция второго порядка. fill_value - экстраполяция при выходе из диапазона stl_max (имеет место при kappa<1)
  fe_x_core     = interpol_core(stl)                ## Фактор рассеяния для рентгена на остове
  ## --- Валентные оболочки ----
  P_fe_valence = 0
  names_curves_valence = [k for k,v in curves.items() if k not in ['neutral atom', 'core']]
  ## --- Сумма по оболочкам ----
  for shell in names_curves_valence:
    interpol_valence_subshells=interp1d(curves[shell]['x'],curves[shell]['y'], kind='quadratic', fill_value="extrapolate")
    P     = pars[full_prefix+shell+'_P']
    kappa = pars[full_prefix+shell+'_kappa']
    fe_x_valence_subshell = interpol_valence_subshells(stl/kappa)    ## Фактор рассеяния для рентгена на валентной оболочке 4s
    P_fe_valence += P*fe_x_valence_subshell                          ## Вклад в сумму для данной оболочки
  fe_el = factor/stl**2*(Z-fe_x_core-P_fe_valence)                   ## Расчет фактора рассеяния для электронов по формуле Мотта-Бете
  return fe_el




## ===== Асферическая модель электронной оболочки ========
## ===== (в разбаротке, брошена за ненадобностью) ========
def f_ab8(stl,a,b):
  """
  Вычисляет атомный фактор рассеяния по гауссовой аппроксимации.
  f(s) = Σ a_i * exp(-b_i * s²).
  """
  f_s=0
  for ai,bi in zip(a,b):
    f_s=f_s+ai*np.exp(-bi*stl**2)
  return f_s

def fe_el_asherical(stl, prefix_KPhase, atom_name, parametrisations, **pars):
  """
  Вычисляет электронный атомный фактор рассеяния в рамках асферической модели
  с учётом угловой деформации валентной оболочки.

  Parameters
  ----------
  stl : float or ndarray    ← Обратное расстояние (sinθ/λ) в Å⁻¹.
  prefix_KPhase : str       ← Префикс, идентифицирующий фазу (например, 'Phase1_').
  atom_name : str           ← Имя атома в структуре.
  parametrisations : dict   ← Параметры аппроксимации (a, b) для остова и валентных оболочек,
                              включая выделенную асферическую оболочку.
  **pars : dict             ← Параметры модели: коэффициенты заселённости P, сжатия κ и
                              угловой параметр асферичности (angle, градусы).

  Returns
  -------
  float or ndarray    ← Электронный фактор рассеяния fₑ(k) с учётом асферической деформации.
  """

  full_prefix=prefix_KPhase+atom_name+'_'
  dict_part={'p': {'sph':  ['p0'],
                   'asph': ['p1']}}
  spherical_shells=[k for k,v in parametrisations.items() if k not in ['neutral atom', 'core', 'aspherical shell']+dict_part[re.sub("[^A-Za-z]", "", parametrisations['aspherical shell'])]['asph']]
  print(spherical_shells)

  fe_el=f_ab8(stl=stl,a=parametrisations['core']['a'],b=parametrisations['core']['b'])                            ## остов - несжимаемый
  for shell in spherical_shells:
    prefix_par = full_prefix+shell if shell not in dict_part[re.sub("[^A-Za-z]", "", parametrisations['aspherical shell'])]['sph'] else full_prefix+parametrisations['aspherical shell']+'_sph'
    P           = pars[prefix_par+'_P']
    kappa       = pars[prefix_par.replace('_sph','')+'_kappa']
    print(prefix_par+'_P', prefix_par.replace('_sph','')+'_kappa')
    a, b        = parametrisations[shell]['a'], parametrisations[shell]['b']

  if re.sub("[^A-Za-z]", "", parametrisations['aspherical shell'])=='p':
    angle=pars[full_prefix+parametrisations['aspherical shell']+'_angle']
    angle_coeffs=[1.5*(math.sin(angle/180*math.pi))**2,  (math.cos(angle/180*math.pi))**2-0.5*(math.sin(angle/180*math.pi))**2]
    for shell in spherical_shells:                                            ## Суммируем вклады от валентных оболочек с учетом их заселенности и сжатия
      P     = pars[full_prefix+shell+'_P']
      kappa = pars[full_prefix+shell+'_kappa']
      a, b  = parametrisations[shell]['a'], parametrisations[shell]['b']
      angle_coeff=angle_coeffs[spherical_shells.index(shell)] if shell in spherical_shells else 1
      fe_el=fe_el+P*angle_coeff*f_ab8(stl/kappa,a,b)
      print(shell, '  P =',P.value ,'kappa =',kappa.value, 'angle coeff =',angle_coeff, 'вклад =', P*angle_coeff*f_ab8(stl/kappa,a,b))
      #y_sum=f(stl=x,a=data['core']['a'],b=data['core']['b']) + 2*f(stl=x,a=data['2s']['a'],b=data['2s']['b']) + 5*f(stl=x,a=data['p0']['a'],b=data['p0']['b']) + 0*f(stl=x,a=data['p1']['a'],b=data['p1']['b'])
  return fe_el



__all__ = ["convol_ADP_h", "ADPanharmonic",
           "fe", "fe_el_kmodel"]