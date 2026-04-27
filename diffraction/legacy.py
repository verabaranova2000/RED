import math
import cmath
import numpy as np
import re
from atoms.generate import get_all_positions_in_cell_for_atom
from phases.models import models_dict, par_form_dict
from phases.params import FORM, hkl_to_str
from scipy.interpolate import interp1d
from atoms.scattering_factors.parametric_factors import PARAM

"""
LEGACY IMPLEMENTATION (non-JAX)

⚠️ Этот модуль содержит старую (не-JAX) реализацию расчёта:
- используется только для отладки и сравнения
- не должен использоваться в основном пайплайне

Актуальная реализация: diffraction/*
"""

"""
LEGACY (non-JAX) implementation.

Used for:
- validation
- debugging
- comparison

Not recommended for new code.
"""


"""
LEGACY (non-JAX) pipeline     
------
Implementation type:
    - scalar (per-hkl loop)

Used for:
- validation
- debugging
- comparison

phase_profile
      │
  iterate_hkl (loop over hkl)
      │
      ├── Σₕₖₗ phase_profile_hkl
      │       ├── peak position (2θ + Δ(2θ))     
      │       ├── compute_intensity
      │       │       ├── expand_atom_sites              
      │       │       ├── F2_hkl ( |F|² )
      │       │       │     └── d_hkl
      │       │       └── blackman_correction
      │       │              
      │       └── peak model ( Gaussian / PseudoVoigt / etc )
      │
      └── L_of_ring correction


Дополнительно (диагностика / сравнение с JAX):
------

F2_array          — массив |F|² для набора hkl

intensity_array   — массив I_hkl для набора hkl
      │
  iterate_hkl (loop over hkl)
      │
      └── intensity_hkl
              └── compute_intensity
"""


# =========================
# tools
# =========================

def iterate_hkl(project_object, prefix_KPhase):
    """ Общий генератор hkl """
    phase_object = project_object.__dict__.get(prefix_KPhase.replace('_',''))
    for line in phase_object.bragg_positions:    # цикл по всем брегговским пикам (h,k,l) для данной фазы
        h, k, l = line[:3]
        p = line[3]                              # фактор повторяемости для данного пика (h,k,l)
        yield h, k, l, p


def expand_atom_sites(phase_object,h=None,k=None,l=None,stl=None, **pars):
  """"" 
  Подготовка полного набора атомов в ячейке 

  Распаковка атомных параметров и формирование массива XYBiso_all
  всех атомов в ячейке, требуемый для расчета структурных амплитуд.
  """""
  prefix = phase_object.prefix
  types_of_atoms =[atom.name for atom in phase_object.atoms]
  XYZoccBiso=[]
  for atom_type in types_of_atoms:
    x_key,y_key,z_key,occ_key,Biso_key = prefix+atom_type+'_x', prefix+atom_type+'_y', prefix+atom_type+'_z', prefix+atom_type+'_occ', prefix+atom_type+'_Biso'  # формируем названия переменных для координат атома
    xa      = pars.get(x_key)                                                    # Извлекаем координаты атома
    ya      = pars.get(y_key)
    za      = pars.get(z_key)
    occ     = pars.get(occ_key)
    Biso_at = pars.get(Biso_key)                                                # Извлекаем тепловой параметр
    if (h is not None) and (k is not None) and (l is not None) and (stl is not None):
      t_at    = phase_object.atoms[types_of_atoms.index(atom_type)].t_at(h,k,l,stl,**pars)
      fe_el   = phase_object.atoms[types_of_atoms.index(atom_type)].fe_el(h,k,l,stl,**pars)
      #ADP=my_phase.atoms[types_of_atoms.index(atom_type)].ADP(h,k,l,**pars)
      #B=my_phase.atoms[types_of_atoms.index(atom_type)].B(h,k,l, **pars)

    XYZoccBiso.append([xa,ya,za, occ, t_at, fe_el, atom_type])
  # Размножим атомные позиции операциями симметрии
  XYZoccBiso_all=[]
  for line in XYZoccBiso:
    name=line[-1]           # Имя атома
    x,y,z=line[:3]          # Координаты атома
    occ=line[3]
    #Biso=line[4]            # Изотропная тепловая поправка атома
    t_at =    line[4]        # Атомная тепловая поправка атома
    fe_el = line[5]
    positions = get_all_positions_in_cell_for_atom(x, y, z, phase_object.symmetry_operations)
    all_info_for_atom=[list(one_pos)+[occ]+[t_at]+[fe_el]+[name] for one_pos in positions]
    XYZoccBiso_all=XYZoccBiso_all+all_info_for_atom
  return XYZoccBiso_all




# =========================
# geometry
# =========================

def d_hkl(h,k,l, a,b,c,alpha,beta,gamma):
    """ Межплоскостные расстояния """
    c_α, c_β, c_γ = math.cos(alpha/180*math.pi), math.cos(beta/180*math.pi), math.cos(gamma/180*math.pi)
    s_α, s_β, s_γ = math.sin(alpha/180*math.pi), math.sin(beta/180*math.pi), math.sin(gamma/180*math.pi)
    ω=(1-c_α**2-c_β**2-c_γ**2+2*c_α*c_β*c_γ)**0.5
    C1=(h/(a/s_α))**2+(k/(b/s_β))**2+(l/(c/s_γ))**2                                               # Первые три слагаемых
    C2=2*h*k/(a*b)*(c_α*c_β-c_γ)+2*h*l/(a*c)*(c_γ*c_α-c_β)+2*k*l/(b*c)*(c_β*c_γ-c_α)              # Первые три слагаемых
    D=(1/ω**2)*(C1+C2)                                                                            # 1/d^2
    d=1/D**0.5 if not h==k==l==0 else 0
    return d


# =========================
# scattering_factor
# =========================

# ---- IT4322 fₑₗ ----
def f_el(stl: float, element_ID: str) -> float:
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
  print("Функция рассеяния электронов f_el(1.5 Å⁻¹, Na) = %f Å"%(f_el(1.5, "Na")))
  """
  element=PARAM["elements"][element_ID]

  f=0
  f+=element["a1"]*math.exp(-element["b1"]*stl*stl)
  f+=element["a2"]*math.exp(-element["b2"]*stl*stl)
  f+=element["a3"]*math.exp(-element["b3"]*stl*stl)
  f+=element["a4"]*math.exp(-element["b4"]*stl*stl)
  f+=element["a5"]*math.exp(-element["b5"]*stl*stl)
  return f


# ---- Каппа-модель fₑₗ ----
def f_el_kmodel(stl, prefix_KPhase, atom_name, curves, **pars):
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




# =========================
# structure_factor
# =========================

def F2_hkl(h,k,l,  a,b,c,alpha,beta,gamma, XYZoccBiso_all, Biso_overall):
  const = 2*cmath.pi*cmath.sqrt(-1)      # коэффициент 2*pi*i в экспоненте
  stl = 1/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)) if not h==k==l==0 else 0
  F = 0
  for x,y,z,occ,t_at,fe_el,atom_type_symbol in XYZoccBiso_all:        # по всем атомам внутри элементарной ячейки
    t = math.exp(-(Biso_overall)*stl**2)*t_at                         # поправка на тепловые колебания
    F = F+fe_el*cmath.exp(const*(h*x+k*y+l*z))*t*occ
  return abs(F)**2


def FF_array(hkl_array, phase_object, **pars):                       # Возвращает массив теоретических интенсивностей
  """ 
  Не используется в расчете дифракционного профиля. 
  Нужен для сравнения старой и jax-версий.
  """
  prefix = phase_object.prefix
  a,b,c,alpha,beta,gamma = [phase_object.param_cell.get(prefix+par).value for par in ['a','b','c','alpha','beta','gamma']]
  Biso_overall = pars.get(prefix+'Biso_overall')
  FF_array=[]
  for h,k,l in hkl_array:
    stl = 1/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)) if not h==k==l==0 else 0
    XYZoccBiso_all = expand_atom_sites(phase_object, h=h,k=k,l=l,stl=stl,**pars)
    FF_array.append(F2_hkl(h,k,l,  a,b,c,alpha,beta,gamma, XYZoccBiso_all, Biso_overall))
  return np.array(FF_array) 



# =========================
# intensity
# =========================
def blackman_correction(F_mod, A, eps=1e-6):
    """ Поправка по Блэкману """
    x = A * F_mod
    if abs(x) < eps:
        return 1.0     # предел при A → 0
    return (1 - np.exp(-x))/x


def compute_intensity(project_object, prefix_KPhase, h, k, l, stl, p, pars):
    my_phase = project_object.__dict__.get(prefix_KPhase.replace('_', ''))
    s = my_phase.settings
    internal_scale = s.internal.internal_scale
    # --- Разные способы получения амплитуды пика в зависимости от settings ---
    if (s.typeref == 'le Beil') or ([h, k, l] in s.corrections):
        return internal_scale * pars.get(f"{prefix_KPhase}I_{hkl_to_str([int(h), int(k), int(l)])}")   # Получили уточняемый параметр интенсивности
    if (s.typeref == 'Rietveld') and ([h, k, l] not in s.corrections):
        scale = pars.get(prefix_KPhase + 'scale')
        phvol = pars.get(prefix_KPhase + 'phvol')
        XYZoccBiso_all = expand_atom_sites(my_phase, h=h, k=k, l=l, stl=stl, **pars)
        Biso_overall = pars.get(prefix_KPhase + 'Biso_overall')
        F2 = F2_hkl(h, k, l, *[pars.get(prefix_KPhase + par) for par in ['a','b','c','alpha','beta','gamma']],
                XYZoccBiso_all, Biso_overall)
        # --- Поправка по Блэкману в зависимости от settings ---
        if (s.blackman.mode is True) and (s.blackman.corrections == 'all' or ([h, k, l] in s.blackman.corrections)):
            A = pars.get(prefix_KPhase + 'A')
            blackman_corr = blackman_correction(F2**0.5, A)
        else:
            blackman_corr = 1
        return scale * phvol * p * F2 * blackman_corr
    raise ValueError(f"Не удалось вычислить амплитуду для {h,k,l}")



def intensity_hkl(project_object, h, k, l, a, b, c, alpha, beta, gamma, λ, p,
                   prefix_KPhase, **pars):
    """ Амплитуда одного пика """
    stl = 1 / (2 * d_hkl(h, k, l, a, b, c, alpha, beta, gamma)) if not h == k == l == 0 else 0
    return compute_intensity(project_object, prefix_KPhase, h, k, l, stl, p, pars)


def intensity_array(project_object, prefix_KPhase, **pars):
    """ Сборка всех амплитуд """
    cell_array      = [pars.get(prefix_KPhase+par) for par in ['a','b','c','alpha','beta','gamma']]
    my_phase        = project_object.__dict__.get(prefix_KPhase.replace('_',''))              # Извлекаем фазу из проекта, чтобы получить форму и индексы рефлексов
    λ               = my_phase.wavelength

    Ampl_array_check=[]
    for h, k, l, p in iterate_hkl(project_object, prefix_KPhase):
      Ampl_array_check.append(intensity_hkl(project_object, h,k,l,  *cell_array,  λ, p, prefix_KPhase, **pars))
    return Ampl_array_check


# =========================
# profile
# =========================
def phase_profile_hkl(axes, h,k,l,   a,b,c, alpha,beta,gamma, λ, p, project_object, KPhase, form: FORM, uvar=False, **pars):
  """
  Профиль одного пика
  """
  prefix_KPhase   = KPhase
  my_phase        = project_object.__dict__.get(prefix_KPhase.replace('_',''))
  s               = my_phase.settings
  model_name      = form    # Модельный профиль

  """"" Поправка для положения пика в зависимости от settings """""
  if (s.calibration_mode is True) and (s.calibrate == 'all' or ([h,k,l] in s.calibrate)):
    delta_hkl=pars.get(f"{prefix_KPhase}delta_{hkl_to_str([int(h), int(k), int(l)])}")
  else: delta_hkl=0
  x_hkl = 2*math.asin(λ/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)))*180/math.pi - delta_hkl    if not h==k==l==0 else 0          # теоретическое положение рефлекса с учетом поправки (поправки потребуются для калибровки шкалы)

  stl=1/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)) if not h==k==l==0 else 0

  Ampl = compute_intensity(project_object, prefix_KPhase, h, k, l, stl, p, pars)
  peak_pars = {'A': Ampl, 'μ': x_hkl}

  ### достанем другие параметры для этой модели формы пика:
  par_names = [line['name'] for line in  par_form_dict[model_name]]      # Список с названиями параметров (без префиксов)
  for i in range(2,len(par_names)):                                      # Ищем в **pars остальные параметры (кроме 'A' и 'μ') и добавляем их в словарь с параметрами формы пиков
    peak_pars[par_names[i]]=pars.get(prefix_KPhase+model_name+'_'+par_names[i])
  f = models_dict.get(model_name)                                          # извлекаем функцию из словаря с моделями
  return f(axes, uvar=uvar, **peak_pars)                                            # возвращаем профиль брэгговского пика в виде столбца numpy




def phase_profile(axes, project_object=None, KPhase=None, uvar=False, **pars):                            # shift, scale, H_L,   a,b,c,α,β,γ,  λ,
    """
    Суммарный профиль для фазы

    Задаем функцию с ключевым словом (префикс 'Phase1_', например)
    """
    assert KPhase is not None
    prefix_KPhase = KPhase

    Biso_overall = pars.get(prefix_KPhase+'Biso_overall')
    shift        = pars.get(prefix_KPhase+'shift')
    cell_array   = [pars.get(prefix_KPhase+par) for par in ['a','b','c','alpha','beta','gamma']]
    axes=axes+shift                                                               # Ось абсцисс с учетом сдвига нуля

    my_phase        = project_object.__dict__.get(prefix_KPhase.replace('_',''))              # Извлекаем фазу из проекта, чтобы получить форму и индексы рефлексов
    model_name      = my_phase.settings.form
    λ               = my_phase.wavelength
    I_calc=0
    for h, k, l, p in iterate_hkl(project_object, prefix_KPhase):
      I_calc += phase_profile_hkl(axes, h,k,l, *cell_array, λ, p, project_object=project_object, KPhase=prefix_KPhase, form=model_name, uvar=uvar, **pars)
    L_of_ring           = np.array([math.sin(xi/2*math.pi/180)/λ for xi in axes])*2*math.pi   ## Учет размазывания интенсивностей по кольцам:
    I_calc_mean_of_ring = I_calc/L_of_ring
    return I_calc_mean_of_ring
