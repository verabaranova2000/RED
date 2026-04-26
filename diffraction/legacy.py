import math
import cmath
import numpy as np
from atoms.generate import get_all_positions_in_cell_for_atom


"""
LEGACY NUMPY IMPLEMENTATION

⚠️ Этот модуль содержит старую (не-JAX) реализацию расчёта:
- используется только для отладки и сравнения
- не должен использоваться в основном пайплайне

Актуальная реализация: diffraction/*
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



### 0. Распаковка атомных параметров и формирование массива всех атомов в ячейке XYBiso_all, требуемый для расчета структурных амплитуд (при уточнении интенсивности пика методом Rietveld)
def extact_XYZoccBiso_all(phase_object,h=None,k=None,l=None,stl=None, **pars):
  """"" Подготовка полного набора атомов в ячейке """""
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
  prefix = phase_object.prefix
  a,b,c,alpha,beta,gamma = [phase_object.param_cell.get(prefix+par).value for par in ['a','b','c','alpha','beta','gamma']]
  Biso_overall = pars.get(prefix+'Biso_overall')
  FF_array=[]
  for h,k,l in hkl_array:
    stl = 1/(2*d_hkl(h,k,l, a,b,c,alpha,beta,gamma)) if not h==k==l==0 else 0
    XYZoccBiso_all = extact_XYZoccBiso_all(phase_object, h=h,k=k,l=l,stl=stl,**pars)
    FF_array.append(F2_hkl(h,k,l,  a,b,c,alpha,beta,gamma, XYZoccBiso_all, Biso_overall))
  return np.array(FF_array) 



# =========================
# intensity
# =========================

#def intensity_hkl_legacy(...):      # бывший compute_ampl
#def intensity_array_legacy(...):   # бывший Apml_calc


# =========================
# profile
# =========================

#def peak_profile_legacy(...):      # бывший Profile_hkl
#def phase_profile_legacy(...):     # бывший Profile_calc