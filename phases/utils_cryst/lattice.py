import numpy as np
import math

## ====== Межплоскостные расстояния ======
def d_hkl(h,k,l, a,b,c,alpha,beta,gamma):
  """
  Межплоскостное расстояние d(hkl) для произвольной решётки.
  Углы в градусах.
  """
  c_α, c_β, c_γ = math.cos(alpha/180*math.pi), math.cos(beta/180*math.pi), math.cos(gamma/180*math.pi)
  s_α, s_β, s_γ = math.sin(alpha/180*math.pi), math.sin(beta/180*math.pi), math.sin(gamma/180*math.pi)
  ω=(1-c_α**2-c_β**2-c_γ**2+2*c_α*c_β*c_γ)**0.5
  C1=(h/(a/s_α))**2+(k/(b/s_β))**2+(l/(c/s_γ))**2                                               # Первые три слагаемых
  C2=2*h*k/(a*b)*(c_α*c_β-c_γ)+2*h*l/(a*c)*(c_γ*c_α-c_β)+2*k*l/(b*c)*(c_β*c_γ-c_α)              # Первые три слагаемых
  D=(1/ω**2)*(C1+C2)                                                                            # 1/d^2
  d=1/D**0.5 if not h==k==l==0 else 0
  return d

## ====== Объем элементарной ячейки ======
def volume_cell(a, b, c, alpha_deg, beta_deg, gamma_deg):
  """
  Объём элементарной ячейки: V = a*b*c*sqrt(1 - cos^2α - cos^2β - cos^2γ + 2 cosα cosβ cosγ)
  Углы задаются в градусах. Параметры могут быть скалярами или array-like (будут broadcast'иться).
  Возвращает numpy scalar или ndarray (в Å^3).
  """
  alpha    = np.deg2rad(alpha_deg)                          ## Перевод в радианы
  beta     = np.deg2rad(beta_deg)
  gamma    = np.deg2rad(gamma_deg)
  ca,cb,cg = np.cos(alpha), np.cos(beta), np.cos(gamma)     ## Косинусы углов

  term   = 1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg
  volume = a*b*c*np.sqrt(term)
  return volume

# Пример
#V = volume_cell(a, b, c, alpha, beta, gamma)
#print(f"Объём элементарной ячейки: {V:.4f} Å³")