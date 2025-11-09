import numpy as np
import math, cmath
import scipy.special as sc
from uncertainties import ufloat, unumpy



## ======== Функции формы пиков ========
""""" 1. GaussianModel """""                                                    # Работает
def f_Gaussian(axes,A,μ,σ,uvar=False):
  x=axes
  A1=np.exp(-(x-μ)**2/(2*σ**2))    if uvar==False else unumpy.exp(-(x-μ)**2/(2*σ**2))
  f=A*A1/(σ*(2*math.pi)**0.5)
  return f

""""" 2. LorentzianModel """""                                                  # Работает
def f_Lorentzian(axes,A,μ,σ,uvar=False):                              # σ - полуширина. Полная ширина=2σ
  x=axes
  f=A/math.pi*(σ/((x-μ)**2+σ**2))
  return f

""""" 3. SplitLorentzianModel """""
def f_SplitLorentzian(axes,A,μ,σ,σr,uvar=False):                                # Преобразовать в float!!!
  x=axes
  a1=σ**2/((x-μ)**2+σ**2)*np.heaviside(μ-x, 0.5)
  a2=σr**2/((x-μ)**2+σr**2)*np.heaviside(x-μ, 0.5)
  f=2*A/(math.pi*(σ+σr))*(a1+a2)
  return f

""""" 4. VoigtModel """""
def f_Voigt(axes,A,μ,σ,γ,uvar=False):                                           # Преобразовать в float!!!
  x=axes
  z=(x-μ+γ*cmath.sqrt(-1))/(σ*2**0.5)
  w=np.exp(-z**2)*sc.erfc(-cmath.sqrt(-1)*z)
  f=A*w.real/(σ*(2*math.pi)**0.5)
  return f

""""" 5. PseudoVoigtModel """""                                                 # Работает
def f_PseudoVoigt(axes,A,μ,σ,η, uvar=False):
  x=axes
  σg=σ/(2*math.log(2))**0.5                                    # the full width at half maximum of each component
  A1=np.exp(-(x-μ)**2/(2*σg**2))            if uvar==False else unumpy.exp(-(x-μ)**2/(2*σg**2))
  A2=σ/((x-μ)**2+σ**2)
  f=(1-η)*A*A1/(σg*(2*math.pi)**0.5) + η*A*A2/math.pi
  return f

""""" 6. MoffatModel """""                                                      # Работает
def f_Moffat(axes,A,μ,σ,βm,uvar=False):
  x=axes
  f=A*(((x-μ)/σ)**2+1)**(-βm)
  return f

""""" 7. Pearson4Model """""
def f_Pearson4(axes,A,μ,σ,m,v, uvar=False):
  x=axes
  ΓΓ=(abs(sc.gamma(m+cmath.sqrt(-1)*v/2)/sc.gamma(m)))**2
  A2=(1+(x-μ)**2/σ**2)**(-m)
  A3=np.exp(-v*np.arctan((x-μ)/σ))
  f=A*ΓΓ/(σ*sc.beta(m-1/2,1/2))*A2*A3
  return f

""""" 8. Pearson7Model """""
def f_Pearson7(axes,A,μ,σ,m,uvar=False):                                        # Преобразовать в float!!!
  x=axes
  A2=(1+(x-μ)**2/σ**2)**(-m)
  f=A/(σ*sc.beta(m-1/2,1/2))*A2
  return f

""""" 9. StudentsTModel """""
def f_StudentsT(axes,A,μ,σ,uvar=False):                                         # Преобразовать в float!!!
  x=axes
  Γ=sc.gamma((σ+1)/2)/sc.gamma(σ/2)
  A2=(1+(x-μ)**2/σ)**(-(σ+1)/2)
  f=A/(σ*math.pi)*Γ*A2
  return f

""""" 10. BreitWignerModel """""                                                # Работает
def f_BreitWigner(axes,A,μ,σ,q,uvar=False):
  x=axes
  A2=(σ/2)**2+(x-μ)**2
  f=A*(q*σ/2+x-μ)**2/A2
  return f

""""" 11. LognormalModel """""                                                  # Работает
def f_Lognormal(axes,A,μ,σ,uvar=False):
  x=axes
  A2=np.exp(-(np.log(x)-μ)**2/(2*σ**2))/x   if uvar==False else unumpy.exp(-(unumpy.log(x)-μ)**2/(2*σ**2))/x
  f=A*A2/(σ*(2*math.pi)**0.5)
  return f

""""" 12. DampedOscillatorModel """""                                           # Работает
def f_DampedOscillator(axes,A,μ,σ,uvar=False):
  x=axes
  a1=(1-(x/μ)**2)**2
  a2=(2*σ*x/μ)**2
  f=A/(a1+a2)**0.5
  return f

""""" 13. DampedHarmonicOscillatorModel """""                                    # Работает
def f_DampedHarmonicOscillator(axes,A,μ,σ,γ,uvar=False):
  x=axes
  a1=(x-μ)**2+σ**2
  a2=(x+μ)**2+σ**2
  f=A*σ/(math.pi*(1-np.exp(-x/γ)))*(1/a1-1/a2)  if uvar==False else A*σ/(math.pi*(1-unumpy.exp(-x/γ)))*(1/a1-1/a2)
  return f

""""" 14. ExponentialGaussianModel """""                                        # Преобразовать в float!!!
def f_ExponentialGaussian(axes,A,μ,σ,γ,uvar=False):
  x=axes
  A1=np.exp(γ*(μ-x+γ*σ**2/2))
  A2=sc.erfc((μ+γ*σ**2-x)/(2**0.5*σ))
  f=A*γ/2*A1*A2
  return f

""""" 15. SkewedGaussianModel """""                                             # Преобразовать в float!!!
def f_SkewedGaussian(axes,A,μ,σ,γ,uvar=False):
  x=axes
  A1=np.exp(-(x-μ)**2/(2*σ**2))
  A2=1+sc.erf(γ*(x-μ)/(σ*2**0.5))
  f=A*A1*A2/(σ*(2*math.pi)**0.5)
  return f

""""" 16. SkewedVoigtModel """""                                                # Преобразовать в float!!!
def f_SkewedVoigt(axes,A,μ,σ,γ,skew,uvar=False):
  x=axes
  A1=1+sc.erf(skew*(x-μ)/(σ*2**0.5))
  f=f_Voigt(x,A,μ,σ,γ)*A1
  return f




"""
Словари и метаинформация по моделям формы пиков:
    - models_dict: соответствие имени → функции
    - par_form_dict: параметры для каждой модели
"""

## ========= Словари для функций ================
# model reference
models_dict = {
        'Gaussian':                 f_Gaussian,
        'Lorentzian':               f_Lorentzian,
        'SplitLorentzian':          f_SplitLorentzian,
        'Voigt':                    f_Voigt,
        'PseudoVoigt':              f_PseudoVoigt,
        'Moffat':                   f_Moffat,
        'Pearson4':                 f_Pearson4,
        'Pearson7':                 f_Pearson7,
        'StudentsT':                f_StudentsT,
        'BreitWigner':              f_BreitWigner,
        'Lognormal':                f_Lognormal,
        'DampedOscillator':         f_DampedOscillator,
        'DampedHarmonicOscillator': f_DampedHarmonicOscillator,
        'ExponentialGaussian':      f_ExponentialGaussian,
        'SkewedGaussian':           f_SkewedGaussian,
        'SkewedVoigt':              f_SkewedVoigt}

# list of model:
model_list = [k for k,v in models_dict.items()]

par_form_dict = {
        'Gaussian':                 [{'name': 'A',   'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',   'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',   'value': 0.01, 'min': 0,              'max': float('inf')  }],

        'Lorentzian':               [{'name': 'A',   'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',   'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',   'value': 0.01, 'min': 0,              'max': float('inf')  }],

        'SplitLorentzian':          [{'name': 'A',   'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',   'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',   'value': 0.01, 'min': 0,              'max': float('inf')  },
                                     {'name': 'σr',  'value': 0.01, 'min': 0,              'max': float('inf')  }],

         'Voigt':                   [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',  'value': 0.01, 'min': 0,              'max': float('inf')  },
                                     {'name': 'γ',  'value': 0.01, 'min': float('-inf'),  'max': float('inf')  }],

         'PseudoVoigt':             [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',  'value': 0.01, 'min': 0,              'max': float('inf')  },
                                     {'name': 'η',  'value': 0.5,  'min': 0,              'max': 1            }],

         'Moffat':                  [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',  'value': 0.01, 'min': 0,              'max': float('inf')  },
                                     {'name': 'βm', 'value': 1,    'min': float('-inf'),  'max': float('inf')  }],

         'Pearson4':                [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',  'value': 0.01, 'min': float('-inf'),   'max': float('inf')  },
                                     {'name': 'm',  'value': 1,    'min': 0.5,            'max': 1000          },
                                     {'name': 'v',  'value': 0.1,  'min': -1000,          'max': 1000          }],

         'Pearson7':                [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'σ',  'value': 0.01, 'min': float('-inf'),  'max': float('inf')  },
                                     {'name': 'm',  'value': 1,    'min': float('-inf'),  'max': 100           }],

        'StudentsT':                 [{'name': 'A',  'value': 1,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.0001,'min': 0,              'max': 100           }],

        'BreitWigner':               [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.03, 'min': 0,              'max': float('inf')  },
                                      {'name': 'q',  'value': 100,  'min': float('-inf'),  'max': float('inf')  }],

         'Lognormal':                [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },       # НЕ ИСПОЛЬЗОВАТЬ. При μ=0 центр пика находится в точке x=1
                                      {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.01, 'min': 0,              'max': float('inf')  }],

         'DampedOscillator':         [{'name': 'A',  'value': 1,    'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,    'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.01, 'min': 0,              'max': float('inf')  }],

        'DampedHarmonicOscillator':  [{'name': 'A',  'value': 1,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,     'min': 0,              'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.01,  'min': 0,              'max': float('inf')  },
                                      {'name': 'γ',  'value': 0.001, 'min': float(1e-19),   'max': float('inf')  }],

        'ExponentialGaussian':       [{'name': 'A',  'value': 1,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.01,  'min': 0,              'max': float('inf')  },
                                      {'name': 'γ',  'value': 200,   'min': 0,              'max': 20  }],

        'SkewedGaussian':            [{'name': 'A',  'value': 1,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.01,  'min': 0,              'max': float('inf')  },
                                      {'name': 'γ',  'value': 1,     'min': float('-inf'),  'max': float('inf')  }],

        'SkewedVoigt':               [{'name': 'A',  'value': 1,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'μ',  'value': 0,     'min': float('-inf'),  'max': float('inf')  },
                                      {'name': 'σ',  'value': 0.01,  'min': 0,              'max': float('inf')  },
                                      {'name': 'γ',  'value': 0.01,  'min': float('-inf'),  'max': float('inf')  },
                                      {'name':'skew','value': 1,     'min': float('-inf'),  'max': float('inf')  }]}



__all__ = [
    'f_Gaussian',
    'f_Lorentzian',
    'f_SplitLorentzian',
    'f_Voigt',
    'f_PseudoVoigt',
    'f_Moffat',
    'f_Pearson4',
    'f_Pearson7',
    'f_StudentsT',
    'f_BreitWigner',
    'f_Lognormal',
    'f_DampedOscillator',
    'f_DampedHarmonicOscillator',
    'f_ExponentialGaussian',
    'f_SkewedGaussian',
    'f_SkewedVoigt',
    'model_list',
    'par_form_dict',
    'models_dict'
]