from .models import *

"""
Словари и метаинформация по моделям формы пиков:
    - models_dict: соответствие имени → функции
    - par_form_dict: параметры для каждой модели
"""

####################################### Словари для функций   ###############################################################################################
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