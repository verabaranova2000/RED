from dataclasses import dataclass, field
from typing import Literal
from utils.observable import ObservableSettings


"""
Эти dataclass'ы описывают только состояние.
Вся логика использования настроек реализуется в Phase и связанных модулях.
"""

@dataclass
class BlackmanSettings(ObservableSettings):
    """
    Настройки поправки Блэкмана для расчёта интенсивности рефлексов.

    Атрибуты
    ----------
    mode : bool
        Флаг включения поправки Блэкмана.
        Если False — поправка не применяется ни к одному рефлексу.

    corrections : list
        Список рефлексов, к которым применяется поправка.
        Формат элементов: [h, k, l] (список целых чисел) или строка 'all'.

        Поведение:
        - если 'all' → поправка применяется ко всем рефлексам;
        - если список → только к указанным рефлексам;
        - если пусто → ни к одному (даже при mode=True).
    """
    LEGACY_MAPPING = {"mode": "mode",
                      "corrections": "corrections",}    
    mode: bool = False
    corrections: list = field(default_factory=list)
    
    #def to_legacy_dict(self):
    #    """
    #    Преобразует настройки в словарь старого формата (для сохранения в файл).
    #    """        
    #    return {"mode": self.mode,
    #            "corrections": list(self.corrections)}

    #classmethod
    #def from_legacy_dict(cls, d):
    #    """
    #    Создаёт объект из словаря старого формата.

    #    Параметры
    #    ----------
    #    d : dict
    #        Словарь вида {"mode": ..., "corrections": ...}

    #    Возвращает
    #    ----------
    #    BlackmanSettings
    #    """        
    #    obj = cls()
    #    obj.mode = d.get("mode", False)
    #    obj.corrections = d.get("corrections", [])
    #    return obj



@dataclass
class InternalSettings(ObservableSettings):
    """
    Внутренние параметры фазы, используемые в расчёте интенсивности.

    Атрибуты
    ----------
    internal_scale : float
        Масштабный коэффициент для интенсивностей в режиме 'le Beil'.

        Используется как множитель:
            I_hkl = internal_scale * I_param

        Не влияет на расчёт в режиме 'Rietveld'.
    """  
    LEGACY_MAPPING = {"internal_scale": "internal scale",}      
    internal_scale: float = 1.0\

    #def to_legacy_dict(self):
    #    """
    #    Преобразует настройки в словарь старого формата (для сохранения в файл).
    #    """            
    #    return {"internal scale": self.internal_scale}

    #classmethod
    #def from_legacy_dict(cls, d):
    #    """
    #    Создаёт объект из словаря старого формата.

    #    Параметры
    #    ----------
    #    d : dict

    #    Возвращает
    #    ----------
    #    InternalSettings
    #    """        
    #    obj = cls()
    #    obj.internal_scale = d.get("internal scale", 1.0)
    #    return obj



@dataclass
class PhaseSettings(ObservableSettings):
    """
    Настройки фазового объекта, определяющие поведение расчёта профиля.

    Атрибуты
    ----------
    typeref : {"Rietveld", "le Beil"}
        Режим расчёта интенсивностей:
        - "Rietveld" → интенсивности вычисляются через структурный фактор;
        - "le Beil" → интенсивности задаются как параметры I_hkl.

    corrections : list
        Список рефлексов [h, k, l], для которых используется режим "le Beil"
        даже при typeref="Rietveld".

    calibration_mode : bool
        Флаг режима калибровки положения пиков.

    calibrate : list
        Список рефлексов для калибровки или строка 'all'.

        Поведение:
        - 'all' → калибруются все рефлексы;
        - список → только указанные;
        - пусто → калибровка отключена.

    blackman : BlackmanSettings
        Настройки поправки Блэкмана.

    form : {"Lorentzian", "Gaussian", "PseudoVoigt"}
        Модель формы пика.

    uvar : tuple
        Кортеж имён параметров, используемых как уточняемые переменные
        при построении профиля.

    internal : InternalSettings
        Внутренние параметры масштабирования интенсивностей.
    """    
    LEGACY_MAPPING = {"typeref": "typeref",
                      "corrections": "corrections",
                      "calibration_mode": "calibration mode",
                      "calibrate": "calibrate",
                      "blackman": "Blackman",
                      "form": "form",
                      "uvar": "uvar",
                      "internal": "internal",}    
    typeref: Literal["Rietveld", "le Beil"] = "Rietveld"
    corrections: list = field(default_factory=list)
    calibration_mode: bool = False
    calibrate: list = field(default_factory=list)
    blackman: BlackmanSettings = field(default_factory=BlackmanSettings)
    form: Literal["Lorentzian", "Gaussian", "PseudoVoigt"] = "Lorentzian"
    uvar: tuple = ("scale", "form", "I_h_k_l")
    internal: InternalSettings = field(default_factory=InternalSettings)

    #def to_legacy_dict(self):     
    #    return {"typeref": self.typeref,
    #        "corrections": list(self.corrections),
    #        "calibration mode": self.calibration_mode,
    #        "calibrate": list(self.calibrate),
    #        "Blackman": self.blackman.to_legacy_dict(),
    #        "form": self.form,
    #        "uvar": list(self.uvar),
    #        "internal": self.internal.to_legacy_dict()}

    #classmethod
    #def from_legacy_dict(cls, d):  
    #    obj = cls()
    #    obj.typeref = d.get("typeref", "Rietveld")
    #    obj.corrections = d.get("corrections", [])
    #    obj.calibration_mode = d.get("calibration mode", False)
    #    obj.calibrate = d.get("calibrate", [])
    #    obj.blackman = BlackmanSettings.from_legacy_dict(d.get("Blackman", {}))
    #    obj.form = d.get("form", "Lorentzian")
    #    obj.uvar = tuple(d.get("uvar", ("scale", "form", "I_h_k_l")))
    #    obj.internal = InternalSettings.from_legacy_dict(d.get("internal", {}))
    #    return obj