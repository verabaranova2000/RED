#@title 🚧 (!) ProfilePointsSettings
from dataclasses import dataclass, field
import math
from typing import Optional, ClassVar
from utils.observable import ObservableSettings



BACKGROUND_TYPE_RULES = {
    "Legendre": {
        "legacy_fields": {"number_of_terms"},
        "defaults": lambda current, two_theta_len: {
            "mode_of_knots": None,
            "use_each_n_point": None,
            "guess_init_vals": None,
            "N_of_knots": None,
            "number_of_terms": min(current.get("number_of_terms", 17), 30),
        },
    },
    "Spline": {
        "legacy_fields": {"mode_of_knots", "use_each_n_point", "guess_init_vals", "N_of_knots"},
        "defaults": lambda current, two_theta_len: {
            "mode_of_knots": current.get("mode_of_knots") or "uniform",
            "use_each_n_point": current.get("use_each_n_point") or math.ceil(two_theta_len / 100),
            "guess_init_vals": True,
            "N_of_knots": current.get("N_of_knots"),
        },
    },
    "Legendre + Spline": {
        "legacy_fields": {"mode_of_knots", "use_each_n_point", "guess_init_vals", "N_of_knots"},
        "defaults": lambda current, two_theta_len: {
            "mode_of_knots": current.get("mode_of_knots") or "uniform",
            "use_each_n_point": current.get("use_each_n_point") or math.ceil(two_theta_len / 100),
            "guess_init_vals": False,
            "N_of_knots": current.get("N_of_knots"),
        },
    },
}

def background_defaults_for_type(bg_type, two_theta_len, current):
    return BACKGROUND_TYPE_RULES[bg_type]["defaults"](current, two_theta_len)


@dataclass
class BackgroundSettings(ObservableSettings):
    MAX_NUMBER_OF_TERMS: ClassVar[int] = 30
    ALLOWED_TYPES: ClassVar[set[str]] = {"Legendre", "Spline", "Legendre + Spline"}
    ALLOWED_MODES_OF_KNOTS: ClassVar[set[str]] = {"uniform","mins", "manually"}
    LEGACY_MAPPING = {"type": "type",
                      "number_of_terms": "number of terms",
                      "mode_of_knots": "mode of knots",
                      "use_each_n_point": "use each n point",
                      "N_of_knots": "N of knots",
                      "guess_init_vals": "guess init vals",}                    
    type: str = "Legendre"
    number_of_terms: Optional[int] = 17

    mode_of_knots: Optional[str] = None
    use_each_n_point: Optional[int] = None
    N_of_knots: Optional[int] = None
    guess_init_vals: Optional[bool] = None


    def snapshot_recursive(self):
        # Не используется
        def convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {
                    k: convert(getattr(obj, k))
                    for k in obj.__dataclass_fields__
                }
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        return convert(self)


    # ==== Валидация полей ====
    # _coerce_value — только маршрутизация.
    # Валидация — в маленьких хелперах.
    # Новых if на каждое поле больше не нужно.
    # Если потом появится ещё одно поле, добавляется одна строка в rules
    def _choice(self, name, value, allowed):
        if value is None:
            return None
        if value not in allowed:
            raise ValueError(f"Недопустимое значение для '{name}': '{value}'. "
                             f"Допустимые значения: {sorted(allowed)}")
        return value

    def _to_int(self, name, value, *, min_value=None, max_value=None):
        if value is None:
            return None
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"'{name}' должно быть целым числом, получено: {value}")
        if min_value is not None and value < min_value:
            raise ValueError(f"'{name}' не может быть меньше {min_value}: {value}")
        if max_value is not None and value > max_value:
            return max_value
        return value

    def _to_bool(self, name, value):
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError(f"'{name}' должно быть True или False, получено: {value}")
        return value

    def _coerce_value(self, name, value):
        rules = {"type": lambda v: self._choice("type", v, self.ALLOWED_TYPES),
                 "number_of_terms": lambda v: self._to_int("number_of_terms", v, min_value=0, max_value=self.MAX_NUMBER_OF_TERMS),
                 "mode_of_knots": lambda v: self._choice("mode_of_knots", v, self.ALLOWED_MODES_OF_KNOTS),
                 "use_each_n_point": lambda v: self._to_int("use_each_n_point", v, min_value=1),
                 "N_of_knots": lambda v: self._to_int("N_of_knots", v, min_value=0),
                 "guess_init_vals": lambda v: self._to_bool("guess_init_vals", v),}
        return rules.get(name, lambda v: v)(value)

    def __post_init__(self):
        """
        Гарантирует валидность начального состояния объекта после создания.

        Назначение
        ----------
        Применяет ограничения к полям (например, number_of_terms),
        когда объект создаётся напрямую через dataclass-конструктор,
        минуя механизм __setattr__ и _coerce_value.

        Обеспечивает, что настройки корректны сразу после инициализации.
        """
        self.number_of_terms = min(self.number_of_terms, self.MAX_NUMBER_OF_TERMS)

    def to_legacy_dict(self):
        """ Переопределяем, т.к. есть условия по type """
        d = {"type": self.type}
        # fields = FIELDS_BY_TYPE.get(self.type, set())             # вариант 1
        fields = BACKGROUND_TYPE_RULES[self.type]["legacy_fields"]  # вариант 2
        for attr, legacy_name in self.LEGACY_MAPPING.items():
            if attr == "type":
                continue          
            if attr in fields:
                value = getattr(self, attr)
                if value is not None:
                    d[legacy_name] = value
        return d


    @classmethod 
    def from_legacy_dict(cls, d): 
      obj = cls() 
      obj.type = d.get("type", "Legendre") 
      obj.number_of_terms = d.get("number of terms", 17) 
      obj.mode_of_knots = d.get("mode of knots", None) 
      obj.N_of_knots = d.get("N of knots", None)
      obj.use_each_n_point = d.get("use each n point", None) 
      obj.guess_init_vals = d.get("guess init vals", None) 
      return obj


@dataclass
class FinderGroupsSettings(ObservableSettings):
    LEGACY_MAPPING = {"window_length": "window_length",
                      "polyorder": "polyorder",
                      "prominence": "prominence",}
    window_length: int = 20
    polyorder: int = 3
    prominence: int = 5


@dataclass
class CalibrationSettings(ObservableSettings):
    LEGACY_MAPPING = {"type": "type",
                      "delta_threshold": "delta threshold",
                      "excluded_manual": "excluded manual",
                      "spline_s": "spline s",}
    
    type: str = "auto"
    delta_threshold: float = 0.02
    excluded_manual: list = field(default_factory=list)
    spline_s: float = 0.0005


@dataclass
class WindowsSettings(ObservableSettings):
    LEGACY_MAPPING = {"mode_of_bounds": "mode of bounds",
                      "width": "width",}  
    mode_of_bounds: str = "mins"
    width: int = 3


@dataclass
class ProfilePointsSettings(ObservableSettings):
    LEGACY_MAPPING = {"finder_groups": "finder_groups",
                      "segment": "segment",
                      "background": "background",
                      "calibration": "calibration",
                      "windows": "windows",}  
    finder_groups: FinderGroupsSettings = field(default_factory=FinderGroupsSettings)
    segment: list = field(default_factory=list)
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    windows: WindowsSettings = field(default_factory=WindowsSettings)
    def to_legacy_dict(self):
        return {"finder_groups": self.finder_groups.to_legacy_dict(),
                "segment": list(self.segment),
                "background": self.background.to_legacy_dict(),
                "calibration": self.calibration.to_legacy_dict(),
                "windows": self.windows.to_legacy_dict()}

    @classmethod
    def from_legacy_dict(cls, d):
        obj = cls()
        obj.finder_groups = FinderGroupsSettings.from_legacy_dict(d.get("finder_groups", {}))
        obj.segment = d.get("segment", [])
        obj.background = BackgroundSettings.from_legacy_dict(d.get("background", {}))
        obj.calibration = CalibrationSettings.from_legacy_dict(d.get("calibration", {}))
        obj.windows = WindowsSettings.from_legacy_dict(d.get("windows", {}))
        return obj