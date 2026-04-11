from typing import Optional, Literal
from dataclasses import dataclass
from utils.observable import ObservableSettings

@dataclass
class AtomSettings(ObservableSettings):
    """
    Настройки атома.
    Описывают допустимое состояние без логики.
    """
    LEGACY_MAPPING = {"ADP_parameters": "ADP parameters",
                      "order": "order",
                      "fe_from": "fe from",}
    ADP_parameters: Literal["isotropic",
                            "harmonic (anisotropic)",
                            "anharmonic"] = "isotropic"
    order: Optional[Literal[3, 4, 5, 6]] = None
    fe_from: Literal["it4322",
                     "Mott-Bethe",
                     "aspherical"] = "it4322"

    #classmethod
    #def from_legacy_dict(cls, d):
    #    """
    #    Создаёт AtomSettings из старого словаря setting.
    #    """
    #    obj = cls()
    #    obj.ADP_parameters = d.get("ADP parameters", "isotropic")
    #    obj.order = d.get("order", None)
    #    obj.fe_from = d.get("fe from", "it4322")
    #    return obj  

    #def to_legacy_dict(self):
    #    return {"ADP parameters": self.ADP_parameters,
    #            "order": self.order,
    #            "fe from": self.fe_from}                          