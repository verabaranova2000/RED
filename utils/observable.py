
from typing import Callable, Optional
from contextlib import contextmanager
from typing import ClassVar
import numpy as np

"""
Наблюдаемые настройки и реактивные контейнеры.

Модуль содержит базовый класс для dataclass-ов настроек и реактивные
обёртки над стандартными контейнерами Python. Эти классы используются
для отслеживания изменений состояния модели без ручного разбора вложенных
словарей и списков и уведомления объектов (Phase, Atoms и др.).

Основная идея:
- присваивание атрибута в настройках порождает событие;
- изменение содержимого списка или словаря тоже порождает событие;
- объект более высокого уровня (Phase, ProfilePoints, Atom) сам решает,
  что нужно пересчитать после изменения состояния.
"""

class ObservableList(list):
    """
    Реактивная обёртка над list.

    Отслеживает мутации списка и вызывает callback при изменениях,
    связанных с содержимым:
    append, remove, extend, присваивание по индексу.

    Используется как контейнер для списочных полей настроек,
    чтобы изменения внутри списка также инициировали обновление модели.
    """    
    def __init__(self, initial, notify, path=""):
        super().__init__(initial)
        self._notify = notify
        self._path = path

    def append(self, value):
        super().append(value)
        print(f"[OBS-LIST] append → {self._path}: {value}")
        self._notify(self._path)

    def remove(self, value):
        super().remove(value)
        print(f"[OBS-LIST] remove → {self._path}: {value}")
        self._notify(self._path)

    def __setitem__(self, idx, value):
        super().__setitem__(idx, value)
        print(f"[OBS-LIST] setitem → {self._path}[{idx}]")
        self._notify(self._path)

    def extend(self, values):
        super().extend(values)
        print(f"[OBS-LIST] extend → {self._path}: {values}")
        self._notify(self._path)
    


class ObservableDict(dict):
    """
    Реактивная обёртка над dict.

    Отслеживает изменение элементов словаря через операции присваивания
    и удаления по ключу, после чего вызывает callback с путём объекта.
    Используется для вложенных словарных настроек, которые должны
    инициировать пересчёт модели при модификации.
    """    
    def __init__(self, initial, notify, path=""):
        super().__init__(initial)
        self._notify = notify
        self._path = path

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        print(f"[OBS-DICT] set {self._path}[{key}] = {value}")
        self._notify(self._path)

    def __delitem__(self, key):
        super().__delitem__(key)
        print(f"[OBS-DICT] delete {self._path}[{key}]")
        self._notify(self._path)    







class ObservableSettings:
    """
    Базовый класс для наблюдаемых объектов настроек.
    Его задача — наблюдаемость и маршрутизация событий изменения состояния.

    Класс предназначен для использования вместе с dataclass-настройками.
    Он обеспечивает:
    - уведомление о присваивании атрибутов;
    - рекурсивную привязку вложенных ObservableSettings;
    - автоматическую обёртку обычных list/dict в ObservableList/ObservableDict;
    - передачу пути изменённого поля в callback.
    """    
    _on_change: Optional[Callable[[str], None]] = None
    _path: str = ""
    LEGACY_MAPPING: ClassVar[dict] = {}

    def _ensure_suspend_flag(self):
        """ Ленивая инициализация (вместро def __init__) """
        if "_suspend_notify" not in self.__dict__:
            object.__setattr__(self, "_suspend_notify", 0)

    @contextmanager
    def suspend_notify(self):
        """ Приостановить уведомление для присваивания сразу нескольких полей (patch) """
        self._ensure_suspend_flag()
        self._suspend_notify += 1
        try:
            yield
        finally:
            self._suspend_notify -= 1


    def bind(self, on_change: Callable[[str], None], path: str = ""):
        """
        Привязывает объект настроек к обработчику изменений.

        Параметры
        ----------
        on_change : Callable[[str], None]
            Колбэк, вызываемый при изменении любого наблюдаемого поля.
        path : str, optional
            Путь текущего объекта в дереве настроек. Используется для формирования
            полного имени изменённого поля, например "blackman.mode".

        Возвращает
        ----------
        self : ObservableSettings
            Тот же объект, но уже связанный с callback'ом.       

        Механизм
        ---------
        Phase получает уведомление через:
        >>> self.settings.bind(self._on_settings_changed)
        >>> # Теперь Phase подписан на изменения.
        """
        print(f"[1] bind: привязываем settings к Phase (path='{path}')")
        object.__setattr__(self, "_on_change", on_change)
        object.__setattr__(self, "_path", path)
        for name in getattr(self, "__dataclass_fields__", {}):
            value = getattr(self, name)
            full_path = f"{path}.{name}" if path else name
            if isinstance(value, ObservableSettings):
                value.bind(on_change, full_path)
            elif isinstance(value, ObservableList):
                value._notify = on_change
                value._path = full_path
            elif isinstance(value, ObservableDict):
                value._notify = on_change
                value._path = full_path        
        return self

    def _coerce_value(self, name, value):
        """
        Нормализует значение перед сохранением атрибута.

        Параметры
        ----------
        name : str
            Имя изменяемого поля. Позволяет применять разные правила
            нормализации для разных атрибутов в одном методе.
        value : Any
            Присваиваемое значение.

        Возвращает
        ----------
        Any
            Преобразованное (валидированное) значение.

        Назначение
        ----------
        Является расширяемым hook-методом: базовый класс не содержит логики,
        а наследники переопределяют его для валидации и ограничения значений
        конкретных полей.
        """        
        #print(f"[COERCE from ObservableSettings] value → {value}")
        return value    
    
    def _wrap_value(self, name, value):
        """
        Преобразует присваиваемое значение в реактивный тип при необходимости.

        Правила:
        - list → ObservableList
        - dict → ObservableDict
        - ObservableSettings → без изменений
        - остальные типы → без изменений

        Важно:
        - используется внутри __setattr__ ДО сохранения значения
        - сохраняет текущий _path для построения полного пути поля
        """
        path = f"{self._path}.{name}" if self._path else name
        if isinstance(value, list):
            return ObservableList(value, self._on_change, path)
        if isinstance(value, dict):
            return ObservableDict(value, self._on_change, path)
        return value


    def __setattr__(self, name, value):
        """
        Перехватывает присваивание атрибутов и реализует реактивное обновление.

        Логика:

        1. Приватные поля ("_...") устанавливаются напрямую без уведомлений.
        2. Значение преобразуется через _wrap_value (list/dict → reactive wrappers).
        3. Значение сохраняется в объект.
        4. Если установлен callback (_on_change), выполняется:
        - рекурсивная привязка вложенных ObservableSettings
        - уведомление об изменении через полный путь поля

        Формат уведомления:
            "blackman.mode", "form", "a.b.c"

        Гарантия:
        - callback вызывается только после фактического сохранения значения
        - вложенные структуры автоматически становятся наблюдаемыми
        """    
        print(f"[2] __setattr__: пытаемся установить {name} = {value}")

        if getattr(self, "_suspend_notify", 0) > 0:
            object.__setattr__(self, name, value)
            return

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # print('📌Вызов self._coerce_value(name, value) в ObservableSettings')
        value = self._coerce_value(name, value)       # 0. базовый класс вызывает один метод, а наследник подменяет только то, что ему нужно.
        value = self._wrap_value(name, value)         # 1. превращаем list/dict в Observable
        object.__setattr__(self, name, value)         # 2. сохраняем

        on_change = getattr(self, "_on_change", None) # 3. биндим вложенные settings
        if on_change is not None:
            if isinstance(value, ObservableSettings):
                value.bind(on_change, f"{self._path}.{name}" if self._path else name)
            # 4. уведомляем Phase
            full_path = f"{self._path}.{name}" if self._path else name
            print(f"[3] notify: поле изменилось → '{full_path}'")
            on_change(full_path)

    def to_legacy_dict(self):
        """
        Преобразует настройки фазы в словарь старого формата.

        Используется для сохранения проекта без изменения структуры файла.

        Возвращает
        ----------
        dict
            Словарь, полностью совместимый с прежним форматом setting.
        """            
        d = {}
        for attr, legacy_name in self.LEGACY_MAPPING.items():
            value = getattr(self, attr)
            if value is not None:
                d[legacy_name] = value
        return d

    #def to_legacy_dict(self):
    #    """
    #    Преобразует настройки фазы в словарь старого формата.

    #    Используется для сохранения проекта без изменения структуры файла.

    #    Возвращает
    #    ----------
    #    dict
    #        Словарь, полностью совместимый с прежним форматом setting.
    #    """          
    #    def convert(value):                        # ✔ делает поведение единообразным
    #        if hasattr(value, "to_legacy_dict"):   # ✔ фиксит главную проблему — вложенные dataclass
    #            return value.to_legacy_dict()
    #        if isinstance(value, tuple):           # ✔ фиксит uvar и подобные
    #            return list(value)
    #        if isinstance(value, np.ndarray):
    #            return value.tolist()
    #        if isinstance(value, list):
    #            return [convert(v) for v in value]
    #        if isinstance(value, dict):
    #            return {k: convert(v) for k, v in value.items()}
    #        return value

    #    d = {}
    #    for attr, legacy_name in self.LEGACY_MAPPING.items():
    #        value = getattr(self, attr)
    #        if value is not None:
    #            d[legacy_name] = convert(value)
    #    return d

    #classmethod
    #def from_legacy_dict(cls, d):
    #    obj = cls()
    #    reverse = {v: k for k, v in cls.LEGACY_MAPPING.items()}
    #    for legacy_name, value in d.items():
    #        if legacy_name in reverse:
    #            setattr(obj, reverse[legacy_name], value)
    #    return obj
    @classmethod     
    def from_legacy_dict(cls, d):        
        """
        Создаёт объект из словаря старого формата.

        Параметры
        ----------
        d : dict
            Словарь настроек, считанный из файла проекта, 
            вида {"mode": ..., "corrections": ...}

        Возвращает
        ----------
        [Phase / Atom / ProfilePoints / Background / ... ]Settings

        Примечания
        ----------
        - Отсутствующие поля заполняются значениями по умолчанию.
        - Вложенные настройки (Blackman, internal) создаются рекурсивно.        
        """            
        obj = cls()
        with obj.suspend_notify():
            reverse = {v: k for k, v in cls.LEGACY_MAPPING.items()}
            for legacy_name, value in d.items():
                if legacy_name in reverse:
                    setattr(obj, reverse[legacy_name], value)
        return obj    


    def snapshot(self):
        """
        Возвращает снимок состояния (dict) с рекурсивным преобразованием вложенных dataclass.
        
        Example
        ------
        {
            "finder_groups": {
                "window_length": 20,
                "polyorder": 3,
                "prominence": 5,
            },
            "segment": [],
            "background": {
                ...
            },
            ...
        }
        """
        def convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(self)


# ========= 🥇 Mixin =============

class ReactiveMixin:
    """
    Mixin, реализующий реактивную обработку настроек.

    Требует от класса:
    - self._effects : dict[path -> tuple[action]]
    - self._actions : dict[action_name -> callable]

    Используется в Phase, Atom и других объектах,
    которые реагируют на изменения settings.
    """
    SETTINGS_CLS = None   # переопределяется в Phase / Atom

    def _on_settings_changed(self, path: str):
        print(f"[4] {self.__class__.__name__} получил уведомление: изменилось '{path}'")
        actions = self._effects.get(path, ())
        if not actions:
            print("[5] → нет действий для этого поля")
            return
        for action_name in actions:
            print(f"[6] → запускаем действие: {action_name}")
            self._actions[action_name]()

    def _trigger_all_effects(self):
        """
        Полная синхронизация Phase после загрузки settings.

        Принцип:
        - Проходим по карте _effects
        - Для каждого action вызываем соответствующий метод Phase

        ВАЖНО:
        - Это batch-синхронизация (не реактивный одиночный триггер)
        - Выполняется ОДИН раз после загрузки settings
        - Гарантирует, что все производные структуры (profile, reflections, scales)
          пересчитаны на согласованных данных

        Почему нет лишних пересчётов:
        - settings уже полностью загружены до вызова
        - bind уже установлен
        - поэтому все зависимости актуальны на момент запуска

        Гарантия:
        - итоговое состояние Phase консистентно
        - нет промежуточных пересчётов “на полпути загрузки”
        """       
        print(f"[TRIGGER] синхронизация ({self.__class__.__name__})")
        for path in self._effects:
            print(f"[TRIGGER] → {path} зависит от {self._effects[path]}")
            self._on_settings_changed(path)

    def load_settings(self, data):
        """
        Загрузка settings из сохранённого состояния и полная синхронизация Phase.

        Порядок выполнения:

        1. from_legacy_dict(data)
          → создаётся новый PhaseSettings со всеми значениями

        2. bind(self._on_settings_changed)
          → подключается реактивная система:
            изменения settings теперь вызывают _on_settings_changed

        3. _trigger_all_effects()
          → выполняется ПОЛНАЯ синхронизация состояния Phase
            на основе уже загруженных settings

        ВАЖНО:
        - На этом этапе все settings уже полностью присвоены
        - Поэтому пересчёты происходят один раз и на актуальных данных
        - Нет “промежуточных” пересчётов во время загрузки
        - Это гарантирует согласованное состояние объекта после load_settings()

        """        
        if self.SETTINGS_CLS is None:
            raise NotImplementedError("SETTINGS_CLS is not set")

        self.settings = self.SETTINGS_CLS.from_legacy_dict(data)
        self.settings.bind(self._on_settings_changed)
        self._trigger_all_effects()