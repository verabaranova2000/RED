
from typing import Callable, Optional


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

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

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

