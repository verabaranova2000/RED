## Reactive Settings System

Система реализует реактивную модель настроек:

любое изменение значения автоматически:
- фиксируется,
- преобразуется (list/dict → reactive),
- передаётся как событие (path),
- обрабатывается через таблицу зависимостей.


---

## Architecture

Система состоит из трёх уровней:

### 1. Settings (данные)
- dataclass-объекты
- только хранение состояния
- без логики

### 2. Observable layer (реактивность)
- `ObservableSettings`
- перехват `__setattr__`
- обёртка list/dict → ObservableList/ObservableDict
- генерация `path` (например: `"blackman.mode"`)
- вызов callback при изменении

### 3. Effects layer (логика объекта)
- `_effects`: `path → actions`
- `_actions`: `name → method`
- обработка через `ReactiveMixin`


---

## Update pipeline

Любое изменение проходит цепочку:
```python
settings.field = value
↓
setattr
↓
wrap (если list/dict)
↓
save value
↓
notify(path)
↓
ReactiveMixin._on_settings_changed
↓
lookup in _effects
↓
execute actions
```



---

## Initialization (важно)

Правильная последовательность:

1. создать settings (из dataclass или from_legacy_dict)
2. выполнить `bind(callback)`
3. вызвать `_trigger_all_effects()`

Гарантии:
- все значения уже установлены
- пересчёт выполняется один раз
- нет промежуточных состояний


---

## Effects model

Используется декларативная схема:

```python
_effects = {
    "form": ("rebuild_profile",),
    "typeref": ("rebuild_reflections",),
}
```

- зависимости задаются как: `path → actions`
- actions независимы друг от друга
- порядок выполнения не гарантируется


---

### ReactiveMixin

Общий механизм для всех объектов (Phase, Atom и др.):

- `_on_settings_changed(path)`
- `_trigger_all_effects()`
- `_load_settings(...)`

Позволяет:

- избежать дублирования кода
- обеспечить единое поведение реактивности


---

### Migration guide

Чтобы сделать класс реактивным:

1. создать `Settings` (dataclass + ObservableSettings)
2. описать `_actions`
3. описать `_effects`
4. унаследоваться от `ReactiveMixin`

Инициализация:

```python
self.settings = Settings().bind(self._on_settings_changed)
```

Загрузка:
```python
self._load_settings(
      Settings.from_legacy_dict(data),
      self._on_settings_changed
      )
```

---

### Constraints

- логика зависит от `__setattr__`
- критичен порядок: `bind → trigger`
- `_effects` не поддерживает сложные графы зависимостей
- отсутствует управление порядком выполнения actions