## Reactive Settings System

Система превращает обычные dict-настройки в реактивное дерево состояния.

Любое изменение значения:
- автоматически оборачивается в реактивный контейнер (list/dict)
- распространяется по дереву через path
- вызывает callback изменения
- может запускать пересчёт зависимых эффектов


## Layers (Архитектурные слои)

1. Settings layer
   - dataclass-структуры
   - хранение состояния

2. Reactive layer (ObservableSettings)
   - перехват __setattr__
   - wrap list/dict
   - bind вложенных объектов
   - генерация path

3. Effects layer (внутри класса Phase / Atom / ProfilePoints)
   - _effects map
   - пересчёт зависимостей
   - batch-trigger после загрузки


## Lifecycle of a setting change (Жизненный цикл значения)

1. User assigns value:
   settings.form = "Gaussian"

2. __setattr__ is triggered

3. Value is wrapped:
   list → ObservableList
   dict → ObservableDict

4. Value is stored in object

5. Callback is triggered:
   on_change("form") or "blackman.mode"

6. If needed:
   effects system triggers recomputation


## Initialization rule (Правило инициализации)

При загрузке settings:

1. создаётся объект settings
2. выполняется bind()
3. выполняется _trigger_all_effects()

Важно:
- все значения уже установлены до trigger
- нет промежуточных recalculation
- система переходит сразу в консистентное состояние


## Migration strategy (Как встраивать в старые классы)

Старые классы могут работать в двух режимах:

### 1. Legacy mode
- обычные dataclass settings
- без реактивности

### 2. Reactive mode
- settings = ObservableSettings
- bind(callback)
- __setattr__ перехватывает изменения

Переход:
- можно заменить settings поэтапно
- логика классов не требует переписывания сразу


## Known constraints (Ограничения системы)

- __setattr__ становится центральной точкой логики
- порядок bind → trigger важен
- циклические зависимости в _effects не обрабатываются автоматически