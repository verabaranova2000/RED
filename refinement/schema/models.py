#@title `StepModel`

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# ------------- конфиг: допустимые хук-имена и типы шага -------------
# ------------- (константы, из которых валидаторы потом проверяют корректность)
ALLOWED_STEP_TYPES = {"fit", "block", "noop"}
ALLOWED_HOOKS = {"fix_all_except", "free_params", "save_snapshot", "save_plot", "report_delta", "noop"}
ALLOWED_COND_NAMES = {"Rp", "chisqr"}

"""
Это — валидатор схемы, который превращает YAML стратегию в строго проверенную модель.
По сути: он гарантирует, что пользователь не сможет написать чушь.
"""

# ---------------- модели схемы -----------------------------------

class StepModel(BaseModel):
    """
    Модель одного шага refinement.

    Используется для строгой проверки структуры YAML-конфигурации шага.
    Валидаторы обеспечивают:
      - корректность списка параметров для шага типа 'fit';
      - правильность диапазона сегмента и индексов;
      - допустимость хуков pre/post;
      - корректность выражения условия cond;
      - согласованность вложенной структуры для шага типа 'block'.

    Attributes
    ----------
    step_id : str
        Уникальный идентификатор шага.
    type : Literal['fit', 'block', 'noop']
        Тип шага: 
          - 'fit' — один шаг уточнения параметров;
          - 'block' — контейнер шагов (вложенная структура);
          - 'noop' — пустой шаг.
    label : str, optional
        Описание шага.
    params : list of str, optional
        Список параметров (только для 'fit').
    segment : list of float or None, optional
        Диапазон углов [start, end].
    segment_idx : list of int or None, optional
        Диапазон индексов [start_idx, end_idx].
    pre : list of str, optional
        Список хуков до выполнения шага.
    post : list of str, optional
        Список хуков после выполнения шага.
    repeat : int, default=1
        Количество повторов шага.
    cond : str, optional
        Выражение условия выполнения шага.
    steps : list of StepModel, optional
        Список вложенных шагов для шага типа 'block'.
    """
    step_id:     str = Field(..., min_length=1)          # обязательно, минимум 1 символ.
    type:        Literal['fit', 'block', 'noop']         # fit   → один шаг, block → контейнер шагов, noop  → пустой шаг 
    label:       Optional[str] = None                    # описание
    params:      Optional[List[str]] = None              # список параметров (только для fit)
    segment:     Optional[List[Optional[float]]] = None  # диапазон углов [startθ, endθ]
    segment_idx: Optional[List[Optional[int]]] = None    # диапазон индексов [i, j] в two_theta
    pre:         Optional[List[str]] = None              # хуки до шага
    post:        Optional[List[str]] = None              # хуки до и после шага
    repeat:      int = Field(1, ge=1)                    # сколько раз повторять (по умолчанию 1)
    cond:        Optional[str] = None                    # условие
    steps:       Optional[List["StepModel"]] = None      # класс ссылается сам на себя (рекурсивная структура)

    # -------- params ----------------------------------------------------
    @field_validator('params', mode='before')
    def params_must_be_list_for_fit(cls, v, info):
        """
        Проверка поля 'params'.

        Для шага type='fit' поле 'params' обязательно и должно быть непустым списком.
        Для всех остальных типов шагов поле игнорируется.
        """
        # info — это ValidationInfo
        step_type = info.data.get('type') 
        if step_type == 'fit':
            if not v or not isinstance(v, list):
                raise ValueError("для шага type='fit' поле 'params' должно быть непустым списком")
        return v

    # -------- segment ----------------------------------------------------
    @field_validator('segment', mode='before')
    def validate_segment(cls, v):
        """
        Проверка диапазона сегмента.

        Поле должно быть списком из двух чисел [start, end] или None.
        Допустимы float, int или None.
        """
        if v is None:
            return v
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError("поле 'segment' должно быть списком из двух элементов [start, end] или None")
        # float или int допускаются, None тоже
        for x in v:
            if x is not None and not isinstance(x, (int, float)):
                raise ValueError("элементы 'segment' должны быть int, float или None")
        return v


    @field_validator('segment_idx', mode='before')
    def validate_segment_idx(cls, v):
        """
        Проверка индексов сегмента.

        Поле должно быть списком из двух целых чисел [start_idx, end_idx] или None.
        """
        if v is None:
            return v
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError("поле 'segment_idx' должно быть списком из двух элементов [start_idx, end_idx] или None")
        for x in v:
            if x is not None and not isinstance(x, int):
                raise ValueError("элементы 'segment_idx' должны быть int или None")
        return v


    # -------- pre/post hooks --------------------------------------------
    @field_validator('pre', 'post', mode='before')
    def validate_hooks(cls, v):
        """
        Проверка списка хуков.

        Поля `pre` и `post` должны содержать список имён разрешённых хуков.
        Таким образом, YAML не сможет вызвать несуществующий хук.
        """
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("поля 'pre' и 'post' должны содержать список имён хуков")
        for name in v:
            if name not in ALLOWED_HOOKS:
                raise ValueError(f"хук '{name}' не входит в список допустимых: {ALLOWED_HOOKS}")
        return v

    # -------- cond expression -------------------------------------------
    @field_validator('cond')
    def validate_cond_expr(cls, v):
        """
        Проверка выражения условия выполнения шага.

        Условие может содержать только допустимые символы и должно ссылаться
        хотя бы на одну разрешённую метрику (например, `Rp` или `chisqr`).

        Пример допустимого выражения:
            cond: "Rp < 0.05 and chisqr < 1.5"
        """
        if v is None:
            return v
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ <>!=().+-*/")
        if any(ch not in allowed_chars for ch in v):
            raise ValueError("выражение 'cond' содержит недопустимые символы")
        if not any(name in v for name in ALLOWED_COND_NAMES):
            raise ValueError("выражение 'cond' должно содержать хотя бы одну допустимую метрику: " + ", ".join(ALLOWED_COND_NAMES))
        return v

    # -------- block logic --------------------------------------------
    @model_validator(mode='after')
    def validate_block_structure(self):
        """
        Проверка согласованности структуры шага.

        Правила:
        - шаг типа `block` должен содержать поле `steps` со списком вложенных шагов;
        - для шагов других типов поле `steps` не допускается.
        """
        if self.type == "block" and self.steps is None:
            raise ValueError("для шага type='block' необходимо указать поле 'steps' со списком вложенных шагов")

        if self.type != "block" and self.steps is not None:
            raise ValueError("поле 'steps' допускается только для шага type='block'")
        return self


StepModel.model_rebuild()   # если на Pydantic v2




class SchemaModel(BaseModel):
    """
    Полная схема уточнения: набор всех шагов.

    Attributes
    ----------
    name : str, optional
        Имя схемы.
    steps : list of StepModel
        Список шагов, которые выполняются в рамках схемы.

    Validators
    ----------
    Проверка уникальности step_id среди всех шагов.
    """
    name: Optional[str]
    steps: List[StepModel]

    @field_validator('steps')
    def unique_step_ids(cls, v):
        """
        Проверка уникальности идентификаторов шагов.

        Все step_id в списке шагов должны быть уникальными,
        иначе выбрасывается ошибка.
        """
        ids = [s.step_id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Идентификаторы шагов step_id должны быть уникальными в рамках схемы")
        return v
  

"""
💡 Маленький архитектурный бонус (1 мысль на будущее):

Сейчас StepModel почти готов к тому, чтобы YAML-стратегию можно было автоматически 
визуализировать как дерево шагов (Graphviz или ASCII).
Ты уже фактически построила чистый tree-AST для refinement workflow, 
и это редкий случай, когда научный код сразу получается архитектурно аккуратным.
"""
