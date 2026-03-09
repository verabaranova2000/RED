from typing import Tuple

"""
Formatting utilities for refinement console output.

Модуль содержит константы и функции для форматирования строк,
используемых при выводе шагов уточнения в терминале/блокноте.

Здесь задаются:
- ширины колонок
- ANSI-цвета
- функции форматирования заголовков шагов и циклов шагов

Модуль используется в классе RefinementSession,
который отвечает за отображение хода refinement в виде
структурированного текстового интерфейса.


Пример вывода:
---------------
04:47:59 | INFO    | RefinementStep  | ▶ CYCLE BLOCK ×5                                                | 
04:47:59 | INFO    | RefinementStep  |     ↻ Cycle 1/5                                                 | 
04:48:27 | INFO    | RefinementStep  |     ▶ [007.001] SCALE        ( 1) | 0.32–5.74° | Rp 76.961%  ⬈  | 
                                                Param                    Value          Δ%
                                                Phase1_scale          0.029323      97.068
                                                ──────────────────────────────────────────
04:48:27 | INFO    | RefinementStep  |     ▶ [007.002] BACKGROUND   (14) | 0.32–5.74° | 

"""



# ====== Header formatting configuration ======
# Константы, определяющие ширины колонок для строк заголовков шагов refinement.

# Используется для вычисления ширины колонки имени шага, чтобы все заголовки выравнивались.
POSSIBLE_STEPS = ["SCALE", "BACKGROUND", "PROFILE", "LATTICE", "ZERO"]

STEP_NAME_WIDTH = max(len(s) for s in POSSIBLE_STEPS) + 2
STEP_INDEX_WIDTH = 4      # [01]
PARAM_COUNT_WIDTH = 4     # (12)
SEGMENT_WIDTH = 9         # 0.32–5.74°
RP_WIDTH = 14             #  Rp 19.875% ⬊

PARAM_COL_WIDTH = 15
VALUE_COL_WIDTH = 15
DELTA_COL_WIDTH = 12

TABLE_WIDTH = PARAM_COL_WIDTH + VALUE_COL_WIDTH + DELTA_COL_WIDTH

SEPARATOR = "─" * TABLE_WIDTH

# ====== Цвет фона (ansi) ======
RESET_ALL = "\x1b[0m"
LIGHTGRAY_BG = "\x1b[48;5;254m"
LIGHTBLUE_BG = "\x1b[48;5;153m"
LIGHT_GREEN_BG = "\x1b[48;5;194m"
LIGHT_RED_BG = "\x1b[48;5;224m"

# ====== Цвет шрифта ======
RED    = "\x1b[31m"
GREEN  = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE   = "\x1b[34m"

# ====== Стиль шрифта ======
BOLD = "\x1b[1m"
BOLD_OFF = "\x1b[22m"


# ====== Отстут для шага "strategy" ======
INDENT_UNIT = 4
def make_indent(depth: int) -> str:
    """
    Создаёт отступ для вложенных шагов refinement.

    Parameters
    ----------
    depth : int
        Уровень вложенности шага (шаг "strategy" → cycle → step).

    Returns
    -------
    str
        Строка из пробелов, используемая как отступ слева.
    """
    return " " * (depth * INDENT_UNIT)


# ====== Формат header для шага "fit"======
def format_step_header(step_path: str, name: str, n_params: int, segment: Tuple[float,float], depth: int) -> str:
    """
    Формирует строку заголовка для шага refinement.

    Пример вывода:

        ▶ [007.001] SCALE   (12) | 0.32–5.74° |

    Parameters
    ----------
    step_path : str
        Путь шага в дереве refinement (например "007.001").

    name : str
        Имя шага (SCALE, BACKGROUND, PROFILE и т.п.).

    n_params : int
        Количество параметров, оптимизируемых на данном шаге.

    segment : tuple[float, float]
        Диапазон 2θ, используемый в refinement.

    depth : int
        Уровень вложенности шага (используется для отступа).

    Returns
    -------
    str
        Отформатированная строка заголовка шага.
    """    
    indent = make_indent(depth)
    s_val, e_val = segment
    step_part = f"▶ [{step_path}]"       
    name_part = f"{BOLD}{name.upper():<{STEP_NAME_WIDTH}}{BOLD_OFF}"
    param_part = f"({n_params:>{PARAM_COUNT_WIDTH-2}})"
    segment_part = f"{s_val:.2f}–{e_val:.2f}°"
    return f"{indent}{LIGHTGRAY_BG}{step_part} {name_part} {param_part}{RESET_ALL} | {segment_part:<{SEGMENT_WIDTH}} |"


# ====== Формат header для шага "block"======
STEP_HEADER_WIDTH = 0*len("▶ [007.001] ") + STEP_NAME_WIDTH + 4 + PARAM_COUNT_WIDTH-2 + 2*len(' | ') + SEGMENT_WIDTH + RP_WIDTH

def format_cycle_header(step_path: str, depth: int, kind: str = "cycle", idx: int = None, total: int = None, label: str = "") -> str:
    """
    Формирует строку заголовка для цикла refinement
    или блока стратегии.

    Поддерживаемые типы:
    - strategy  → "▶ CYCLE BLOCK ×5"
    - cycle     → "↻ Cycle 2/5"

    Parameters
    ----------
    step_path : str
        Путь текущего шага refinement.

    depth : int
        Уровень вложенности (для отступа).

    kind : str
        Тип заголовка: "cycle" или "strategy".

    idx : int, optional
        Номер текущего цикла.

    total : int, optional
        Общее количество циклов.

    label : str
        Подпись для блока стратегии.

    Returns
    -------
    str
        Отформатированная строка заголовка.
    """
    width = len(f"▶ [{step_path}.000] ") + STEP_HEADER_WIDTH
    indent = make_indent(depth)
    
    if kind == "cycle":
        step_label = f"↻ Cycle {idx}/{total}"
        padded_label = f"{step_label:<{width}}"
        #return f"{indent}{LIGHTGRAY_BG}{RED}{BOLD}{padded_label}{BOLD_OFF}{RESET}"   # с серым фоном
        return f"{indent}{RED}{BOLD}{padded_label}{BOLD_OFF}{RESET_ALL}"
    
    elif kind == "block":
        step_label = f"▶ {label}"  # блок стратегии, например "cycle_block ×5"
        padded_label = f"{step_label:<{width + len(make_indent(depth+1))}}"
        return f"{indent}{LIGHTGRAY_BG}{BOLD}{padded_label}{BOLD_OFF}{RESET_ALL}"
    
    else:
        raise ValueError("Unknown kind for cycle line")