from typing import Tuple

# ====== FORMAT CONFIG (Константы форматирования) для заголовка шага ======
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

# ====== Цвета и фон (ansi) ======
LIGHTGRAY_BG = "\x1b[48;5;254m"
LIGHTBLUE_BG = "\x1b[48;5;153m"
LIGHT_GREEN_BG = "\x1b[48;5;194m"
LIGHT_RED_BG = "\x1b[48;5;224m"
RESET_ALL = "\x1b[0m"
BOLD = "\x1b[1m"
BOLD_OFF = "\x1b[22m"

# ====== Цвет шрифта ======
GREEN = "\x1b[32m"
RED   = "\x1b[31m"
YELLOW= "\x1b[33m"


# ====== Отстут для шага "strategy") ======
INDENT_UNIT = 4
def make_indent(depth: int) -> str:
    return " " * (depth * INDENT_UNIT)


# ====== Формат header для шага "fit"======
def format_step_header(step_path: str, name: str, n_params: int, segment: Tuple[float,float], depth: int) -> str:
    indent = make_indent(depth)
    s_val, e_val = segment
    step_part = f"▶ [{step_path}]"
    name_part = f"{BOLD}{name.upper():<{STEP_NAME_WIDTH}}{BOLD_OFF}"
    param_part = f"({n_params:>{PARAM_COUNT_WIDTH-2}})"
    segment_part = f"{s_val:.2f}–{e_val:.2f}°"
    segment_part = f"{str(s_val):<{SEGMENT_WIDTH}}"
    return f"{indent}{LIGHTGRAY_BG}{step_part} {name_part} {param_part}{RESET_ALL} | {segment_part:<{SEGMENT_WIDTH}} |"


# ====== Формат header для шага "strategy"======
STEP_HEADER_WIDTH = 0*len("▶ [007.001] ") + STEP_NAME_WIDTH + 4 + PARAM_COUNT_WIDTH-2 + 2*len(' | ') + SEGMENT_WIDTH + RP_WIDTH

def format_cycle_header(step_path: str, depth: int, kind: str = "cycle", idx: int = None, total: int = None, label: str = "") -> str:
    """
    Форматирование строк для "Cycle X/Y", или блока стратегии
    """ 
    width = len(f"▶ [{step_path}.000] ") + STEP_HEADER_WIDTH
    indent = make_indent(depth)

    if kind == "cycle":
        if idx is None or total is None:
            step_label = "↻ Cycle"
        else:
            step_label = f"↻ Cycle {idx}/{total}"
        padded_label = f"{step_label:<{width}}"
        return f"{indent}{RED}{BOLD}{padded_label}{BOLD_OFF}{RESET_ALL}"

    elif kind == "strategy":
        step_label = f"▶ {label}"   # блок стратегии, например "CYCLE BLOCK ×5"
        padded_label = f"{step_label:<{width + len(make_indent(depth+1))}}"
        return f"{indent}{LIGHTGRAY_BG}{BOLD}{padded_label}{BOLD_OFF}{RESET_ALL}"

    else:
        raise ValueError("Unknown kind for cycle line")