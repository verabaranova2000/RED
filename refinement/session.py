#@title RefinementSession

from IPython.display import display, HTML
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


from utils.logging_setup import logger, BASE_FORMAT
from refinement.logutils.live_header import LiveHeader
from refinement.logutils.formatting import (
    format_step_header, format_cycle_header,
    BOLD, BLUE, 
    LIGHT_GREEN_BG, LIGHT_RED_BG, RESET_ALL,
    PARAM_COL_WIDTH, VALUE_COL_WIDTH, DELTA_COL_WIDTH, RP_WIDTH,
    SEPARATOR
)
from .param_utils import parse_background_param, format_value, format_dperc


"""
Refinement session controller.

Модуль содержит класс RefinementSession, который управляет выводом
логов шагов refinement, хранит историю выполнения и формирует
сводный отчёт по результатам.

Класс используется во время выполнения процесса refinement
(см. execution.py) и отвечает за:

- вывод строк-заголовков шагов и циклов шагов 
  (в виде обновляемой строки-заголовка LiveHeader, затем лога loguru)
- отслеживание изменения метрики Rp
- вывод таблиц с результатами уточнения параметров
- накопление истории шагов refinement

Форматирование строк и механизмы live-вывода реализованы
в модуле refinement.logutils.
"""


class RefinementSession:
    """
    Контроллер логирования процесса refinement.

    Класс управляет выводом информации о шагах уточнения,
    отслеживает изменение метрики Rp и сохраняет историю
    выполнения refinement.

    Methods
    -------
    start_step(...)
        Начать логирование шага refinement.

    start_block(...)
        Вывести заголовок блока стратегии.

    start_cycle(...)
        Вывести заголовок цикла refinement.

    report_Rp(...)
        Завершить шаг и вывести значение Rp.

    report_parameters(...)
        Вывести таблицу обновлённых параметров.

    report_background_group(...)
        Вывести группу параметров фона.

    save_step(...)
        Сохранить информацию о шаге в историю.

    summary()
        Вывести итоговую сводку refinement.

    
    Attributes
    ----------
    history : list of dict
        История шагов refinement.

    prev_Rp : float or None
        Значение Rp на предыдущем шаге.

    Notes
    -----
    Класс не выполняет вычисления refinement. Он используется
    совместно с модулем execution, который отвечает за запуск
    шагов уточнения.

    Examples
    -------
    >>> from refinement import RefinementSession
    
    Использование через внешний API:

    >>> session = RefinementSession()
    >>> execute_strategy(strategy_steps, pr, out_prev=None, session=session)
    >>> session.summary()

    Прямое использование методов класса (обычно не требуется для пользователя):

    >>> session = RefinementSession()
    >>> session.start_step("SCALE", (0.3, 5.7), n_params=2, depth=0, step_path="001")
    >>> session.report_Rp(12.345)
    >>> session.summary()
    """
    def __init__(self, pylogger="RefinementStep"):
        self.pylogger = pylogger
        self.logger = logger.bind(pylogger=pylogger)
        self.history = []
        self.prev_Rp = None
        self.step_number = 0
        self.current_cycle = None
        self.live = None
        self.log_indent = None

    def _get_log_indent(self):
        prefix = self.live._build_prefix()
        return " " * len(prefix + " | " + " ▶  " + "[001]")

    # ---------- STEP HEADER ----------
    def start_step(self, name, segment, n_params, depth, step_path):
        """
        Начать логирование нового шага refinement.

        Печатает заголовок шага в режиме live (без перевода строки).
        После завершения шага строка будет заменена финальным
        лог-сообщением после вызова метода report_Rp().

        Parameters
        ----------
        name : str
            Имя шага (например SCALE, BACKGROUND).

        segment : tuple[float, float]
            Диапазон 2θ, используемый в шаге refinement.

        n_params : int
            Количество уточняемых параметров.

        depth : int
            Уровень вложенности шага.

        step_path : str
            Идентификатор шага в дереве refinement (напр., 007.001).
        """
        self.step_number += 1
        header_text = format_step_header(step_path, name, n_params, segment, depth)
        # --- создаём live ---
        self.live = LiveHeader(pylogger=self.pylogger, logger=self.logger, base_format=BASE_FORMAT)
        self.live.start(header_text)
        self.log_indent = self._get_log_indent()
        # сохраняем для использования при finish
        self.current_header = header_text


    # ---------- BLOCK START ----------
    def start_block(self, label, step_path, repeat, depth):
        line = format_cycle_header(step_path=step_path, depth=depth, kind="block", label=f"{label} ×{repeat}")
        self.logger.info(line)

    # ---------- CYCLE START ----------
    def start_cycle(self, label, step_path, idx, total, depth):
        self.current_cycle = idx
        line = format_cycle_header(step_path=step_path, depth=depth, kind="cycle", idx=idx, total=total)
        self.logger.info(line)


    # ---------- REPORT METRICS ----------
    def report_Rp(self, Rp):
        """
        Завершить текущий шаг и вывести значение Rp.

        Метод заменяет временную строку live-заголовка
        финальным лог-сообщением, содержащим метрику Rp.
        Также отображается направление изменения Rp
        относительно предыдущего шага (arrows:    ⬊⬈➘➚↗↘⬀⬂➴➶➷➹→).

        Parameters
        ----------
        Rp : float
            Значение R-фактора после завершения шага refinement.
        """
        if self.prev_Rp is None:
            text = f"Rp {Rp:.3f}%"
            final_suffix = f"{text:<{RP_WIDTH}}"
        elif self.prev_Rp is not None:
            if Rp < self.prev_Rp:
                text = f"Rp {Rp:.3f}% ⬊"
                text = f"{text:<{RP_WIDTH}}"
                final_suffix = f"{LIGHT_GREEN_BG}{text}{RESET_ALL}"
            elif Rp > self.prev_Rp:
                text = f"Rp {Rp:.3f}% ⬈"
                text = f"{text:<{RP_WIDTH}}"
                final_suffix = f"{LIGHT_RED_BG}{text}{RESET_ALL}"
            else:
                text = f"Rp {Rp:.3f}%"
                final_suffix = f"{text:<{RP_WIDTH}}"
        # завершаем live, выводим финальный лог через loguru
        if self.live is not None:
            self.live.finish(self.current_header, final_suffix)
            self.live = None
        else:
            self.logger.info(final_suffix)
        self.prev_Rp = Rp
        self.current_Rp = Rp



    # ---------- NORMAL PARAM TABLE ----------
    def report_parameters(self, param_data):
        """
        Вывести таблицу обновлённых параметров.

        Parameters
        ----------
        param_data : dict
            Словарь вида:

            {
                "scale": (value, delta_percent),
                ...
            }

        Notes
        -----
        Таблица выводится без стандартного форматирования логгера (напр., "HH:mm:ss | INFO | ...")
        (raw output), чтобы сохранить чистый вывод. Учтен сдвиг для выравнивания колонок.
        """       
        header = (f"{'Param':<{PARAM_COL_WIDTH}}"
                  f"{'Value':>{VALUE_COL_WIDTH}}"
                  f"{'Δ%':>{DELTA_COL_WIDTH}}")
        rows = []
        for p, (val, dperc) in param_data.items():
            val_str = format_value(val, fmt=".6f")
            dperc_str = format_dperc(dperc, fmt=".3f")
            row = (f"{BLUE}{BOLD}{p:<{PARAM_COL_WIDTH}}{RESET_ALL}"
                  f"{BLUE}{val_str:>{VALUE_COL_WIDTH}}{RESET_ALL}"
                  f"{BLUE}{dperc_str:>{DELTA_COL_WIDTH}}{RESET_ALL}")
            rows.append(row)
        block = "\n".join([header] + rows + [SEPARATOR])
        indented_block = "\n".join(self.log_indent + line for line in block.split("\n")) # добавляем отступ к каждой строке таблицы
        self.logger.opt(raw=True).info(indented_block + "\n")                   # выводим без лог-форматирования


    # ---------- BACKGROUND GROUP ----------
    def report_background_group(self, param_data):       
        indent_ch = len(self.log_indent)  # число пробелов
        # --- вычислить диапазон ---
        prefix, indices = None, []
        for p in param_data:
            pref, idx = parse_background_param(p)
            if prefix is None:
                prefix = pref
            indices.append(idx)
        first_idx, last_idx = min(indices), max(indices)

        group_label = f"{prefix}[{first_idx}–{last_idx}]"
        # --- таблица ---
        rows = []
        for p, (val, dperc) in param_data.items():
            val_str = format_value(val, fmt=".6f")
            dperc_str = format_dperc(dperc, fmt=".3f")
            rows.append(
                f"<tr>"
                f"<td style='padding-right:25px; color:blue; font-weight:bold'>{p}</td>"
                f"<td style='text-align:right;padding-right:25px; color:blue'>{val_str:>{VALUE_COL_WIDTH}}</td>"
                f"<td style='text-align:right; color:blue'>{dperc_str:>{DELTA_COL_WIDTH}}</td>"
                f"</tr>")
        rows_html = "".join(rows)
        html = f"""
        <div style="margin-left:{indent_ch}ch; font-family:monospace;">
            <details>
                <summary style="cursor:pointer;">
                    {group_label:<15} updated
                </summary>
                <table style="margin-top:6px;">
                    <tr>
                        <th align="left">Param</th>
                        <th align="right">Value</th>
                        <th align="right">Δ%</th>
                    </tr>
                    {rows_html}
                </table>
            </details>
        </div>
        """
        display(HTML(html))
        # аккуратный разделитель под таблицей
        self.logger.opt(raw=True).info(self.log_indent + SEPARATOR + "\n")



    # ---------- SAVE STEP ----------
    def save_step(self, label, step_path=None, depth=None, params=None):
        """
        Сохранить информацию о выполненном шаге в историю refinement.

        Parameters
        ----------
        label : str
            Название шага.
        step_path : str, optional
            Идентификатор шага в дереве стратегии.
        depth : int, optional
            Уровень вложенности шага.
        params : list[str], optional
            Список параметров, уточняемых на шаге.
        """
        self.history.append({"step_number": self.step_number,
                             "label": label,
                             "step_path": step_path,
                             "cycle": self.current_cycle,
                             "depth": depth,
                             "params": params,
                             "timestamp": datetime.now(),
                             "Rp": self.current_Rp})

    # ---------- SUMMARY ----------
    def summary(self):
        """
        Вывести итоговую сводку refinement.

        Отображает таблицу истории шагов и график изменения
        метрики Rp.
        """
        if not self.history:
            print("История шагов уточнения пуста.")
            return
        print("═" * 40)
        print("FINAL SUMMARY")
        print("═" * 40)

        df = pd.DataFrame(self.history)
        df["params"] = df["params"].apply(lambda x: ", ".join(x) if x else "")
        df.index.name = "step"  # индекс = номер шага
        display(df)

        print(f"Final Rp: {self.history[-1]['Rp']:.3f}%")

        # График
        plt.figure()
        plt.plot(df["Rp"])
        plt.xlabel("Step")
        plt.ylabel("Rp (%)")
        plt.title("Refinement convergence")
        plt.show()