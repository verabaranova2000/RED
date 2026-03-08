#@title RefinementSession

from IPython.display import display, HTML
import pandas as pd
import matplotlib.pyplot as plt


class RefinementSession:
    def __init__(self, pylogger="RefinementStep"):
        self.pylogger = pylogger
        self.logger = logger.bind(pylogger=pylogger)
        self.history = []
        self.prev_Rp = None
        self.step_index = 0
        self.live = None
        self.log_indent = None

    def _get_log_indent(self):
        prefix = self.live._build_prefix()
        return " " * len(prefix + " | " + " ▶  " + "[001]")

    # ---------- STEP HEADER ----------
    def start_step(self, name, segment, n_params, depth, step_path):
        self.step_index += 1
        header_text = format_step_header(step_path, name, n_params, segment, depth)
        # --- создаём live ---
        self.live = LiveHeader(pylogger=self.pylogger, logger=self.logger, base_format=color_to_base_format(COLOR_FORMAT))
        self.live.start(header_text)
        self.log_indent = self._get_log_indent()
        # сохраняем для использования при finish
        self.current_header = header_text


    # ---------- STRATEGY START ----------
    def start_strategy(self, label, step_path, repeat, depth):
        line = format_cycle_header(step_path=step_path, depth=depth, kind="strategy", label=f"{label} ×{repeat}")
        self.logger.info(line)

    # ---------- CYCLE START ----------
    def start_cycle(self, label, step_path, idx, total, depth):
        line = format_cycle_header(step_path=step_path, depth=depth, kind="cycle", idx=idx, total=total)
        self.logger.info(line)


    # ---------- REPORT METRICS ----------
    def report_Rp(self, Rp):
        # arrow:    ⬊⬈➘➚↗↘⬀⬂➴➶➷➹→
        if self.prev_Rp is None:
            text = f"Rp {Rp:.3f}%"
            final_suffix = f"{text:<{RP_WIDTH}}"
        elif self.prev_Rp is not None:
            if Rp < self.prev_Rp:
                text = f"Rp {Rp:.3f}% ⬊"
                text = f"{text:<{RP_WIDTH}}"
                final_suffix = f"{LIGHT_GREEN_BG}{text}{RESET}"
            elif Rp > self.prev_Rp:
                text = f"Rp {Rp:.3f}% ⬈"
                text = f"{text:<{RP_WIDTH}}"
                final_suffix = f"{LIGHT_RED_BG}{text}{RESET}"
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
        param_data = dict:
        {
            "scale": (value, delta_percent),
            ...
        }
        разделители всегда одинаковые
        ширины всегда одинаковые
        менять формат — в одном месте
        """        
        BOLD = "\x1b[1m"
        BLUE = "\x1b[34m"
        RESET = "\x1b[0m"
        header = (f"{'Param':<{PARAM_COL_WIDTH}}"
                  f"{'Value':>{VALUE_COL_WIDTH}}"
                  f"{'Δ%':>{DELTA_COL_WIDTH}}")
        rows = []
        for p, (val, dperc) in param_data.items():
            row = (f"{BLUE}{BOLD}{p:<{PARAM_COL_WIDTH}}{RESET}"
                  f"{BLUE}{val:>{VALUE_COL_WIDTH}.6f}{RESET}"
                  f"{BLUE}{dperc:>{DELTA_COL_WIDTH}.3f}{RESET}")
            rows.append(row)
        block = "\n".join([header] + rows + [SEPARATOR])
        indented_block = "\n".join(self.log_indent + line for line in block.split("\n")) # добавляем отступ к каждой строке таблицы
        self.logger.opt(raw=True).info(indented_block + "\n")                   # выводим без лог-форматирования


    # ---------- BACKGROUND GROUP ----------
    def report_background_group(self, param_data):
        indent_ch = len(self.log_indent)  # число пробелов
        keys = list(param_data.keys())
        first = keys[0]
        last = keys[-1]

        group_label = f"bckg[{first[-1]}–{last[-1]}]"
        # --- таблица ---
        rows = ""
        for p, (val, dperc) in param_data.items():
            rows += (f"<tr>"
                     f"<td style='padding-right:25px; color:blue; font-weight:bold'>{p}</td>"
                     f"<td style='text-align:right;padding-right:25px; color:blue'>{val: <6}</td>"
                     f"<td style='text-align:right; color:blue'>{dperc: <4}</td>"
                     f"</tr>")
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
                    {rows}
                </table>
            </details>
        </div>
        """
        display(HTML(html))
        # аккуратный разделитель под таблицей
        self.logger.opt(raw=True).info(self.log_indent + SEPARATOR + "\n")



    # ---------- SAVE STEP ----------
    def save_step(self, label, step_path=None, depth=None):
        self.history.append({"label": label,
                             "step_path": step_path,
                             "depth": depth,
                             "timestamp": datetime.now(),
                             "Rp": self.current_Rp})

    # ---------- SUMMARY ----------
    def summary(self):
        print("═" * 40)
        print("FINAL SUMMARY")
        print("═" * 40)

        df = pd.DataFrame(self.history)
        display(df)

        print(f"Final Rp: {self.history[-1]['Rp']:.3f}%")

        # График
        plt.figure()
        plt.plot(df["Rp"])
        plt.xlabel("Step")
        plt.ylabel("Rp (%)")
        plt.title("Refinement convergence")
        plt.show()