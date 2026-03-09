import sys
from datetime import datetime

from utils.logging_setup import BASE_FORMAT

"""
Модуль для live-вывода заголовка шага refinement.

Содержит класс LiveHeader, который используется для временного
отображения строки лога в момент начала выполнения шага.
Эта строка заменяется финальным лог-сообщением после завершения шага.

Позволяет пользователю видеть заголовок стартовавшего, но еще незавершенного шага refinement
до того, как будут уточнены параметры и вычислены итоговые метрики (например, Rp).

Используется в классе RefinementSession для структурированного вывода
хода refinement в терминале/блокноте.
"""


class LiveHeader:
    """
    Управляет временной строкой лога для выполняющегося шага refinement.

    Класс печатает строку заголовка шага в текущей строке терминала,
    пока шаг выполняется. После завершения шага эта строка удаляется
    и заменяется обычным лог-сообщением через loguru.

    Parameters
    ----------
    logger : loguru.Logger
        Экземпляр логгера, через который будет выводиться финальный лог.

    pylogger : str
        Имя логгера (обычно имя класса), отображаемое в колонке логов.

    level : str, default="INFO"
        Уровень логирования для финального сообщения.

    base_format : str, optional
        Базовый формат строки лога без цветовых тегов. 
        Если не указан, используется глобальный BASE_FORMAT 
        (созданный на основе глобального формата логов loguru) из logging_setup.
    
    Example
    -------   
    >>> liveh = LiveHeader(logger, pylogger="RefinementStep")
    >>> current_header = "HH:MM:SS | INFO    | RefinementStep  | ▶ [001] SCALE        ( 1) | 0.32–2.13° | "
    >>> final_suffix = "Rp 76.956% ⬈"
    >>> liveh.start(current_header)
    >>> liveh.finish(current_header, final_suffix)   

    Before-after
    ------- 
    05:25:11 | INFO    | RefinementStep  | ▶ [001] SCALE        ( 1) | 0.32–2.13° | 
    05:26:01 | INFO    | RefinementStep  | ▶ [001] SCALE        ( 1) | 0.32–2.13° | Rp 76.956% ⬈   |
    """
    def __init__(self, logger, pylogger: str, level: str ="INFO", base_format: str | None = None):
        self.logger = logger
        self.pylogger = pylogger
        self.level = level
        self.base_format = base_format or BASE_FORMAT
        self.active = False

    def _build_prefix(self):
        now = datetime.now()
        formatted = self.base_format.format( 
            time=now,
            level=self.level,
            extra={"pylogger": self.pylogger},
            message="") # временно подставляем пустое message
        prefix = formatted.rstrip(" |") + " "
        return prefix

    def start(self, message):
        self.active = True
        prefix = self._build_prefix()
        sys.stdout.write(prefix + " | " + message + " ")
        sys.stdout.flush()

    def finish(self, message, final_suffix):
        if not self.active:
            return
        # стереть текущую строку
        sys.stdout.write("\r")
        sys.stdout.write(" " * 300)
        sys.stdout.write("\r")
        # финальный лог через loguru (с цветами)
        self.logger.info(f"{message} {final_suffix}")

        self.active = False