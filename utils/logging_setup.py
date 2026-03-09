import sys
import logging
import re
from loguru import logger


"""
Глобальная настройка логирования проекта.

Настраивает библиотеку loguru для вывода логов и
перехватывает стандартный модуль logging через InterceptHandler,
чтобы все сообщения логирования выводились в едином формате.

Использование:
    from utils.logging_setup import setup_logging
    setup_logging()
"""


# ==== Формат логгера loguru ====
COLOR_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{extra[pylogger]: <15}</cyan> | "
    "{message} | "
    #"<level>{message}</level> | "    # жирный шрифт
)

# ==== Формат (для использования в LiveHeader) ====
def color_to_base_format(color_fmt: str) -> str:
    """
    Превращает COLOR_FORMAT loguru в BASE_FORMAT:
      - убирает все теги <...> и </...>
      - заменяет время :HH:mm:ss (формат loguru) на :%H:%M:%S  (формат datetime)
    """
    no_tags = re.sub(r"</?([a-zA-Z]+)>", "", color_fmt)
    base_fmt = no_tags.replace(":HH:mm:ss", ":%H:%M:%S")
    return base_fmt

BASE_FORMAT = color_to_base_format(COLOR_FORMAT)



class InterceptHandler(logging.Handler):
  """ 
  Перехватываем стандартный Logging → Loguru 
  Теперь, когда где-то пишем: 
  >>> self.logger = logging.getLogger(self.__class__.__name__),
  происходит цепочка:
      logging → InterceptHandler → loguru → stdout
  и сохраняется заданный формат COLOR_FORMAT.
  """
  def emit(self, record):
    try:
      level = logger.level(record.levelname).name
    except Exception:
      level = record.levelno
    #level = record.levelname
    logger.bind(pylogger=record.name). \
           opt(depth=6, exception=record.exc_info). \
           log(level, record.getMessage())



def setup_logging(level="INFO"):
    """
    Настроить глобальное логирование проекта.

    Функция инициализирует систему логирования на основе библиотеки
    loguru и перенаправляет стандартный модуль ``logging`` в loguru
    через ``InterceptHandler``. Это позволяет использовать обычный
    ``logging.getLogger(...)`` внутри классов, но получать единый
    формат логов, заданный в ``COLOR_FORMAT``.

    Параметры
    ----------
    level : str, по умолчанию "INFO"
        Минимальный уровень логирования для вывода сообщений
        (например: "DEBUG", "INFO", "WARNING", "ERROR").

    Примечания
    ----------
    Функцию нужно вызвать один раз в начале программы
    (например в ``main`` или в начале notebook).

    Пример
    -------
    >>> from utils.logging_setup import setup_logging
    >>> setup_logging()
    """
    # Убираем стандартные обработчики (handlers) logging (Jupyter-safe)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format=COLOR_FORMAT,
        level=level,
    )