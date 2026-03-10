import numpy as np
from typing import Tuple
from .schema.models import StepModel


"""
Работа с диапазонами профиля.

Модуль содержит функции для преобразования сегментов,
заданных в стратегии refinement (StepModel), в реальные
индексы массива профиля.
"""


def resolve_segment(step: StepModel, two_theta: np.ndarray,
                    clamp_out_of_bounds: bool = True
                   ) -> Tuple[int,int,float,float]:
    """
    Определяет диапазон профиля (сегмент электронограммы), используемый на данном шаге.

    Функция преобразует параметры `segment` или `segment_idx`, заданные в объекте
    `StepModel`, в индексы массива `two_theta` и соответствующие значения углов.

    Параметры
    ----------
    step : StepModel
        Шаг стратегии. Может содержать:
        - segment_idx = [start_idx, end_idx] — диапазон индексов массива two_theta
        - segment = [start_theta, end_theta] — диапазон углов (в градусах)

    two_theta : np.ndarray
        Массив углов 2θ профиля (например: [5.0, 5.01, 5.02, ..., 70.0]).

    clamp_out_of_bounds : bool, optional
        Если True, индексы, выходящие за границы массива, будут автоматически
        зажаты в допустимый диапазон [0, len(two_theta)-1].
        Если False — будет выброшено исключение.

    Возвращает
    ----------
    Tuple[int, int, float, float]
        Кортеж:
        (start_idx, end_idx, start_val, end_val)

        где
        - start_idx, end_idx — индексы в массиве two_theta
        - start_val, end_val — соответствующие значения углов.

    Правила работы
    --------------
    1. Если задан `segment_idx`, используются индексы.
    2. Если задан `segment`, для каждого угла ищется ближайшая точка в two_theta.
    3. Если значение равно None:
       - начало → используется первая точка массива
       - конец → используется последняя точка массива.
    4. Если start_idx > end_idx — индексы автоматически меняются местами.
    5. Если ни `segment`, ни `segment_idx` не заданы — используется весь диапазон.

    Пример
    -------
    >>> two_theta = np.linspace(10, 80, 1001)
    >>> step.segment = [25.0, 35.0]
    >>> resolve_segment(step, two_theta)
    (214, 357, 25.0, 35.0)
    """

    n = len(two_theta)
    if n == 0:
        raise ValueError("Массив two_theta пуст.")

    def clamp_idx(i):
        """Проверка и корректировка индекса."""
        if i is None:
            return None
        if i < 0 or i >= n:
            if clamp_out_of_bounds:
                return max(0, min(n-1, i))
            else:
                raise IndexError(f"Индекс segment_idx={i} выходит за границы массива [0, {n-1}]")
        return i

    # --- если заданы индексы ---
    if step.segment_idx is not None:
        s_idx, e_idx = step.segment_idx
        s = clamp_idx(s_idx)
        e = clamp_idx(e_idx)
        # None → использовать крайние точки массива
        if s is None: s = 0
        if e is None: e = n - 1
        # привести порядок индексов к start ≤ end
        if s > e:
            s, e = e, s
        s_val = float(two_theta[s])
        e_val = float(two_theta[e])
        return s, e, s_val, e_val

    # --- если задан диапазон углов ---
    if step.segment is not None:
        s_val_raw, e_val_raw = step.segment

        # начало диапазона
        if s_val_raw is None:
            s_idx = 0
            s_val = float(two_theta[0])
        else:
            # поиск индекса ближайшего значения угла
            arr = np.asarray(two_theta)
            # если two_theta строго монотонный, можно использовать searchsorted,
            # но argmin по |Δθ| работает надёжно в любом случае
            s_idx = int(np.argmin(np.abs(arr - float(s_val_raw))))
            s_val = float(two_theta[s_idx])

        # конец диапазона
        if e_val_raw is None:
            e_idx = n - 1
            e_val = float(two_theta[-1])
        else:
            arr = np.asarray(two_theta)
            e_idx = int(np.argmin(np.abs(arr - float(e_val_raw))))
            e_val = float(two_theta[e_idx])

        # привести порядок индексов к start ≤ end
        if s_idx > e_idx:
            s_idx, e_idx = e_idx, s_idx
            s_val, e_val = e_val, s_val

        return s_idx, e_idx, s_val, e_val

    # --- если сегмент не задан: использовать весь диапазон ---
    return 0, n-1, float(two_theta[0]), float(two_theta[-1])
