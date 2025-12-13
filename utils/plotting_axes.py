import numpy as np

def align_zero_axes(fig, row=1, col=1, pad=1.05,
                    left_axis='y', right_axis='y2',
                    show_zeroline=True):
    """
    Симметричное выравнивание диапазонов осей относительно y=0
    по видимым трассам subplot'а.

    fig        — plotly Figure
    row, col   — номер subplot'а
    pad        — запас по амплитуде
    left_axis  — имя левой оси
    right_axis — имя правой оси
    """
    y_left, y_right = [], []
    for tr in fig.data:
        if not tr.visible or tr.y is None:
            continue
        ys = np.asarray(tr.y, dtype=float)
        if getattr(tr, 'yaxis', left_axis) == right_axis:
            y_right += [np.nanmin(ys), np.nanmax(ys)]
        else:
            y_left += [np.nanmin(ys), np.nanmax(ys)]
    if y_left:
        M = max(abs(min(y_left)), abs(max(y_left))) * pad or pad
        fig.update_yaxes(range=[-M, M], row=row, col=col,
                         autorange=False, zeroline=show_zeroline)
    if y_right:
        M = max(abs(min(y_right)), abs(max(y_right))) * pad or pad
        fig.update_yaxes(range=[-M, M], row=row, col=col,
                         secondary_y=True,
                         autorange=False, zeroline=show_zeroline)


def symmetric_range_from_arrays(arrays, pad=1.05):
    """
    Вычисляет симметричный диапазон оси [-M, M] по набору массивов данных.

    Значение M определяется как максимальное по модулю значение среди
    минимальных и максимальных элементов всех массивов, умноженное на коэффициент pad.
    Если входные массивы отсутствуют или содержат только NaN, возвращается None.
    """
    vals = []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            continue
        vals.append(np.nanmin(a))
        vals.append(np.nanmax(a))
    if not vals:
        return None
    mn, mx = min(vals), max(vals)
    M = max(abs(mn), abs(mx)) * pad
    if M == 0:
        M = 1.0 * pad
    return [-float(M), float(M)]