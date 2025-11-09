## === Функция get_value ===
def get_value(x):
    """
    Для корректного использования объектов Parameter в jax-расчетах
    Возвращает число из lmfit.Parameter или просто число.
    """
    return x.value if hasattr(x, "value") else x