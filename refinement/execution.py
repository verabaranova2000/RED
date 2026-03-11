
from .metrics import profile_R_factor
from .session import RefinementSession
from .param_utils import params_for_next, val_delta_percent, is_background_param
from .schema.models import StepModel
from .segment import resolve_segment


# ==== Исполнитель шага "fit" ====
def execute_step(step: StepModel, pr, out_prev, session: RefinementSession, depth: int, step_path: str):
    """
    Исполнитель отдельного шага типа 'fit'.

    Выполняет уточнение параметров профиля для одного шага,
    применяет pre-хуки, вычисляет сегмент подгонки, запускает
    модель и фиксирует результаты в сессии.

    Аргументы
    ---------
    step : StepModel
        Объект шага с типом 'fit', параметрами, сегментом и хуками.
    pr : Project
        Объект с данными I_obs, I_calc и моделью для подгонки.
    out_prev : Any
        Результат предыдущего шага (для передачи параметров по цепочке).
    session : RefinementSession
        Сессия для логирования, накопления истории и отчётов.
    depth : int
        Глубина вложенности шага (для логирования и визуальной структуры).
    step_path : str
        Уникальный идентификатор текущего шага в иерархии схемы.

    Возвращает
    -------
    out : ModelResult из библиотеки lmfit (результат выполнения fit)
        Объект с уточнёнными параметрами, рассчитанными значениями
        модели и другой информацией о подгонке.

    Исключения
    ----------
    ValueError : при некорректных данных в step или failure модели fit.
    
    Примечания
    ---------
    - Для pre-хука 'fix_all_except' фиксируются все параметры, кроме указанных.
    - Расчёт метрики Rp и отчёт о параметрах выполняется через session.
    - Функция не изменяет саму схему. Обновляет параметры объекта Project и сессию.
    """
    # --- pre hooks ---
    if step.pre:
        for hook in step.pre:
            if hook == "fix_all_except":
                # фиксируем все, кроме указанных
                my_pars = params_for_next(pr, out_prev, refonly=step.params)
            elif hook == "noop":
                pass
    # --- resolve segment ---
    y = pr.Profile_points.I_obs
    two_theta = pr.Profile_points.two_theta
    s_idx, e_idx, s_val, e_val = resolve_segment(step, two_theta)

    session.start_step(name=step.label,
                       segment=(s_val, e_val),
                       n_params=len(step.params),
                       depth=depth,
                       step_path=step_path)
    # --- основной fit ---
    out = pr.model.fit(y[s_idx:e_idx+1], axes=two_theta[s_idx:e_idx+1], params=my_pars)
    pr.params = out.params
    Rp = profile_R_factor(y_obs=pr.Profile_points.I_obs,
                          y_calc=pr.Profile_points.I_calc)
    session.report_Rp(Rp)

    normal_params = {}
    background_params = {}

    for p in step.params:
        value, delta_percent = val_delta_percent(out.params, p)

        if is_background_param(p):
            background_params[p] = (value, delta_percent)
        else:
            normal_params[p] = (value, delta_percent)

    if normal_params:
        session.report_parameters(normal_params)

    if background_params:
        session.report_background_group(background_params)

    session.save_step(step.label, step_path=step_path, depth=depth, params=step.params)
    return out



# Исполнитель всех шагов
def execute_schema(schema_steps, pr, out_prev, session, depth=0, path=""):
    """
    Исполнитель схемы шагов refinement.

    Рекурсивно обходит список шагов:
      - fit → выполняет отдельный шаг;
      - block → контейнер шагов с повторениями и рекурсией;
      - noop → пропускает шаг.

    Аргументы
    ---------
    schema_steps : list[StepModel]
        Список шагов текущего уровня.
    pr : объект профиля
        Профиль для подгонки.
    out_prev : Any
        Результат предыдущего шага.
    session : RefinementSession
        Объект для логирования и сохранения истории.
    depth : int
        Глубина рекурсии (для логирования).
    path : str
        Идентификатор текущей ветки схемы.
    
    Возвращает
    -------
    out_prev : результат последнего выполненного шага

    Исключения
    ----------
    ValueError : если встречается неизвестный тип шага

    Примечания
    ----------
    на верхнем уровне — execute_schema() всегда вызывается для списка шагов
    внутри блока — рекурсивно вызываем execute_schema(block.steps)
    start_block — только для логирования начала блока
    """
    for step in schema_steps:
      step_path = f"{path}.{step.step_id}" if path else step.step_id
      if step.type == "fit":
        out_prev = execute_step(step, pr, out_prev, session, depth=depth, step_path=step_path)
    
      elif step.type == "block":
        repeat = step.repeat or 1
        session.start_block(step.label, step_path, repeat, depth)   
        for i in range(repeat):
          session.start_cycle(step.label, step_path, i+1, repeat, depth+1)
          out_prev = execute_schema(step.steps, pr, out_prev, session, depth=depth+1, path=step_path)
        session.current_cycle = None                 # сброс номера цикла после завершения всех циклов блока
      else:
        raise ValueError(f"Неизвестный тип шага: {step.type}")
    return out_prev