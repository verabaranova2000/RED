
from .metrics import profile_R_factor
from .session import RefinementSession
from .param_utils import val_delta_percent

# ==== Исполнитель шага "fit" ====
def execute_step(step: StepModel, pr, out_prev, session: RefinementSession, depth: int, step_path: str):
    # --- pre hooks ---
    if step.pre:
        for hook in step.pre:
            if hook == "fix_all_except":
                # фиксируем все, кроме указанных
                my_pars = params_for_next(out_prev, refonly=step.params)
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

        if p.startswith("bckg"):
            background_params[p] = (value, delta_percent)
        else:
            normal_params[p] = (value, delta_percent)

    if normal_params:
        session.report_parameters(normal_params)

    if background_params:
        session.report_background_group(background_params)

    session.save_step(step.label, step_path=step_path, depth=depth)
    return out



# ==== Исполнитель всех шагов (типа "fit" и "strategy") в цикле ====
def execute_strategy(strategy_steps, pr, out_prev, session, depth=0, path=""):
  for step in strategy_steps:
    step_path = f"{path}.{step.step_id}" if path else step.step_id
    if step.type == "fit":
      out_prev = execute_step(step, pr, out_prev, session, depth=depth, step_path=step_path)
    
    elif step.type == "strategy":
      repeat = step.repeat or 1
      session.start_strategy(step.label, step_path, repeat, depth)   
      for i in range(repeat):
        session.start_cycle(step.label, step_path, i+1, repeat, depth+1)
        out_prev = execute_strategy(step.strategy, pr, out_prev, session, depth=depth+1, path=step_path)
    else:
      raise ValueError(f"Unknown step type: {step.type}")
  return out_prev