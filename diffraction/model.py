from lmfit import Model
from profiles.models import Background, Spline
from diffraction.profile import phase_profile_jax_snap


def build_background_model_from_snapshot(profile_snap):
    """ Модель фона """
    bg_type = profile_snap["background_type"]
    knots = profile_snap["knots"]["x"]
    if bg_type == "Legendre":
        return Model(Background)
    elif bg_type == "Spline":
        return Model(Spline, xknots_str="_".join(str(x) for x in knots))
    elif bg_type == "Legendre + Spline":
        return (Model(Background) +
                Model(Spline, xknots_str="_".join(str(x) for x in knots)))
    else:
        raise ValueError(f"Unknown background type: {bg_type}")


def build_total_model_from_snapshot(project_snapshot):
    """ Полная модель дифракционного профиля """
    # --- 1. фон ---
    profile_snap = project_snapshot["profile"]
    total_model = build_background_model_from_snapshot(profile_snap)

    # --- 2. фазы ---
    for phase_name in project_snapshot["phases"].keys():
        total_model += Model(phase_profile_jax_snap,
                             project_snap=project_snapshot,
                             phase_name=phase_name,
                             uvar=False)

    return total_model


# === Пример использования ===
# --- Сделать snapshot ---
# project_snapshot = project_to_snapshot(pr)

# --- Старая модель ---
# y_calc_from_pr = pr.model.eval(out_0.params,
#                                axes=pr.Profile_points.two_theta)

# --- Новая модель ---
# model = build_model_from_snapshot(project_snapshot)
# y_calc_from_snap = model.eval(out_0.params,
#                               axes=project_snapshot["profile"]["data"]["two_theta"])
