# utils/paths.py

from pathlib import Path

BASE_DIR: Path | None = None


def set_base_dir(path: Path):
    global BASE_DIR
    BASE_DIR = path.resolve()


def _base() -> Path:
    if BASE_DIR is None:
        raise RuntimeError("BASE_DIR не задан. Вызови set_base_dir().")
    return BASE_DIR


def get_scattering_path(atom_name):
    path = _base() / "atoms/scattering_factors/data" / f"{atom_name}.txt"
    return path if path.exists() else None


def get_bragg_path(project_name, phase_name):
    path = _base() / "examples" / project_name / f"{phase_name}_bragg_positions.txt"
    return path if path.exists() else None


def get_cif_path(project_name, phase_name):
    path = _base() / "examples" / project_name / f"{phase_name}.cif"
    return path if path.exists() else None


def get_profile_path(project_name):
    path = _base() / "examples" / project_name / "Profile1.txt"
    return path if path.exists() else None