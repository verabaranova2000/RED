from pathlib import Path

""" Возвращают путь к файлу, если файл существует. """

# ------ Константы модуля -------
def find_project_root(start=Path.cwd()):
    """ Ищет корень проекта по pyproject.toml """
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start

BASE_DIR = find_project_root()

SCATTERING_DATA_DIR = BASE_DIR / "atoms/scattering_factors/data"
EXAMPLES_DIR = BASE_DIR / "examples"

# ------ Путь к файлу с кривой рассеяния -------
def get_scattering_path(atom_name):
    path = SCATTERING_DATA_DIR / f"{atom_name}.txt"
    return path if path.exists() else None

## ------ Путь к файлу с bragg_positions -------
def get_bragg_path(project_name, phase_name):
    path = EXAMPLES_DIR / project_name / f"{phase_name}_bragg_positions.txt"
    return path if path.exists() else None

## ------ Путь к файлу cif -------
def get_cif_path(project_name, phase_name):
    path = EXAMPLES_DIR / project_name / f"{phase_name}.cif"
    return path if path.exists() else None

## ------ Путь к файлу с экспериментальным профилем -------
def get_profile_path(project_name):
    path = EXAMPLES_DIR / project_name / "Profile1.txt"
    return path if path.exists() else None