
import numpy as np

def profilepoints_to_snapshot(pp):
    return {
        "data": {
            "I_obs_calibr": np.asarray(pp.I_obs_calibr),
            "two_theta": np.asarray(pp.two_theta),
        },
        "background_type": pp.settings.to_legacy_dict()['background']['type'],
        "knots": {
            "x": list(pp.knots.get("x", []))
        }
    }

def atom_to_snapshot(atom):
    return {
        "name": atom.name,
        "Z": atom.Z,
        "fe_from": atom.settings.to_legacy_dict()['fe from'],
        "it4322": atom.info.get("it4322"),
        "curves": atom.info.get("curves"),
        "KPhase": atom.KPhase,
    }


def phase_to_snapshot(phase):
    return {
        "prefix": phase.prefix,
        "bragg_positions": phase.bragg_positions,
        "atoms": [atom_to_snapshot(a) for a in phase.atoms],
        "symmetry_operations": phase.symmetry_operations,
        "wavelength": phase.wavelength,
        "settings": {
            "typeref": phase.settings.to_legacy_dict()['typeref'],
            "form": phase.settings.to_legacy_dict()['form'],
            "internal_scale": phase.settings.to_legacy_dict()['internal']['internal scale'],

            "calibration_mode": bool(phase.settings.to_legacy_dict()['calibration mode']),
            "calibrate": phase.settings.to_legacy_dict()['calibrate'],
            "corrections": phase.settings.to_legacy_dict()["corrections"]
        }
    }


def project_to_snapshot(project):
    return {
        "phases": {
            phase.name: phase_to_snapshot(phase)
            for phase in project.phases
        },        
        "profile": profilepoints_to_snapshot(project.Profile_points)
    }