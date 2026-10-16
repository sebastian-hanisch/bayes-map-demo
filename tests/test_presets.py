"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import bys_constants as C
import bys_evaluation as ev
import bys_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["data"] in C.DATA_CHOICES and p["truth"] in C.TRUTHS and p["family"] in C.FAMILIES, name
        assert p["tau"] in C.TAU_CHOICES and p["ell"] in C.ELL_CHOICES and p["sfactor"] in C.SIGMA_FACTORS and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, prior, over, white, sig = (res[k] for k in C.PRESETS)
    assert 0.10 < std.err < 0.13 and std.coverage < 0.85 and std.z2 > 3.0
    assert prior.settings.truth == "prior" and prior.coverage >= C.OK_COVERAGE and prior.z2 <= C.OK_Z2 and prior.err < 0.05
    assert 0.10 < over.err < 0.14 and over.coverage < 0.7 and over.z2 > 100.0
    assert white.settings.family == "white" and 0.14 < white.err < 0.2 and white.coverage > 0.95 and white.z2 < C.WIDE_Z2
    assert sig.coverage < std.coverage and sig.z2 > std.z2


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 200 and C.EXP_REPS_SMALL >= 100 and C.EXP_DRAWS >= 400 and 1.0 in C.EXP_CAL_TAUS and C.EXP_CAL_ELL in C.ELL_CHOICES
    assert C.FAIL_FACTOR == 2.0 and C.NOMINAL_COVERAGE == 0.95 and 1.0 in C.SIGMA_FACTORS and C.WIDE_Z2 < 1.0 < C.OK_Z2
