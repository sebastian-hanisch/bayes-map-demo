"""Auswertung: Formatierung, Instanz, Verdikte (Art und Text, Schwellen an handgesetzten Werten), Form der Experimente und ihre Reproduzierbarkeit."""

import dataclasses
import math

import numpy as np
import pytest

import bys_constants as C
import bys_evaluation as ev


# ----------------------------------------------------------------------------- Formatierung

def test_de_sci_and_share():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"
    assert ev.sci(0.0) == "0,0" and ev.sci(float("inf")) == "unendlich" and ev.sci(9.96) == "1,0·10¹"


def test_err_pct_digits_and_boundaries():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]
    assert ev.err_pct(10.0) == "1,0·10¹-fach" and ev.err_pct(9.99) == "999 %" and ev.err_pct(0.995) == "100 %" and ev.err_pct(0.994) == "99,4 %"
    assert ev.err_pct(0.0095) == "0,9 %" and ev.err_pct(0.0094) == "0,94 %" and ev.err_pct(0.0001) == "0,01 %" and ev.err_pct(0.00009) == "9,0·10⁻³ %"


def test_tau_text_num_text_and_noise_label():
    assert [ev.tau_text(v) for v in (0.5, 0.01, 0.00316)] == ["0,500", "0,010", "3,16·10⁻³"]
    assert [ev.num_text(v) for v in (1.0, 12.34, 164.5, 1500.0, 0.004)] == ["1,00", "12,3", "164,5", "1,5·10³", "4,0·10⁻³"]
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]


def test_tau_choices_are_exact_and_cover_the_presets():
    assert len(C.TAU_CHOICES) == 17 and C.TAU_CHOICES[0] == 0.001 and C.TAU_CHOICES[-1] == 10.0 and 1.0 in C.TAU_CHOICES
    assert C.tau_of(-3) == pytest.approx(10 ** -0.75, rel=1e-5)


# ----------------------------------------------------------------------------- Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings())
    assert a.mean.shape == a.sd.shape == a.lo.shape == a.hi.shape == (48,) and a.draws.shape == (C.DRAWS_SHOWN, 48)
    assert np.allclose(a.hi - a.lo, 2 * C.BAND_Z * a.sd) and np.all(a.sd > 0)
    inside = (a.x >= a.lo) & (a.x <= a.hi)
    assert a.coverage == pytest.approx(float(inside.mean())) and a.overconf == pytest.approx(math.sqrt(a.z2))
    assert a.err == pytest.approx(float(np.linalg.norm(a.mean - a.x) / np.linalg.norm(a.x)))
    assert a.tik_err <= a.err + 0.01 and a.naive_err > 2.0 and a.map_gap < 1e-8
    assert a.tau_hat == pytest.approx(float(a.evid_taus[int(np.argmax(a.evid_values))]))


def test_the_prior_truth_comes_from_the_prior_and_the_data_are_a_x_plus_e():
    a = ev.analyse(ev.Settings(truth="prior", family="se", tau=1.0, ell=0.05))
    ph = ev.analyse(ev.Settings())
    assert not np.allclose(a.x, ph.x) and 0.2 < float(np.std(a.x)) < 3.0
    p = ev.problem(48, 0.03, 1e-3)
    assert a.b == pytest.approx(p.a @ a.x + p.noise_rows(1, 11)[0])
    # Daten des Phantoms: ehrlich gegen aus dem Gitter
    h, c = ev.analyse(ev.Settings(data="ehrlich")), ev.analyse(ev.Settings(data="verbrechen"))
    assert not np.allclose(h.b, c.b) and h.b == pytest.approx(p.g + p.noise_rows(1, 11)[0])


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings()), ev.analyse(ev.Settings()), ev.analyse(ev.Settings(seed=12))
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.draws, b.draws) and not np.array_equal(a.b, c.b)


def test_a_smaller_assumed_sigma_narrows_the_band():
    a1, a4 = ev.analyse(ev.Settings(sfactor=1.0)), ev.analyse(ev.Settings(sfactor=0.25))
    assert a4.mean_sd < a1.mean_sd and a4.coverage < a1.coverage and a4.sigma_assumed == pytest.approx(0.25 * a1.sigma)


def test_the_singular_flag_follows_the_condition_number():
    assert ev.analyse(ev.Settings(n=96, w=0.05)).singular and not ev.analyse(ev.Settings()).singular


def test_small_grids_and_all_corners_are_finite():
    for n in (12, 24, 96):
        for w in (0.01, 0.05):
            for noise in (0.0, 1e-1):
                for family in C.FAMILIES:
                    for truth in C.TRUTHS:
                        a = ev.analyse(ev.Settings(n=n, w=w, noise=noise, family=family, truth=truth, tau=0.5, ell=0.1))
                        assert np.isfinite(a.mean).all() and np.isfinite(a.err) and np.isfinite(a.sd).all() and np.isfinite(a.z2)


def test_prior_draws_have_the_prior_covariance():
    p = ev.problem(12, 0.03, 1e-3)
    x = ev.prior_draws(p, "exp", 0.2, 0.7, 20000, 3)
    assert np.cov(x.T) == pytest.approx(0.49 * p.kernel("exp", 0.2), abs=0.03)


# ----------------------------------------------------------------------------- Verdikte

def _with(coverage, z2):
    a = ev.analyse(ev.Settings())
    return dataclasses.replace(a, coverage=coverage, z2=z2, overconf=math.sqrt(z2))


def test_verdict_thresholds_at_the_exact_boundaries():
    assert ev.verdict_instance(_with(0.80, 1.0))[0] == "success" and ev.verdict_instance(_with(0.79, 1.0))[0] == "warning"
    assert ev.verdict_instance(_with(1.0, 3.0))[0] == "success" and ev.verdict_instance(_with(1.0, 3.01))[0] == "warning"
    assert ev.verdict_instance(_with(1.0, 0.4))[0] == "success" and ev.verdict_instance(_with(1.0, 0.39))[0] == "info"
    assert "zu eng" in ev.verdict_instance(_with(0.5, 100.0))[1] and "zu breit" in ev.verdict_instance(_with(1.0, 0.25))[1]


def test_verdict_kinds_of_the_presets():
    out = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p)))[0] for name, p in C.PRESETS.items()}
    assert out == {"Standard": "warning", "Prior-Ziehung": "success", "Zu selbstsicher": "warning", "Kalibrierter weißer Prior": "info", "σ unterschätzt": "warning"}


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_calibration_structure_and_reproducibility():
    r1, r2 = ev.calibration_experiment(reps=30, draws=60), ev.calibration_experiment(reps=30, draws=60)
    assert [r["tau"] for r in r1["rows"]] == list(C.EXP_CAL_TAUS) and r1["rows"][0]["phantom"]["coverage"] == r2["rows"][0]["phantom"]["coverage"]
    assert all(0.0 <= r["phantom"]["coverage"] <= 1.0 and 0.0 <= r["prior"]["coverage"] <= 1.0 for r in r1["rows"])


def test_mmse_structure():
    r = ev.mmse_experiment(draws=100)
    assert len(r["rows"]) == 1 + len(C.EXP_LAMS) + len(C.EXP_TSVD_RANKS) and r["bayes"] == r["rows"][0]["mse"] and r["best_other"] > r["bayes"]


def test_evidence_structure():
    r = ev.evidence_experiment(reps=30, reps_small=10)
    assert set(r) >= {"white", "joint", "se"} and r["white"]["tau_q"][0] <= r["white"]["tau_median"] <= r["white"]["tau_q"][1] and r["se"]["reps"] == 10


# ----------------------------------------------------------------------------- Formatgrenzen, Instanzfelder und Zählungen

def test_num_text_boundaries():
    assert ev.num_text(1000.0) == "1,0·10³" and ev.num_text(999.99) == "1000,0" and ev.num_text(0.01) == "0,01" and ev.num_text(0.0099) == "9,9·10⁻³"
    assert ev.num_text(10.0) == "10,0" and ev.num_text(9.99) == "9,99" and ev.num_text(0.0) == "0,00" and ev.num_text(float("inf")) == "unendlich"


def test_naive_error_and_error_at_the_evidence_maximum_by_an_independent_route():
    a = ev.analyse(ev.Settings(family="se", tau=0.3, ell=0.05))
    p = ev.problem(48, 0.03, 1e-3)
    xn = float(np.linalg.norm(a.x))
    assert a.naive_err == pytest.approx(float(np.linalg.norm(np.linalg.solve(p.a, a.b) - a.x) / xn), rel=1e-6)
    import bys_model as m

    mean_hat = m.posterior(p.a, p.kernel("se", 0.05), a.tau_hat, a.sigma_assumed, a.b)[0][0]
    assert a.err_hat == pytest.approx(float(np.linalg.norm(mean_hat - a.x) / xn), rel=1e-9)
    assert a.err_hat != pytest.approx(float(np.linalg.norm(mean_hat + a.x) / xn), rel=1e-3)


def test_eigen_route_evidence_equals_the_general_formula_for_the_se_family():
    import bys_model as m

    p = ev.problem(48, 0.03, 1e-3)
    b = p.g[None, :] + p.noise_rows(3, 4)
    for ell, tau in ((0.03, 0.2), (0.12, 1.5), (0.2, 0.05)):
        c = p.kernel("se", ell)
        evals, q = np.linalg.eigh(p.a @ c @ p.a.T)
        direct = m.log_evidence(p.a, c, tau, p.sigma, b)
        assert ev._eig_evidence(evals, b @ q, tau, p.sigma) == pytest.approx(direct, rel=1e-6)


def test_the_experiments_use_every_repetition():
    r = ev.evidence_experiment(reps=20, reps_small=7)
    assert r["joint"]["count"] == 7 and r["se"]["count"] == 7 and r["joint"]["reps"] == 7
