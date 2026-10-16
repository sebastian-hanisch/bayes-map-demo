"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit."""

import numpy as np
import pytest

import bys_constants as C
import bys_evaluation as ev
import bys_model as m


@pytest.fixture(scope="module")
def calibration():
    return ev.calibration_experiment()


@pytest.fixture(scope="module")
def mmse():
    return ev.mmse_experiment()


@pytest.fixture(scope="module")
def evidence():
    return ev.evidence_experiment()


# ----------------------------------------------------------------------------- Instanz und Presets

def test_standard_instance_numbers():
    a = ev.analyse(ev.Settings())
    assert 1.2e4 < a.cond < 1.3e4 and a.sigma == pytest.approx(3.4e-4, rel=0.02)
    assert a.err == pytest.approx(0.116, abs=0.003) and a.coverage == pytest.approx(0.77, abs=0.03) and a.z2 == pytest.approx(19.6, rel=0.1)


def test_preset_help_numbers():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    assert res["Prior-Ziehung"].err == pytest.approx(0.0079, abs=0.002) and res["Prior-Ziehung"].coverage == pytest.approx(0.85, abs=0.05)
    assert res["Zu selbstsicher"].err == pytest.approx(0.121, abs=0.003) and res["Zu selbstsicher"].coverage == pytest.approx(0.60, abs=0.03) and res["Zu selbstsicher"].z2 > 100
    assert res["Kalibrierter weißer Prior"].err == pytest.approx(0.165, abs=0.01) and res["Kalibrierter weißer Prior"].coverage == 1.0
    assert res["σ unterschätzt"].err == pytest.approx(0.127, abs=0.005) and res["σ unterschätzt"].coverage == pytest.approx(0.27, abs=0.04)


def test_map_equals_tikhonov_for_the_white_prior_at_every_preset():
    for name, p in C.PRESETS.items():
        assert ev.analyse(ev.Settings(**p)).map_gap < 1e-8, name


def test_the_calibrated_white_prior_is_worse_than_the_best_one_in_the_mean():
    p = ev.problem(48, 0.03, 1e-3)
    b = p.g[None, :] + p.noise_rows(200, C.SEED_BASE + 1)
    c = p.kernel("white", 0.0)
    xn = float(np.linalg.norm(p.x_phantom))
    tau_cal = float(np.sqrt(np.mean(p.x_phantom ** 2)))
    err_cal = float(np.sqrt(np.mean((np.linalg.norm(m.posterior(p.a, c, tau_cal, p.sigma, b)[0] - p.x_phantom, axis=1) / xn) ** 2)))
    taus = np.logspace(-3, 1, 25)
    errs = [float(np.sqrt(np.mean((np.linalg.norm(m.posterior(p.a, c, t, p.sigma, b)[0] - p.x_phantom, axis=1) / xn) ** 2))) for t in taus]
    assert tau_cal == pytest.approx(0.375, abs=0.005) and err_cal == pytest.approx(0.210, abs=0.01)
    assert min(errs) == pytest.approx(0.117, abs=0.003) and taus[int(np.argmin(errs))] == pytest.approx(0.0316, rel=0.3)


def test_the_shape_of_the_prior_hardly_changes_the_error():
    p = ev.problem(48, 0.03, 1e-3)
    b = p.g[None, :] + p.noise_rows(200, C.SEED_BASE + 1)
    xn = float(np.linalg.norm(p.x_phantom))
    best = {}
    for family, ells in (("se", (0.02, 0.03, 0.05, 0.08)), ("m32", (0.12, 0.2, 0.4)), ("exp", (0.08, 0.12, 0.2))):
        for ell in ells:
            for tau in np.logspace(-1.5, 0.5, 9):
                mean, _ = m.posterior(p.a, p.kernel(family, ell), float(tau), p.sigma, b)
                e = float(np.sqrt(np.mean((np.linalg.norm(mean - p.x_phantom, axis=1) / xn) ** 2)))
                best[family] = min(best.get(family, 9.0), e)
    assert all(0.114 < v < 0.12 for v in best.values()), best


# ----------------------------------------------------------------------------- Experiment 1

def test_calibration_numbers(calibration):
    rows = {r["tau"]: r for r in calibration["rows"]}
    for tau, cov, z2 in ((0.1, 0.403, 1232.0), (0.3, 0.526, 483.0), (1.0, 0.639, 166.0), (3.0, 0.727, 63.0)):
        assert rows[tau]["phantom"]["coverage"] == pytest.approx(cov, abs=0.01) and rows[tau]["phantom"]["z2"] == pytest.approx(z2, rel=0.05)
    for tau in C.EXP_CAL_TAUS:
        assert rows[tau]["prior"]["coverage"] == pytest.approx(0.95, abs=0.01) and rows[tau]["prior"]["z2"] == pytest.approx(1.0, abs=0.06)
    assert rows[1.0]["phantom"]["rms"] == pytest.approx(0.122, abs=0.004) and rows[1.0]["prior"]["rms"] == pytest.approx(0.005, abs=0.001)


# ----------------------------------------------------------------------------- Experiment 2

def test_mmse_numbers(mmse):
    assert mmse["bayes"] == pytest.approx(mmse["trace"], rel=0.06) and mmse["bayes"] == pytest.approx(8.5e-4, rel=0.1)
    rows = {r["label"]: r["mse"] for r in mmse["rows"]}
    assert rows["Tikhonov, λ = 1·10⁻²"] == pytest.approx(4.05e-3, rel=0.1) and rows["Tikhonov, λ = 1·10⁻⁴"] > 10.0
    assert rows["TSVD, Rang 24"] == pytest.approx(4.8e-2, rel=0.1) and rows["TSVD, Rang 16"] == pytest.approx(0.92, rel=0.1) and rows["TSVD, Rang 8"] > 5.0
    assert mmse["best_other"] / mmse["bayes"] == pytest.approx(4.7, abs=0.4)
    assert all(r["mse"] > mmse["bayes"] for r in mmse["rows"][1:])


# ----------------------------------------------------------------------------- Experiment 3

def test_evidence_numbers(evidence):
    w, j, s = evidence["white"], evidence["joint"], evidence["se"]
    assert w["tau_median"] == pytest.approx(0.398, rel=0.01) and w["tau_q"][0] == pytest.approx(0.398, rel=0.01) and w["tau_opt"] == pytest.approx(0.0316, rel=0.01)
    assert w["median"] == pytest.approx(0.215, abs=0.01) and w["rms"] == pytest.approx(0.218, abs=0.01) and w["opt"] == pytest.approx(0.117, abs=0.003)
    assert w["fail"] == pytest.approx(0.275, abs=0.05)
    assert j["sigma_ratio_median"] == pytest.approx(0.68, abs=0.06) and j["median"] == pytest.approx(0.269, abs=0.02)
    assert s["ell_median"] == 0.03 and s["median"] == pytest.approx(0.117, abs=0.004) and s["median"] < 0.6 * w["median"]
    assert j["rms"] == pytest.approx(0.282, abs=0.02) and s["rms"] == pytest.approx(0.117, abs=0.004) and s["tau_median"] == pytest.approx(0.316, rel=0.01)


def test_verdict_texts_carry_the_numbers(calibration, mmse, evidence):
    cal = ev.verdict_calibration(calibration)[1]
    assert "95 %" in cal and "z² = 0,99" in cal and "64 %" in cal and "z² = 166,2" in cal and "12,2 %" in cal
    mm = ev.verdict_mmse(mmse)[1]
    assert "400 Ziehungen" in mm and "4,7-Fache" in mm
    e = ev.verdict_evidence(evidence)[1]
    assert "τ̂ = 0,398" in e and "0,032" in e and "21,5 %" in e and "28 % gescheitert" in e and "0,68" in e and "ℓ̂ im Median 0,03" in e
