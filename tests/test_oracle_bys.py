"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen gegen einen anderen Rechenweg (Präzisionsmatrix-Formel, Minimierung der negativen Log-Posterior, scipy-Dichte,
numerische Integration, Monte Carlo, sequentielles Update, Eigenzerlegung der Kovarianz). Jeder Test unter 10 s.

Fehlt scipy im Test-venv, überspringen sich die scipy-Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import bys_evaluation as ev
import bys_model as m
from inverse_generator import SplitMix64, blurred_exact, grid, operator


def _inst(n=12, w=0.03, noise=1e-2, seed=5, family="exp", ell=0.2, tau=0.5):
    a = operator(n, w)
    s = grid(n)
    c = m.kernel(s, ell, family)
    g = blurred_exact(s, w)
    sigma = noise * float(np.sqrt(np.mean(g ** 2)))
    b = g + SplitMix64(seed).normal(n) * sigma
    return a, c, tau, sigma, b


@pytest.mark.parametrize("n,family", [(12, "exp"), (24, "m32"), (48, "exp"), (12, "white")])
def test_data_space_formulas_match_the_precision_matrix_formula(n, family):
    a, c, tau, sigma, b = _inst(n, family=family)
    mean, cov = m.posterior(a, c, tau, sigma, b)
    prec = a.T @ a / sigma ** 2 + np.linalg.inv(tau ** 2 * c)
    cov_ref = np.linalg.inv(prec)
    mean_ref = cov_ref @ a.T @ b / sigma ** 2
    assert cov == pytest.approx(cov_ref, rel=1e-5, abs=1e-9 * float(np.abs(cov_ref).max()))
    assert mean[0] == pytest.approx(mean_ref, rel=1e-5, abs=1e-8)


def test_map_is_the_minimizer_of_the_negative_log_posterior():
    scipy = pytest.importorskip("scipy")
    from scipy import optimize

    a, c, tau, sigma, b = _inst(12, family="se", ell=0.2, tau=0.5)
    c = c + 1e-6 * np.eye(12)
    cinv = np.linalg.inv(c)

    def f(x):
        r = a @ x - b
        return 0.5 * float(r @ r) / sigma ** 2 + 0.5 * float(x @ cinv @ x) / tau ** 2

    def g(x):
        return a.T @ (a @ x - b) / sigma ** 2 + cinv @ x / tau ** 2

    res = optimize.minimize(f, np.zeros(12), jac=g, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 20000})
    mean = m.posterior(a, c, tau, sigma, b)[0][0]
    assert f(mean) <= f(res.x) + 1e-8 * max(1.0, abs(f(res.x)))
    assert mean == pytest.approx(res.x, rel=1e-3, abs=1e-4)


def test_evidence_matches_the_scipy_gaussian_density():
    scipy = pytest.importorskip("scipy")
    from scipy import stats

    a, c, tau, sigma, b = _inst(12)
    k = m.data_covariance(a, c, tau, sigma)
    assert m.log_evidence(a, c, tau, sigma, b)[0] == pytest.approx(float(stats.multivariate_normal(mean=np.zeros(12), cov=k).logpdf(b)), rel=1e-9)


def test_evidence_matches_numerical_integration_over_the_prior():
    """n = 2: p(b) = integral N(b; A x, sigma^2 I) N(x; 0, tau^2 C) dx auf einem feinen Gitter."""
    a = np.array([[0.8, 0.2], [0.3, 0.7]])
    c = np.array([[1.0, 0.4], [0.4, 1.0]])
    tau, sigma = 0.7, 0.2
    b = np.array([0.5, -0.3])
    xs = np.linspace(-4.0, 4.0, 801)
    x1, x2 = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([x1.ravel(), x2.ravel()], axis=1)
    cinv = np.linalg.inv(tau ** 2 * c)
    log_prior = -0.5 * np.einsum("ij,jk,ik->i", pts, cinv, pts) - 0.5 * math.log((2.0 * math.pi) ** 2 * np.linalg.det(tau ** 2 * c))
    res = pts @ a.T - b
    log_lik = -0.5 * np.sum(res ** 2, axis=1) / sigma ** 2 - math.log(2.0 * math.pi * sigma ** 2)
    integral = float(np.sum(np.exp(log_prior + log_lik)) * (xs[1] - xs[0]) ** 2)
    assert m.log_evidence(a, c, tau, sigma, b)[0] == pytest.approx(math.log(integral), abs=1e-4)


def test_white_evidence_via_svd_matches_the_general_formula():
    p = ev.problem(48, 0.03, 1e-3)
    b = p.g + p.noise_rows(1, 3)[0]
    beta = (b @ p.u)[None, :]
    for tau in (0.03, 0.4, 2.0):
        direct = m.log_evidence(p.a, p.kernel("white", 0.0), tau, p.sigma, b)[0]
        assert ev._white_evidence(p, beta, tau, p.sigma)[0] == pytest.approx(direct, rel=1e-8)


def test_calibration_holds_in_the_model_monte_carlo():
    n = 24
    a, c, tau, sigma, b = _inst(n, family="se", ell=0.1, tau=1.0)
    p = ev.problem(n, 0.03, 1e-2)
    draws = 4000
    x = ev.prior_draws(p, "se", 0.1, 1.0, draws, 77)
    bs = x @ p.a.T + p.noise_rows(draws, 78)
    mean, cov = m.posterior(p.a, p.kernel("se", 0.1), 1.0, p.sigma, bs)
    cover, z2 = m.coverage_and_z2(mean, cov, x)
    assert cover == pytest.approx(0.95, abs=0.01) and z2 == pytest.approx(1.0, abs=0.04)
    mse = float(np.mean(np.sum((mean - x) ** 2, axis=1)))
    assert mse == pytest.approx(float(np.trace(cov)), rel=0.04)


def test_the_posterior_mean_has_the_smallest_bayes_risk():
    n = 12
    p = ev.problem(n, 0.03, 0.3)
    c = p.kernel("exp", 0.2)
    draws = 6000
    x = ev.prior_draws(p, "exp", 0.2, 0.5, draws, 5)
    bs = x @ p.a.T + p.noise_rows(draws, 6)
    mean, _ = m.posterior(p.a, c, 0.5, p.sigma, bs)
    base = float(np.mean(np.sum((mean - x) ** 2, axis=1)))
    rng = np.random.default_rng(1)
    for _ in range(40):
        direction = rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        assert float(np.mean(np.sum((mean + 5e-2 * direction - x) ** 2, axis=1))) > base
    others = []
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        t = m.tikhonov(p.u, p.sv, p.vt, bs, lam)
        others.append(float(np.mean(np.sum((t - x) ** 2, axis=1))))
    assert all(o >= 0.99 * base for o in others) and max(others) > 1.5 * base


def test_sequential_update_equals_the_batch_posterior():
    """Das Kalman-Update: erst die erste Datenhälfte, dann die zweite mit dem Posterior als Prior, ergibt den Gesamt-Posterior."""
    n = 12
    a, c, tau, sigma, b = _inst(n, family="exp", ell=0.2, tau=0.5)
    m1, g1 = m.posterior(a[:6], c, tau, sigma, b[:6])
    a2, b2 = a[6:], b[6:]
    gain = g1 @ a2.T @ np.linalg.inv(a2 @ g1 @ a2.T + sigma ** 2 * np.eye(6))
    m2 = m1[0] + gain @ (b2 - a2 @ m1[0])
    g2 = g1 - gain @ a2 @ g1
    mb, gb = m.posterior(a, c, tau, sigma, b)
    assert m2 == pytest.approx(mb[0], rel=1e-8, abs=1e-9)
    assert g2 == pytest.approx(gb, rel=1e-6, abs=1e-10)


def test_laplace_prior_map_is_the_l1_solution_with_lambda_sigma_squared_over_b():
    scipy = pytest.importorskip("scipy")
    from scipy import optimize

    a, c, tau, sigma, b = _inst(12, noise=1e-2)
    scale = 0.05
    lam = sigma ** 2 / scale

    def f(v):
        x = v[:12] - v[12:]
        r = a @ x - b
        return 0.5 * float(r @ r) / sigma ** 2 + float(v.sum()) / scale

    def g(v):
        x = v[:12] - v[12:]
        gr = a.T @ (a @ x - b) / sigma ** 2
        return np.concatenate([gr + 1.0 / scale, -gr + 1.0 / scale])

    res = optimize.minimize(f, np.zeros(24), jac=g, method="L-BFGS-B", bounds=[(0, None)] * 24, options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 20000})
    x_map = res.x[:12] - res.x[12:]
    # ISTA für 1/2 |A x - b|^2 + lam |x|_1 mit Schrittweite 1/sigma_1^2
    l2 = float(np.linalg.norm(a, 2) ** 2)
    x = np.zeros(12)
    for _ in range(40000):
        z = x - a.T @ (a @ x - b) / l2
        x = np.sign(z) * np.maximum(np.abs(z) - lam / l2, 0.0)
    assert x == pytest.approx(x_map, abs=2e-4)


def test_exact_gaussian_draws_reproduce_the_posterior_covariance():
    a, c, tau, sigma, b = _inst(12)
    mean, cov = m.posterior(a, c, tau, sigma, b)
    z = SplitMix64(9).normal(12 * 20000).reshape(20000, 12)
    d = m.draw_gaussian(mean[0], cov, z)
    assert np.cov(d.T) == pytest.approx(cov, rel=0.05, abs=0.05 * float(np.abs(np.diag(cov)).max()))
    assert d.mean(axis=0) == pytest.approx(mean[0], abs=5.0 * float(np.sqrt(np.diag(cov).max())) / math.sqrt(20000))
