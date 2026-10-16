"""Kernroutinen an der 3 x 3-Mini-Instanz von Hand (DEMO-PLAYBOOK §1b); jede Einheit einzeln."""

import math

import numpy as np
import pytest

import bys_model as m


def test_kernels_by_hand():
    s = np.array([0.0, 1.0])
    assert m.kernel(s, 1.0, "white").tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert m.kernel(s, 1.0, "se")[0, 1] == pytest.approx(math.exp(-0.5))
    assert m.kernel(s, 1.0, "exp")[0, 1] == pytest.approx(math.exp(-1.0))
    assert m.kernel(s, 1.0, "m32")[0, 1] == pytest.approx((1.0 + math.sqrt(3.0)) * math.exp(-math.sqrt(3.0)))
    assert m.kernel(s, 2.0, "se")[0, 1] == pytest.approx(math.exp(-0.125))
    assert np.allclose(m.kernel(s, 1.0, "se"), m.kernel(s, 1.0, "se").T) and m.kernel(s, 1.0, "se").diagonal().tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        m.kernel(s, 1.0, "unbekannt")


def test_posterior_mean_covariance_and_evidence_by_hand(mini):
    a, x, b = mini
    mean, cov = m.posterior(a, np.eye(3), 1.0, 0.1, b)
    assert mean.shape == (1, 3)
    assert mean[0] == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6)
    assert cov[0, 0] == pytest.approx(0.105864, abs=2e-6) and cov[1, 1] == pytest.approx(0.336634, abs=2e-6) and cov[0, 2] == pytest.approx(0.067403, abs=2e-6)
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0)
    assert sorted(np.linalg.eigvalsh(cov)) == pytest.approx([1.0 / 101.0, 1.0 / 26.0, 0.5])
    assert m.log_evidence(a, np.eye(3), 1.0, 0.1, b)[0] == pytest.approx(-2.9391, abs=1e-3)


def test_map_is_tikhonov_with_lambda_sigma_over_tau(mini):
    a, x, b = mini
    u, sv, vt = m.decompose(a)
    for tau, sigma in ((1.0, 0.1), (0.5, 0.05), (2.0, 0.3)):
        mean = m.posterior(a, np.eye(3), tau, sigma, b)[0][0]
        assert m.tikhonov(u, sv, vt, b, sigma / tau)[0] == pytest.approx(mean, abs=1e-12)
    assert m.tikhonov(u, sv, vt, b, 0.0)[0] == pytest.approx(x, abs=1e-9)


def test_limits_vanishing_noise_and_vague_prior_give_the_direct_inversion(mini):
    a, x, b = mini
    assert m.posterior(a, np.eye(3), 1.0, 1e-9, b)[0][0] == pytest.approx(x, abs=1e-5)
    assert m.posterior(a, np.eye(3), 1e6, 0.1, b)[0][0] == pytest.approx(x, abs=1e-4)
    assert m.posterior(a, np.eye(3), 1e-9, 0.1, b)[0][0] == pytest.approx(np.zeros(3), abs=1e-6)     # enger Prior: Mittel null


def test_posterior_handles_several_rows_and_the_covariance_does_not_depend_on_the_data(mini):
    a, x, b = mini
    rows = np.vstack([b, b + 0.1, 0.0 * b])
    mean, cov = m.posterior(a, np.eye(3), 1.0, 0.1, rows)
    assert mean.shape == (3, 3) and not mean[2].any()
    assert mean[0] == pytest.approx(m.posterior(a, np.eye(3), 1.0, 0.1, b)[0][0])
    assert cov == pytest.approx(m.posterior(a, np.eye(3), 1.0, 0.1, b)[1])


def test_evidence_decreases_when_the_prior_is_wrong_by_a_large_factor(mini):
    a, x, b = mini
    good = m.log_evidence(a, np.eye(3), 1.0, 0.1, b)[0]
    assert m.log_evidence(a, np.eye(3), 1e-3, 0.1, b)[0] < good - 10.0
    assert m.log_evidence(a, np.eye(3), 1.0, 0.1, np.vstack([b, b])).shape == (2,)


def test_coverage_and_z2_by_hand():
    cov = np.diag([1.0, 4.0])
    c, z2 = m.coverage_and_z2(np.array([[0.0, 0.0]]), cov, np.array([1.0, 3.0]))
    assert c == 1.0 and z2 == pytest.approx(1.625)                                        # z = (-1, -1,5)
    c, z2 = m.coverage_and_z2(np.array([[0.0, 0.0]]), cov, np.array([3.0, 3.0]))
    assert c == 0.5 and z2 == pytest.approx((9.0 + 2.25) / 2.0)                           # z = (-3, -1,5): nur der zweite innerhalb 1,96
    c, _ = m.coverage_and_z2(np.array([[0.0, 0.0]]), cov, np.array([1.96, 0.0]))
    assert c == 1.0                                                                       # genau auf der Grenze zählt als innerhalb
    c, _ = m.coverage_and_z2(np.array([[0.0, 0.0]]), cov, np.array([1.97, 0.0]))
    assert c == 0.5


def test_sqrt_factor_and_gaussian_draws_by_hand():
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    r = m.sqrt_factor(cov)
    assert r @ r.T == pytest.approx(cov)
    assert m.sqrt_factor(np.diag([-1e-12, 4.0])) @ m.sqrt_factor(np.diag([-1e-12, 4.0])).T == pytest.approx(np.diag([0.0, 4.0]), abs=1e-9)
    d = m.draw_gaussian(np.array([1.0, 2.0]), np.diag([4.0, 9.0]), np.array([[1.0, -1.0], [0.0, 0.0]]))
    assert d.shape == (2, 2) and d[0] == pytest.approx([3.0, -1.0]) and d[1] == pytest.approx([1.0, 2.0])
