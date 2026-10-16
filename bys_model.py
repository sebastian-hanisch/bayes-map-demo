"""Kern: Prior-Kovarianzen, linear-gaußscher Posterior, Evidenz, exakte Ziehungen, Kalibrierungsmaße. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_bys.py` gegen die Präzisionsmatrix-Formel, scipy (Dichte, Minimierung), Monte Carlo und das sequentielle Update.

Modell: x ~ N(0, tau^2 C), b = A x + e, e ~ N(0, sigma^2 I). Mit K = tau^2 A C A^T + sigma^2 I (Kovarianz der Daten) gilt in den Datenraumformeln
    Mittel   m = tau^2 C A^T K^{-1} b,
    Kovarianz G = tau^2 C - tau^4 C A^T K^{-1} A C,
    log p(b) = -1/2 (log det(2 pi K) + b^T K^{-1} b).
Das Mittel ist zugleich das MAP (Gaußscher Fall) und für C = I die Tikhonov-Lösung mit lambda = sigma / tau.
"""

from __future__ import annotations

import math

import numpy as np

import bys_constants as C

_TINY = 1e-300


def kernel(s: np.ndarray, ell: float, family: str) -> np.ndarray:
    """Korrelationsmatrix k(|s_i - s_j|) der Prior-Familie; `white` ignoriert ell."""
    s = np.asarray(s, dtype=float)
    d = np.abs(s[:, None] - s[None, :])
    if family == "white":
        return np.eye(len(s))
    if family == "se":
        return np.exp(-0.5 * (d / ell) ** 2)
    if family == "exp":
        return np.exp(-d / ell)
    if family == "m32":
        r = math.sqrt(3.0) * d / ell
        return (1.0 + r) * np.exp(-r)
    raise ValueError(family)


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    return np.linalg.svd(a)


def tikhonov(u: np.ndarray, sv: np.ndarray, vt: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov-Lösung für jede Zeile von b: Filter sv^2 / (sv^2 + lam^2) wie in Stück 2."""
    f = sv ** 2 / (sv ** 2 + lam ** 2)
    return ((np.atleast_2d(b) @ u) * f / np.maximum(sv, _TINY)) @ vt


def data_covariance(a: np.ndarray, c: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """K = tau^2 A C A^T + sigma^2 I."""
    return tau ** 2 * (a @ c @ a.T) + sigma ** 2 * np.eye(a.shape[0])


def posterior(a: np.ndarray, c: np.ndarray, tau: float, sigma: float, b: np.ndarray):
    """(Mittel (reps, n), Kovarianz (n, n)) des Posteriors für jede Zeile von b; die Kovarianz hängt nicht von b ab."""
    b = np.atleast_2d(b)
    k = data_covariance(a, c, tau, sigma)
    gain = tau ** 2 * (c @ a.T)                                        # tau^2 C A^T, (n, m)
    mean = np.linalg.solve(k, b.T).T @ gain.T
    cov = tau ** 2 * c - gain @ np.linalg.solve(k, gain.T)
    return mean, 0.5 * (cov + cov.T)


def log_evidence(a: np.ndarray, c: np.ndarray, tau: float, sigma: float, b: np.ndarray) -> np.ndarray:
    """log p(b | tau, sigma, C) für jede Zeile von b (die Daten sind N(0, K))."""
    b = np.atleast_2d(b)
    k = data_covariance(a, c, tau, sigma)
    _, logdet = np.linalg.slogdet(k)
    q = np.einsum("ij,ij->i", b, np.linalg.solve(k, b.T).T)
    return -0.5 * (logdet + q + b.shape[1] * math.log(2.0 * math.pi))


def sqrt_factor(c: np.ndarray) -> np.ndarray:
    """R mit R R^T = c (Eigenzerlegung, negative Eigenwerte durch Rundung auf null gesetzt)."""
    ev, evec = np.linalg.eigh(0.5 * (c + c.T))
    return evec * np.sqrt(np.maximum(ev, 0.0))


def draw_gaussian(mean: np.ndarray, cov: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Ziehungen mean + R z für die Standardnormalen z (Form (draws, n)); mean der Form (n,) oder (draws, n)."""
    return np.asarray(mean) + normals @ sqrt_factor(cov).T


def coverage_and_z2(mean: np.ndarray, cov: np.ndarray, x: np.ndarray, z_crit: float = C.BAND_Z) -> tuple[float, float]:
    """(Anteil der Punkte mit Wahrheit im Intervall mean +- z_crit sd, mittleres z^2) über alle Zeilen von mean."""
    sd = np.sqrt(np.maximum(np.diag(cov), _TINY))
    z = (np.atleast_2d(mean) - np.atleast_2d(x)) / sd[None, :]
    return float(np.mean(np.abs(z) <= z_crit)), float(np.mean(z ** 2))
