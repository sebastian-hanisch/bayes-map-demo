"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 6 (symmetrisch, Zeilensumme 1, Singulärwerte 1, 0,5, 0,1; Eigenvektoren u1 = (1,1,1)/sqrt3, u2 = (1,0,-1)/sqrt2,
    u3 = (1,-2,1)/sqrt6), rauschfreie Daten b = A x = (1,3, 1,4, 1,3) für x = (1, 2, 1). Prior N(0, tau^2 I) mit tau = 1, Rauschen sigma = 0,1 (lambda = 0,1).
    Handrechnung in der Eigenbasis (u1.b = 4/sqrt3 = 2,3094, u2.b = 0, u3.b = -0,2/sqrt6 = -0,08165):
    MAP: Faktoren s_i/(s_i^2 + 0,01) = (0,990099, 1,923077, 5,0): m = (1,153465, 1,653465, 1,153465).
    Posterior-Eigenwerte 1/(s_i^2/sigma^2 + 1/tau^2) = (1/101, 1/26, 1/2): Gamma_11 = 0,105864, Gamma_22 = 0,336634, Gamma_13 = 0,067403.
    Evidenz: K hat die Eigenwerte s_i^2 tau^2 + sigma^2 = (1,01, 0,26, 0,02): log p(b) = -1/2 [3 ln 2pi + ln(1,01 * 0,26 * 0,02) + 5,3333/1,01 + 0,006667/0,02] = -2,9391."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b
