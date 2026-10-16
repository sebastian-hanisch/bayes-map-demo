"""Auswertung: Wahrheiten, Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bys_constants as C
import bys_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def tau_text(tau: float) -> str:
    """τ mit drei gültigen Stellen (ab 0,01 als Dezimalzahl, darunter als Zehnerpotenz)."""
    return de(tau, 3) if tau >= 0.01 else sci(tau, 2)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


def num_text(x: float) -> str:
    """Zahl mit drei gültigen Stellen, ab 1000 als Zehnerpotenz."""
    if not math.isfinite(x):
        return "unendlich"
    if abs(x) >= 1000.0 or (x != 0.0 and abs(x) < 0.01):
        return sci(x, 1)
    return de(x, 1 if abs(x) >= 10 else 2)


# ----------------------------------------------------------------------------- Instanz

class Problem:
    """Matrix, SVD, Phantom und seine analytische Antwort, absolutes Rauschen."""

    def __init__(self, n: int, w: float, noise: float):
        self.n, self.w, self.noise = n, w, noise
        self.a = operator(n, w)
        self.u, self.sv, self.vt = model.decompose(self.a)
        self.s = grid(n)
        self.x_phantom = phantom(self.s)
        self.g = blurred_exact(self.s, w)
        self.sigma = float(noise) * float(np.sqrt(np.mean(self.g ** 2)))
        self.cond = float(self.sv[0] / self.sv[-1]) if self.sv[-1] > 0 else math.inf
        self.kernels: dict = {}

    def kernel(self, family: str, ell: float) -> np.ndarray:
        key = (family, ell if family != "white" else 0.0)
        if key not in self.kernels:
            self.kernels[key] = model.kernel(self.s, ell, family)
        return self.kernels[key]

    def noise_rows(self, reps: int, seed: int) -> np.ndarray:
        return SplitMix64(seed).normal(self.n * reps).reshape(reps, self.n) * self.sigma


@lru_cache(maxsize=32)
def problem(n: int, w: float, noise: float) -> Problem:
    return Problem(n, w, noise)


def prior_draws(p: Problem, family: str, ell: float, tau: float, draws: int, seed: int) -> np.ndarray:
    """Ziehungen x ~ N(0, tau^2 C), Form (draws, n)."""
    z = SplitMix64(seed).normal(p.n * draws).reshape(draws, p.n)
    return tau * (z @ model.sqrt_factor(p.kernel(family, ell)).T)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    truth: str = "phantom"
    family: str = "se"
    tau: float = C.DEFAULT_TAU
    ell: float = 0.03
    sfactor: float = C.DEFAULT_SFACTOR
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    mean: np.ndarray          # Posterior-Mittel = MAP
    sd: np.ndarray            # punktweise Standardabweichung des Posteriors
    lo: np.ndarray
    hi: np.ndarray
    draws: np.ndarray         # exakte Ziehungen aus dem Posterior, (DRAWS_SHOWN, n)
    err: float
    coverage: float           # Anteil der Gitterpunkte mit Wahrheit im 95-%-Intervall
    z2: float                 # mittleres z^2 (soll 1 sein)
    overconf: float           # sqrt(z2): Faktor, um den die Intervalle zu eng sind
    mean_sd: float
    tik_err: float            # Tikhonov bei dem λ mit dem kleinsten Fehler dieser Stichprobe
    tik_lam: float
    x_tik: np.ndarray
    naive_err: float
    log_evidence: float
    evid_taus: np.ndarray
    evid_values: np.ndarray   # log p(b) über τ für die gewählte Familie und ℓ
    tau_hat: float
    err_hat: float            # Fehler des Posterior-Mittels bei τ̂
    map_gap: float            # größte Abweichung MAP gegen Tikhonov (λ = σ/τ) bei weißem Prior
    sigma: float              # wahres absolutes Rauschen
    sigma_assumed: float
    cond: float
    singular: bool


def analyse(settings: Settings) -> Analysis:
    st = settings
    p = problem(st.n, st.w, st.noise)
    c = p.kernel(st.family, st.ell)
    e = p.noise_rows(1, st.seed)[0]
    if st.truth == "prior":
        x = prior_draws(p, st.family, st.ell, st.tau, 1, st.seed + 7919)[0]
        b = p.a @ x + e
    else:
        x = p.x_phantom
        b = (p.g if st.data != "verbrechen" else p.a @ x) + e
    sig = st.sfactor * p.sigma
    mean, cov = model.posterior(p.a, c, st.tau, sig, b)
    mean = mean[0]
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    normals = SplitMix64(st.seed + 31).normal(C.DRAWS_SHOWN * st.n).reshape(C.DRAWS_SHOWN, st.n)
    draws = model.draw_gaussian(mean, cov, normals)
    xn = float(np.linalg.norm(x))
    err = float(np.linalg.norm(mean - x) / xn)
    cov_frac, z2 = model.coverage_and_z2(mean, cov, x)
    lams = np.logspace(-6, 0, 61)
    tik = np.array([model.tikhonov(p.u, p.sv, p.vt, b, float(l))[0] for l in lams])
    te = np.linalg.norm(tik - x[None, :], axis=1) / xn
    it = int(np.argmin(te))
    lo, hi, pts = C.EVID_TAU_GRID
    taus = np.logspace(lo, hi, pts)
    ev = np.array([float(model.log_evidence(p.a, c, float(t), sig, b)[0]) for t in taus])
    tau_hat = float(taus[int(np.argmax(ev))])
    m_hat = model.posterior(p.a, c, tau_hat, sig, b)[0][0]
    cw = p.kernel("white", 0.0)
    m_w = model.posterior(p.a, cw, st.tau, sig, b)[0][0]
    t_w = model.tikhonov(p.u, p.sv, p.vt, b, sig / st.tau)[0]
    x_naive = (b @ p.u / p.sv) @ p.vt
    return Analysis(
        settings=st, s=p.s, x=x, b=b, mean=mean, sd=sd, lo=mean - C.BAND_Z * sd, hi=mean + C.BAND_Z * sd, draws=draws, err=err, coverage=cov_frac, z2=z2,
        overconf=math.sqrt(z2), mean_sd=float(sd.mean()), tik_err=float(te[it]), tik_lam=float(lams[it]), x_tik=tik[it], naive_err=float(np.linalg.norm(x_naive - x) / xn),
        log_evidence=float(model.log_evidence(p.a, c, st.tau, sig, b)[0]), evid_taus=taus, evid_values=ev, tau_hat=tau_hat,
        err_hat=float(np.linalg.norm(m_hat - x) / xn), map_gap=float(np.abs(m_w - t_w).max()), sigma=p.sigma, sigma_assumed=sig, cond=p.cond,
        singular=bool(p.cond >= C.SINGULAR_COND),
    )


# ----------------------------------------------------------------------------- Experiment 1: Kalibrierung

def calibration_experiment(reps: int = C.EXP_REPS, draws: int = C.EXP_DRAWS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    p = problem(n, w, noise)
    c = p.kernel("se", C.EXP_CAL_ELL)
    b_ph = p.g[None, :] + p.noise_rows(reps, C.SEED_BASE + 1)
    rows = []
    for tau in C.EXP_CAL_TAUS:
        m, cov = model.posterior(p.a, c, tau, p.sigma, b_ph)
        cov_p, z2_p = model.coverage_and_z2(m, cov, p.x_phantom)
        err_p = np.linalg.norm(m - p.x_phantom[None, :], axis=1) / float(np.linalg.norm(p.x_phantom))
        x = prior_draws(p, "se", C.EXP_CAL_ELL, tau, draws, C.SEED_BASE + 2)
        b = x @ p.a.T + p.noise_rows(draws, C.SEED_BASE + 3)
        m2, cov2 = model.posterior(p.a, c, tau, p.sigma, b)
        cov_d, z2_d = model.coverage_and_z2(m2, cov2, x)
        err_d = np.linalg.norm(m2 - x, axis=1) / np.linalg.norm(x, axis=1)
        rows.append({"tau": tau, "phantom": {"coverage": cov_p, "z2": z2_p, "rms": float(np.sqrt(np.mean(err_p ** 2))), "sd": float(np.sqrt(np.diag(cov)).mean())},
                     "prior": {"coverage": cov_d, "z2": z2_d, "rms": float(np.sqrt(np.mean(err_d ** 2))), "sd": float(np.sqrt(np.diag(cov2)).mean())}})
    return {"rows": rows, "reps": reps, "draws": draws, "n": n, "ell": C.EXP_CAL_ELL}


# ----------------------------------------------------------------------------- Experiment 2: MMSE-Optimalität

def mmse_experiment(draws: int = C.EXP_DRAWS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    p = problem(n, w, noise)
    tau = 1.0
    c = p.kernel("se", C.EXP_CAL_ELL)
    x = prior_draws(p, "se", C.EXP_CAL_ELL, tau, draws, C.SEED_BASE + 2)
    b = x @ p.a.T + p.noise_rows(draws, C.SEED_BASE + 3)
    m, cov = model.posterior(p.a, c, tau, p.sigma, b)
    bayes = float(np.mean(np.sum((m - x) ** 2, axis=1)))
    rows = [{"label": "Posterior-Mittel", "mse": bayes}]
    for lam in C.EXP_LAMS:
        t = model.tikhonov(p.u, p.sv, p.vt, b, lam)
        rows.append({"label": f"Tikhonov, λ = {sci(lam, 0)}", "mse": float(np.mean(np.sum((t - x) ** 2, axis=1)))})
    beta = b @ p.u
    for k in C.EXP_TSVD_RANKS:
        y = np.zeros_like(beta)
        y[:, :k] = beta[:, :k] / p.sv[:k]
        t = y @ p.vt
        rows.append({"label": f"TSVD, Rang {k}", "mse": float(np.mean(np.sum((t - x) ** 2, axis=1)))})
    others = [r["mse"] for r in rows[1:]]
    return {"rows": rows, "bayes": bayes, "trace": float(np.trace(cov)), "best_other": min(others), "draws": draws, "n": n, "tau": tau, "ell": C.EXP_CAL_ELL}


# ----------------------------------------------------------------------------- Experiment 3: empirisches Bayes

def _white_evidence(p: Problem, beta: np.ndarray, tau, sigma) -> np.ndarray:
    """log p(b) für den weißen Prior über die SVD: Eigenwerte tau^2 sv^2 + sigma^2; tau und sigma als Felder gleicher Form (…, 1)."""
    k = (tau ** 2) * p.sv[None, :] ** 2 + sigma ** 2
    return -0.5 * np.sum(np.log(2.0 * math.pi * k) + beta ** 2 / k, axis=-1)


def _eig_evidence(evals: np.ndarray, gamma: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """log p(b) über die Eigenzerlegung von A C A^T = Q diag(evals) Q^T und gamma = b Q: Eigenwerte tau^2 evals + sigma^2 (Zeilen von gamma)."""
    k = tau ** 2 * np.maximum(evals, 0.0)[None, :] + sigma ** 2
    return -0.5 * np.sum(np.log(2.0 * math.pi * k) + gamma ** 2 / k, axis=1)


def evidence_experiment(reps: int = C.EXP_REPS, reps_small: int = C.EXP_REPS_SMALL, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    p = problem(n, w, noise)
    xn = float(np.linalg.norm(p.x_phantom))
    b = p.g[None, :] + p.noise_rows(reps, C.SEED_BASE + 1)
    beta = b @ p.u
    lo, hi, pts = C.EVID_TAU_GRID
    taus = np.logspace(lo, hi, pts)
    c_w = p.kernel("white", 0.0)
    # weißer Prior, τ aus der Evidenz (σ bekannt)
    ev = np.array([_white_evidence(p, beta, t, p.sigma) for t in taus])          # (τ, reps)
    pick = np.argmax(ev, axis=0)
    err_tau = np.array([np.linalg.norm(model.posterior(p.a, c_w, float(t), p.sigma, b)[0] - p.x_phantom[None, :], axis=1) / xn for t in taus])   # (τ, reps)
    e_hat = err_tau[pick, np.arange(reps)]
    opt = float(np.sqrt(np.mean(err_tau ** 2, axis=1)).min())
    white = {"tau_median": float(np.median(taus[pick])), "tau_q": [float(q) for q in np.quantile(taus[pick], [0.05, 0.95])], "median": float(np.median(e_hat)),
             "rms": float(np.sqrt(np.mean(e_hat ** 2))), "opt": opt, "fail": float(np.mean(e_hat > C.FAIL_FACTOR * opt)), "tau_opt": float(taus[int(np.argmin(np.sqrt(np.mean(err_tau ** 2, axis=1))))])}
    # weißer Prior, τ und σ gemeinsam
    bs = b[:reps_small]
    bsb = bs @ p.u
    lo2, hi2, pts2 = C.EVID_TAU_SIGMA
    t2 = np.logspace(lo2, hi2, pts2)
    lo3, hi3, pts3 = C.EVID_SIGMA_GRID
    s2 = p.sigma * np.logspace(lo3, hi3, pts3)
    tt, ss = np.meshgrid(t2, s2, indexing="ij")
    vals = np.array([_white_evidence(p, bsb[r][None, None, :], tt[..., None], ss[..., None]) for r in range(reps_small)])      # (reps, τ, σ)
    flat = vals.reshape(reps_small, -1).argmax(axis=1)
    ti, si = np.unravel_index(flat, tt.shape)
    errs2 = np.array([float(np.linalg.norm(model.posterior(p.a, c_w, float(t2[i]), float(s2[j]), bs[r:r + 1])[0][0] - p.x_phantom) / xn) for r, (i, j) in enumerate(zip(ti, si))])
    ratio = s2[si] / p.sigma
    joint = {"sigma_ratio_median": float(np.median(ratio)), "sigma_ratio_q": [float(q) for q in np.quantile(ratio, [0.05, 0.95])], "median": float(np.median(errs2)),
             "rms": float(np.sqrt(np.mean(errs2 ** 2))), "reps": reps_small, "count": int(len(errs2))}
    # SE-Prior, (ℓ, τ) aus der Evidenz (σ bekannt)
    lo4, hi4, pts4 = C.EVID_TAU_COARSE
    t4 = np.logspace(lo4, hi4, pts4)
    best = np.full(reps_small, -np.inf)
    arg = np.zeros((reps_small, 2))
    for ell in C.EVID_ELL_GRID:
        c = p.kernel("se", ell)
        evals, q = np.linalg.eigh(p.a @ c @ p.a.T)
        gamma = bs @ q
        for t in t4:
            v = _eig_evidence(evals, gamma, float(t), p.sigma)
            better = v > best
            best = np.where(better, v, best)
            arg[better] = (ell, t)
    errs3 = np.array([float(np.linalg.norm(model.posterior(p.a, p.kernel("se", float(arg[r, 0])), float(arg[r, 1]), p.sigma, bs[r:r + 1])[0][0] - p.x_phantom) / xn)
                      for r in range(reps_small)])
    se = {"ell_median": float(np.median(arg[:, 0])), "tau_median": float(np.median(arg[:, 1])), "median": float(np.median(errs3)), "rms": float(np.sqrt(np.mean(errs3 ** 2))),
          "reps": reps_small, "count": int(len(errs3))}
    return {"white": white, "joint": joint, "se": se, "reps": reps, "n": n, "taus": taus, "evidence_mean": ev.mean(axis=1)}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    st = a.settings
    lead = (f"{C.FAMILY_SHORT[st.family]}-Prior, τ = {tau_text(st.tau)}: Fehler {err_pct(a.err)}; die 95-%-Intervalle decken {share(a.coverage)} der Gitterpunkte ab, "
            f"das mittlere z² ist {num_text(a.z2)}.")
    if a.coverage < C.OK_COVERAGE or a.z2 > C.OK_Z2:
        return "warning", f"{lead} Der Posterior ist überzuversichtlich: die Intervalle sind um den Faktor {num_text(a.overconf)} zu eng."
    if a.z2 < C.WIDE_Z2:
        return "info", f"{lead} Der Posterior ist eher zu vorsichtig: die Intervalle sind um den Faktor {num_text(1.0 / max(a.overconf, 1e-12))} zu breit."
    return "success", (f"{lead} Das liegt im Rahmen dessen, was eine einzelne Stichprobe bei einem kalibrierten Posterior ergibt "
                       f"(Abdeckung ab {share(C.OK_COVERAGE)}, z² bis {num_text(C.OK_Z2)}).")


def verdict_calibration(res: dict) -> tuple[str, str]:
    r = {x["tau"]: x for x in res["rows"]}
    mid = r[1.0]
    return "warning", (f"SE-Prior, ℓ = {de(res['ell'], 2)}, τ = 1: Stammt die Wahrheit aus dem Prior, decken die 95-%-Intervalle {share(mid['prior']['coverage'])} ab (z² = {num_text(mid['prior']['z2'])}); "
                       f"auf dem Phantom {share(mid['phantom']['coverage'])} (z² = {num_text(mid['phantom']['z2'])}), obwohl der Fehler dort mit {err_pct(mid['phantom']['rms'])} klein ist. "
                       "Der Posterior ist nur im Modell kalibriert.")


def verdict_mmse(res: dict) -> tuple[str, str]:
    return "info", (f"Wahrheit aus dem Prior ({res['draws']} Ziehungen): das Posterior-Mittel erreicht den mittleren quadratischen Fehler {num_text(res['bayes'] * 1e4)}·10⁻⁴ "
                    f"(spur Γ = {num_text(res['trace'] * 1e4)}·10⁻⁴), das beste der anderen Verfahren {num_text(res['best_other'] * 1e4)}·10⁻⁴, also das {de(res['best_other'] / res['bayes'], 1)}-Fache.")


def verdict_evidence(res: dict) -> tuple[str, str]:
    w, j, s = res["white"], res["joint"], res["se"]
    return "warning", (f"Weißer Prior: die Evidenz wählt τ̂ = {tau_text(w['tau_median'])} (der beste τ liegt bei {tau_text(w['tau_opt'])}); Median-Fehler {err_pct(w['median'])} gegen "
                       f"das Optimum {err_pct(w['opt'])}, {share(w['fail'])} gescheitert. Mit σ mitgeschätzt: σ̂/σ im Median {de(j['sigma_ratio_median'], 2)}, Fehler {err_pct(j['median'])}. "
                       f"Passende Familie (SE): ℓ̂ im Median {de(s['ell_median'], 2)}, Median-Fehler {err_pct(s['median'])}.")
