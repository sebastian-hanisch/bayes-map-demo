"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: Matrizen der Größe n <= 96 (Cholesky und Lösen in Mikrosekunden bis Millisekunden); die Experimente nutzen feste Ströme.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
TRUTHS = ("phantom", "prior")
TRUTH_LABELS = {"phantom": "Phantom (zwei glatte Hügel und ein Kasten)", "prior": "Aus dem Prior gezogen (das Modell stimmt)"}
TRUTH_SHORT = {"phantom": "Phantom", "prior": "Prior-Ziehung"}
FAMILIES = ("white", "se", "m32", "exp")
FAMILY_LABELS = {"white": "Weiß (unabhängige Werte, Tikhonov)", "se": "Quadratisch-exponentiell (sehr glatt)", "m32": "Matérn 3/2 (einmal differenzierbar)",
                 "exp": "Exponentiell (Ornstein-Uhlenbeck, rau)"}
FAMILY_SHORT = {"white": "weiß", "se": "SE", "m32": "Matérn 3/2", "exp": "exponentiell"}
ELL_CHOICES = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4)      # Korrelationslänge (Anteil der Länge 1)
DEFAULT_ELL = 0.05
SIGMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)                        # angenommenes σ als Vielfaches des wahren
DEFAULT_SFACTOR = 1.0
SEED_MAX, DEFAULT_SEED = 999999, 11
JITTER = 1e-10                                                    # Zuschlag auf die Diagonale vor der Zerlegung der Prior-Kovarianz
BAND_Z = 1.96                                                     # 95-%-Intervall
NOMINAL_COVERAGE = 0.95
OK_COVERAGE, OK_Z2, WIDE_Z2 = 0.80, 3.0, 0.4                       # Rahmen einer einzelnen Stichprobe (48 korrelierte Punkte) bei kalibriertem Posterior
DRAWS_SHOWN = 4                                                   # Ziehungen aus dem Posterior in der Primäransicht
SINGULAR_COND = 1e15                                              # ab hier ist die Matrix in doppelter Genauigkeit numerisch singulär
SHOW_RANGE = (-1.5, 2.5)                                          # Wertebereich der Rekonstruktion im Diagramm
FAIL_FACTOR = 2.0                                                 # „gescheitert“: Fehler über dem Doppelten des besten Fehlers


def tau_of(quarter_decades: int) -> float:
    """τ = 10^(quarter_decades / 4), auf sechs Stellen gerundet (damit Permalink und Voreinstellungen den Auswahlwert exakt treffen)."""
    return float(f"{10.0 ** (quarter_decades / 4.0):.6g}")


TAU_CHOICES = tuple(tau_of(k) for k in range(-12, 5))             # 1e-3 bis 10 in Viertel-Zehnerpotenzen
DEFAULT_TAU = tau_of(-3)                                          # 0,177828

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
EXP_REPS = 200
EXP_REPS_SMALL = 100                                              # gemeinsame Suche über (τ, σ) und (ℓ, τ)
EXP_DRAWS = 400                                                   # Ziehungen aus dem Prior je Zelle
EXP_CAL_TAUS = (0.1, 0.3, 1.0, 3.0)                               # Experiment 1: SE-Prior mit ℓ = 0,05
EXP_CAL_ELL = 0.05
EXP_LAMS = (1e-4, 1e-3, 1e-2, 1e-1)                               # Experiment 2: Tikhonov
EXP_TSVD_RANKS = (8, 12, 16, 24)                                  # Experiment 2
EVID_TAU_GRID = (-3.0, 1.0, 41)                                   # Experiment 3: Zehnerlogarithmen von τ
EVID_ELL_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)
EVID_TAU_COARSE = (-1.0, 1.0, 9)
EVID_SIGMA_GRID = (-1.0, 1.0, 13)                                 # σ̂ / σ von 0,1 bis 10
EVID_TAU_SIGMA = (-3.0, 1.0, 25)

COLOR_TRUTH = "#2F6B65"
COLOR_MAP = "#4c78a8"
COLOR_BAND = "rgba(76, 120, 168, 0.18)"
COLOR_DRAW = "rgba(76, 120, 168, 0.35)"
COLOR_TIK = "#555555"
COLOR_NAIVE = "#c0392b"
COLOR_GREY = "#8a8a8a"
COLOR_EVID = "#D68A2E"

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "truth": "phantom", "family": "se", "tau": tau_of(-3), "ell": 0.03,
                 "sfactor": 1.0, "seed": DEFAULT_SEED},
    "Prior-Ziehung": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "truth": "prior", "family": "se", "tau": 1.0, "ell": 0.05,
                      "sfactor": 1.0, "seed": DEFAULT_SEED},
    "Zu selbstsicher": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "truth": "phantom", "family": "se", "tau": 1.0, "ell": 0.05,
                        "sfactor": 1.0, "seed": DEFAULT_SEED},
    "Kalibrierter weißer Prior": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "truth": "phantom", "family": "white", "tau": tau_of(-2),
                                  "ell": DEFAULT_ELL, "sfactor": 1.0, "seed": DEFAULT_SEED},
    "σ unterschätzt": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "truth": "phantom", "family": "se", "tau": tau_of(-3), "ell": 0.03,
                       "sfactor": 0.25, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Phantom mit dem Prior, der den kleinsten Fehler gibt (SE, ℓ = 0,03, τ = 0,18): der Fehler liegt bei rund 12 %, die 95-%-Intervalle decken aber nur rund drei Viertel der Punkte ab.",
    "Prior-Ziehung": "Die Wahrheit stammt aus dem Prior: im Mittel über viele Ziehungen decken die 95-%-Intervalle 95 % ab und das mittlere z² ist 1; eine einzelne Ziehung schwankt darum.",
    "Zu selbstsicher": "Phantom mit SE-Prior (ℓ = 0,05, τ = 1): Fehler rund 12 %, aber die Intervalle sind um ein Vielfaches zu eng.",
    "Kalibrierter weißer Prior": "Weißer Prior mit τ nahe dem Effektivwert der Wahrheit (0,375; nächste Stufe 0,32): der richtige Prior ist hier nicht der beste, die Intervalle sind hier eher zu breit, der Fehler liegt im Mittel bei rund 20 %.",
    "σ unterschätzt": "Dasselbe mit einem um den Faktor 4 zu klein angenommenen σ: das Modell hält sich für genauer als die Daten.",
}
