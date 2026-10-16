"""Bayes-Sicht: MAP, Posterior und Evidenz im linear-gaußschen Fall - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 6 (Phantom, Gauß-Unschärfe, Rauschen) wird die Strafe als Prior gelesen:
Das Posterior-Mittel ist das MAP, der Posterior gibt Unsicherheit, und die Evidenz wählt die Prior-Parameter aus den Daten. Der Gaußsche Fall ist exakt lösbar; das
ist der Maßstab für Stück 8 bis 10. Kern: bys_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bys_constants as C
from bys_evaluation import (
    Settings,
    analyse,
    calibration_experiment,
    de,
    err_pct,
    evidence_experiment,
    mmse_experiment,
    noise_label,
    num_text,
    sci,
    share,
    tau_text,
    verdict_calibration,
    verdict_evidence,
    verdict_instance,
    verdict_mmse,
)
from bys_pdf_export import build_pdf
from bys_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bys_visualization import (
    build_calibration_chart,
    build_evidence_chart,
    build_mmse_chart,
    build_recon_chart,
    build_sd_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Bayes-Sicht: MAP und Posterior – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _calibration():
    return calibration_experiment()


@st.cache_data(show_spinner=SPINNER)
def _mmse():
    return mmse_experiment()


@st.cache_data(show_spinner=SPINNER)
def _evidence():
    return evidence_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎲 Bayes-Sicht: Was das MAP verschweigt")
st.markdown(
    """
**Der Strafterm ist ein Prior, und der Posterior sagt, wie sicher man sein darf.** Dies ist das siebte Stück der Linie **Inverse Probleme und Regularisierung** und baut auf
**Tikhonov** und **Sparsity und TV** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen. Mit einem Gaußschen Prior und Gaußschem Rauschen ist der Posterior
**exakt** bekannt: sein Mittel ist das MAP (für den weißen Prior die Tikhonov-Lösung), seine Kovarianz die Unsicherheit, seine Evidenz ein Maß, wie gut der Prior zu den Daten
passt. Die Frage ist, **ob man dem Posterior trauen darf**, wenn die Wahrheit nicht aus dem Prior stammt. Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was ein Prior, ein Posterior und die Evidenz sind", expanded=True):
    st.markdown(
        r"""
1. **Prior.** $x \sim \mathcal N(0, \tau^2 C)$: die Annahme über die Lösung vor den Daten. $C$ legt die Gestalt fest (weiß, glatt, rau), $\tau$ die Größe.
2. **Posterior.** Mit $b = Ax + e$, $e \sim \mathcal N(0, \sigma^2 I)$ ist $x \mid b$ wieder Gaußsch: Mittel $m$ (zugleich das **MAP**) und Kovarianz $\Gamma$. Für $C = I$ ist $m$
   die Tikhonov-Lösung mit $\lambda = \sigma/\tau$ (Stück 2).
3. **Intervalle.** $m_i \pm 1{,}96\,\sqrt{\Gamma_{ii}}$ soll die Wahrheit mit 95 % Wahrscheinlichkeit enthalten, *wenn die Wahrheit aus dem Prior stammt*. Das **mittlere $z^2$** der
   standardisierten Fehler $(m_i - x_i)/\sqrt{\Gamma_{ii}}$ soll dann 1 sein.
4. **Evidenz.** $p(b \mid \tau, \sigma, C)$: wie wahrscheinlich die Daten unter dem Prior sind. Ihr Maximum über $\tau$ ist *empirisches Bayes*, eine Parameterwahl wie in Stück 4.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Wahrheit**")
    truth = st.selectbox("Woher stammt die Wahrheit?", C.TRUTHS, key="truth_select", format_func=lambda v: C.TRUTH_LABELS[v],
                         help="Das Phantom ist die Welt der Linie. „Aus dem Prior gezogen“ ist die Welt, in der das Modell stimmt: die Daten sind A x + e mit x aus dem gewählten Prior.")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    data = st.selectbox("Woher kommen die Daten des Phantoms?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Nur für das Phantom: ehrliche Daten kommen aus der analytischen Unschärfe, die anderen aus der Matrix, die danach invertiert wird (inverse crime).")
    st.markdown("**Prior und Modell**")
    family = st.selectbox("Prior-Familie", C.FAMILIES, key="family_select", format_func=lambda v: C.FAMILY_LABELS[v],
                          help="Legt die Korrelation der Werte fest. „Weiß“ ist Tikhonov; die anderen sind Gauß-Prozesse mit der Korrelationslänge ℓ.")
    tau = st.select_slider("Prior-Standardabweichung τ", options=C.TAU_CHOICES, key="tau_select", format_func=tau_text,
                           help="Typische Größe der Werte unter dem Prior. Beim weißen Prior ist λ = σ/τ.")
    ell = st.select_slider("Korrelationslänge ℓ", options=C.ELL_CHOICES, key="ell_select", format_func=lambda v: de(v, 2),
                           help="Anteil der Länge 1, über den Werte korreliert sind. Beim weißen Prior ohne Wirkung.")
    sfactor = st.select_slider("Angenommenes σ (Vielfaches des wahren)", options=C.SIGMA_FACTORS, key="sfactor_select", format_func=lambda v: f"{de(v, 2)}×",
                               help="Mit 1 kennt das Modell das wahre Rauschen; mit 0,25 hält es sich für viermal genauer als die Daten.")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data, "truth_select": truth, "family_select": family,
                   "tau_select": float(tau), "ell_select": float(ell), "sfactor_select": float(sfactor), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, truth, family, float(tau), float(ell), float(sfactor), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.singular:
    st.warning(f"Bei dieser Wahl von n und w ist die Matrix in doppelter Genauigkeit numerisch singulär (Kondition {sci(a.cond)}): Die kleinen Singulärwerte sind "
               "Rundungsrauschen, und die Zahlen unten sind nur noch Zahlen.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fehler des Posterior-Mittels (MAP)", err_pct(a.err), delta=f"Tikhonov, bester λ: {err_pct(a.tik_err)}", delta_color="off",
          help="‖m − x‖ / ‖x‖ mit der Wahrheit x; Tikhonov wählt das λ mit dem kleinsten Fehler dieser Stichprobe.")
c2.metric("Abdeckung des 95-%-Intervalls", share(a.coverage), delta="Soll: 95 %", delta_color="off", help="Anteil der Gitterpunkte, an denen die Wahrheit im Intervall m ± 1,96 sd liegt.")
c3.metric("Mittleres z²", num_text(a.z2), delta=f"Intervalle um Faktor {num_text(a.overconf)} zu eng" if a.z2 > C.OK_Z2 else "Soll: 1", delta_color="off",
          help="Mittel von ((m − x)/sd)²; bei kalibriertem Posterior 1, deutlich darüber heißt überzuversichtlich.")
c4.metric("Log-Evidenz", de(a.log_evidence, 1), delta=f"Maximum bei τ̂ = {tau_text(a.tau_hat)}", delta_color="off", help="log p(b | τ, σ, Prior); größer heißt: die Daten passen besser zum Prior.")

tab_rec, tab_sd, tab_ev, tab_tab = st.tabs(["Rekonstruktion", "Unsicherheit", "Evidenz", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, blau das Posterior-Mittel mit 95-%-Band und vier exakten Ziehungen aus dem Posterior, punktiert Tikhonov beim besten λ dieser Stichprobe. Die Achse ist fest.")
with tab_sd:
    st.plotly_chart(build_sd_chart(a), width="stretch", key="sd_chart")
    st.caption("Die Standardabweichung des Posteriors hängt nur vom Modell ab (nicht von den Daten); der tatsächliche Fehler liegt auf dem Phantom weit darüber, wo die Annahmen nicht stimmen.")
with tab_ev:
    st.plotly_chart(build_evidence_chart(a), width="stretch", key="evidence_chart")
    st.caption(f"Die Evidenz über τ für die gewählte Familie. Beim Evidenz-Maximum τ̂ = {tau_text(a.tau_hat)} liegt der Fehler dieser Stichprobe bei {err_pct(a.err_hat)}.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="bayes-map.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Darf man dem Posterior trauen?")
verdict(verdict_instance(a))
st.caption("Eine einzelne Stichprobe über 48 korrelierte Punkte schwankt: bei Wahrheit aus dem Prior liegt die Abdeckung meist zwischen 85 und 100 %. Die Experimente mitteln.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wann ist der Posterior kalibriert?")
res1 = _calibration()
st.caption(f"SE-Prior mit ℓ = {de(res1['ell'], 2)}, n = {res1['n']}, Rauschen 0,1 %; Phantom: {res1['reps']} feste Wiederholungen des Rauschens, Prior-Ziehungen: {res1['draws']} feste Ziehungen. "
           "Abdeckung des 95-%-Intervalls und mittleres z² über alle Gitterpunkte.")
st.plotly_chart(build_calibration_chart(res1), width="stretch", key="calibration_chart")
verdict(verdict_calibration(res1))
st.dataframe(
    [{"τ": de(r["tau"], 1), "Phantom: Abdeckung": share(r["phantom"]["coverage"]), "Phantom: z²": num_text(r["phantom"]["z2"]), "Phantom: Fehler (RMS)": err_pct(r["phantom"]["rms"]),
      "Prior-Ziehung: Abdeckung": share(r["prior"]["coverage"]), "Prior-Ziehung: z²": num_text(r["prior"]["z2"]), "Prior-Ziehung: Fehler (RMS)": err_pct(r["prior"]["rms"]),
      "mittlere sd (Phantom)": num_text(r["phantom"]["sd"])} for r in res1["rows"]],
    hide_index=True, width="stretch")
st.caption("Der Fehler auf dem Phantom ist gut, die Intervalle sind es nicht: Der Posterior kennt nur sein Modell. Ein größeres τ macht die Intervalle breiter, ändert aber das "
           "Missverhältnis kaum, weil das Phantom (Kasten, Spitzen) nichts mit einem glatten Gauß-Prozess zu tun hat.")

st.subheader("🔬 Experiment 2: Das Posterior-Mittel ist optimal – im Modell")
res2 = _mmse()
st.caption(f"Wahrheit aus dem Prior (SE, ℓ = {de(res2['ell'], 2)}, τ = {de(res2['tau'], 0)}), n = {res2['n']}, {res2['draws']} feste Ziehungen: mittlerer quadratischer Fehler ‖m − x‖².")
st.plotly_chart(build_mmse_chart(res2), width="stretch", key="mmse_chart")
verdict(verdict_mmse(res2))
st.dataframe([{"Schätzer": r["label"], "mittlerer quadratischer Fehler": num_text(r["mse"]), "Vielfaches des Bayes-Risikos": de(r["mse"] / res2["bayes"], 1)} for r in res2["rows"]],
             hide_index=True, width="stretch")
st.caption("Das Posterior-Mittel minimiert den mittleren quadratischen Fehler unter dem Prior; das Bayes-Risiko ist spur Γ. Jede andere Regularisierung, auch mit dem besten λ, "
           "liegt darüber.")

st.subheader("🔬 Experiment 3: Die Evidenz wählt den Prior – wenn die Familie stimmt")
res3 = _evidence()
st.caption(f"Phantom, n = {res3['n']}, Rauschen 0,1 %, {res3['reps']} feste Wiederholungen (gemeinsame Suchen: {res3['joint']['reps']}): τ (und σ, ℓ) aus dem Maximum der Evidenz.")
verdict(verdict_evidence(res3))
w3, j3, s3 = res3["white"], res3["joint"], res3["se"]
st.dataframe(
    [{"Prior": "weiß, τ aus der Evidenz (σ bekannt)", "gewählt": f"τ̂ = {tau_text(w3['tau_median'])} (5/95 %: {tau_text(w3['tau_q'][0])} / {tau_text(w3['tau_q'][1])})",
      "Median-Fehler": err_pct(w3["median"]), "RMS": err_pct(w3["rms"]), "Optimum über τ": err_pct(w3["opt"]), "gescheitert": share(w3["fail"])},
     {"Prior": "weiß, τ und σ aus der Evidenz", "gewählt": f"σ̂/σ = {de(j3['sigma_ratio_median'], 2)} (5/95 %: {de(j3['sigma_ratio_q'][0], 2)} / {de(j3['sigma_ratio_q'][1], 2)})",
      "Median-Fehler": err_pct(j3["median"]), "RMS": err_pct(j3["rms"]), "Optimum über τ": "–", "gescheitert": "–"},
     {"Prior": "SE, (ℓ, τ) aus der Evidenz (σ bekannt)", "gewählt": f"ℓ̂ = {de(s3['ell_median'], 2)}, τ̂ = {tau_text(s3['tau_median'])}", "Median-Fehler": err_pct(s3["median"]),
      "RMS": err_pct(s3["rms"]), "Optimum über τ": "–", "gescheitert": "–"}],
    hide_index=True, width="stretch")
st.caption("Die Evidenz bewertet, wie gut der Prior die **Daten** beschreibt, nicht, wie klein der Fehler wird. Mit weißem Prior wählt sie τ nahe dem Effektivwert der Wahrheit, "
           "und der Fehler ist doppelt so groß wie das Optimum; mit der passenden glatten Familie trifft sie das Optimum.")

st.subheader("🔬 Experiment 4: Prior und Strafterm im Wörterbuch")
st.caption("Der negative Logarithmus des Posteriors ist eine Zielfunktion; das MAP minimiert sie. Die Tabelle zeigt, welcher Prior zu welchem Strafterm aus den früheren Stücken führt.")
st.dataframe(
    [{"Prior": "N(0, τ² I)", "Strafterm (MAP)": "½‖Ax − b‖² + (σ²/2τ²)‖x‖²", "Verfahren": "Tikhonov, λ = σ/τ", "Stück": "2"},
     {"Prior": "N(0, τ² C)", "Strafterm (MAP)": "½‖Ax − b‖² + (σ²/2τ²) xᵀ C⁻¹ x", "Verfahren": "verallgemeinerte Tikhonov-Regularisierung", "Stück": "2"},
     {"Prior": "Laplace(b) je Wert", "Strafterm (MAP)": "½‖Ax − b‖² + (σ²/b)‖x‖₁", "Verfahren": "L1, λ = σ²/b", "Stück": "6"},
     {"Prior": "Laplace(b) je Sprung", "Strafterm (MAP)": "½‖Ax − b‖² + (σ²/b)‖Dx‖₁", "Verfahren": "Total Variation", "Stück": "6"},
     {"Prior": "beliebig, log-konkav", "Strafterm (MAP)": "½‖Ax − b‖² − σ² log π(x)", "Verfahren": "MAP; Mittel und Kovarianz nur per Sampling", "Stück": "8"}],
    hide_index=True, width="stretch")
st.caption(f"Kontrolle an dieser Stichprobe (weißer Prior, τ = {tau_text(settings.tau)}): größte Abweichung zwischen Posterior-Mittel und Tikhonov-Lösung mit λ = σ/τ: {sci(a.map_gap)}. "
           "Außerhalb des Gaußschen Falls ist das MAP nicht der Bayes-Schätzer: Das Posterior-Mittel eines Laplace-Priors ist nicht dünn besetzt. Das zeigt das Sampling in Stück 8.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
mid = {r["tau"]: r for r in res1["rows"]}[1.0]
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Wahrheit stammt aus dem Prior** | Auf dem Phantom decken die 95-%-Intervalle {share(mid['phantom']['coverage'])} ab (z² = {num_text(mid['phantom']['z2'])}) statt {share(mid['prior']['coverage'])}. | Stück 16 (Abdeckungsrate als Maß) |
| **Die Prior-Familie passt** | Mit weißem Prior wählt die Evidenz τ̂ = {tau_text(res3['white']['tau_median'])} und liegt bei {err_pct(res3['white']['median'])} statt {err_pct(res3['white']['opt'])}. | – |
| **σ ist bekannt** | Ein viermal zu kleines σ senkt die Abdeckung weiter (siehe Voreinstellung „σ unterschätzt“); gemeinsam mit τ geschätzt liegt σ̂/σ bei {de(res3['joint']['sigma_ratio_median'], 2)}. | – |
| **Linear und Gaußsch** | Nur hier sind Mittel und Kovarianz geschlossen; sonst Sampling (Stück 8) oder Näherung (Stück 10). | Stück 8, 10 |
| **MAP ist der Schätzer** | Für nicht-Gaußsche Priors ist das MAP nicht das Posterior-Mittel. | Stück 8 |
| **Stationäre Priors auf dem Gitter** | Kasten und Spitzen verlangen ortsabhängige Glätte; keine Mischung oder Sprung-Priors hier. | Stück 6 (Strafe), 13 (Denoiser als Prior) |
| **Ein Phantom, eine Matrix** | Die Zahlen gelten für dieses Beispiel. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $x \sim \mathcal N(0, \tau^2 C)$, $b = Ax + e$, $e \sim \mathcal N(0, \sigma^2 I)$. Mit $K = \tau^2 A C A^\top + \sigma^2 I$ ist der Posterior $\mathcal N(m, \Gamma)$,
$$m = \tau^2 C A^\top K^{-1} b, \qquad \Gamma = \tau^2 C - \tau^4 C A^\top K^{-1} A C = \Big(\tfrac{1}{\sigma^2}A^\top A + \tfrac{1}{\tau^2}C^{-1}\Big)^{-1}.$$

**MAP.** $m = \arg\min_x \tfrac{1}{2\sigma^2}\|Ax - b\|^2 + \tfrac{1}{2\tau^2}x^\top C^{-1}x$. Für $C = I$ ist das die Tikhonov-Lösung mit $\lambda = \sigma/\tau$.

**Evidenz.** $\log p(b) = -\tfrac12\big(\log\det(2\pi K) + b^\top K^{-1} b\big)$; ihr Maximum über $\tau$ (und $\sigma$, $\ell$) ist die Wahl des empirischen Bayes.

**Kalibrierung.** Stammt $x$ aus dem Prior, sind $z_i = (m_i - x_i)/\sqrt{\Gamma_{ii}}$ standardnormal: $P(|z_i| \le 1{,}96) = 0{,}95$ und $\mathbb E z_i^2 = 1$; der erwartete
quadratische Fehler ist $\mathrm{tr}\,\Gamma$ (das Bayes-Risiko, das kein anderer Schätzer unterbietet).

**Prior-Familien.** $C_{ij} = k(|s_i - s_j|)$ mit $k_{\mathrm{SE}}(d) = e^{-d^2/2\ell^2}$, $k_{3/2}(d) = (1 + \sqrt3 d/\ell)e^{-\sqrt3 d/\ell}$, $k_{\mathrm{exp}}(d) = e^{-d/\ell}$.

**Literatur.** Das Beispiel ist erfunden; die Bayessche Sicht inverser Probleme geht auf J. Kaipio und E. Somersalo (2005) und A. M. Stuart (2010) zurück, Gauß-Prozesse und die
Evidenz auf C. E. Rasmussen und C. K. I. Williams (2006) und D. J. C. MacKay (1992) (alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
