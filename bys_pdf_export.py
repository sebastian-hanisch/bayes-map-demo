"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import bys_constants as C
from bys_evaluation import Analysis, de, err_pct, share, sci, tau_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0", "¹": "1", "²": "2",
            "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Bayes-Sicht: MAP und Posterior"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    kind = "Daten aus demselben Gitter (inverse crime)" if s.data == "verbrechen" else "ehrliche Daten (analytisch)"
    kind = "Wahrheit aus dem Prior gezogen (Daten A x + e)" if s.truth == "prior" else f"Phantom, {kind}"
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, {kind}, "
                            f"Prior: {C.FAMILY_SHORT[s.family]}, tau = {tau_text(s.tau)}, ell = {de(s.ell, 2)}, angenommenes sigma = {de(s.sfactor, 2)} * wahres sigma, "
                            f"Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Posterior-Mittel (= MAP): Fehler dieser Stichprobe {err_pct(a.err)}; Tikhonov beim besten lambda dieser Stichprobe {err_pct(a.tik_err)}; direkte Inversion {err_pct(a.naive_err)}.",
        f"Kalibrierung dieser Stichprobe: 95-%-Intervalle decken {share(a.coverage)} der Gitterpunkte ab, mittleres z^2 = {de(a.z2, 2)}, mittlere sd {sci(a.mean_sd)}.",
        f"Log-Evidenz {de(a.log_evidence, 2)}; Evidenz-Maximum bei tau = {tau_text(a.tau_hat)} mit Fehler {err_pct(a.err_hat)}.",
        f"Kondition der Matrix: {sci(a.cond)}; wahres Rauschen sigma = {sci(a.sigma)}, angenommen {sci(a.sigma_assumed)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Messwert b", "Posterior-Mittel", "Std.-abw."], [16, 26, 30, 34, 38, 36]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.b[i], 4), de(a.mean[i], 3), de(a.sd[i], 4)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("MAP, Posterior und Evidenz im linear-gaußschen Fall. Quelle: bayes-map-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
