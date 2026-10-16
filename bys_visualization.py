"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import bys_constants as C
from bys_evaluation import Analysis, de, err_pct, num_text, sci, tau_text


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Messwert b": a.b[:k],
        "Posterior-Mittel (MAP)": a.mean[:k],
        "Standardabweichung": a.sd[:k],
        "Untergrenze 95 %": a.lo[:k],
        "Obergrenze 95 %": a.hi[:k],
    })


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, Posterior-Mittel mit 95-%-Band, exakte Ziehungen, Tikhonov (bester λ) und die direkte Inversion; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="95-%-Intervall", hoverinfo="skip"))
    for i, d in enumerate(a.draws):
        fig.add_trace(go.Scatter(x=a.s, y=d, mode="lines", name="Ziehungen aus dem Posterior", line=dict(color=C.COLOR_DRAW, width=1), legendgroup="draw", showlegend=(i == 0),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_tik, mode="lines", name=f"Tikhonov, bester λ dieser Stichprobe ({sci(a.tik_lam)})", line=dict(color=C.COLOR_TIK, width=1.5, dash="dot"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.mean, mode="lines+markers", name="Posterior-Mittel (= MAP)", line=dict(color=C.COLOR_MAP, width=2.5), marker=dict(size=5),
                             hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak = float(np.max(np.abs(a.mean))) if np.all(np.isfinite(a.mean)) else float("inf")
    note = ""
    if peak > max(abs(lo), abs(hi)):
        pt = "unendlich" if not np.isfinite(peak) else num_text(peak)
        note = f"Das Posterior-Mittel reicht bis ±{pt} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten."
    return _fixed(fig, 400, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def build_sd_chart(a: Analysis) -> go.Figure:
    """Punktweise Standardabweichung des Posteriors gegen den tatsächlichen Fehler dieser Stichprobe."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.sd, mode="lines", name="Standardabweichung des Posteriors", line=dict(color=C.COLOR_MAP, width=3)))
    fig.add_trace(go.Scatter(x=a.s, y=np.abs(a.mean - a.x), mode="lines+markers", name="tatsächlicher Fehler |m − x|", line=dict(color=C.COLOR_NAIVE, width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=a.s, y=C.BAND_Z * a.sd, mode="lines", name="Halbbreite des 95-%-Intervalls", line=dict(color=C.COLOR_GREY, width=1.5, dash="dash")))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Betrag", type="log")
    return _fixed(fig, 360)


def build_evidence_chart(a: Analysis) -> go.Figure:
    fig = go.Figure()
    ev = a.evid_values - float(np.max(a.evid_values))
    fig.add_trace(go.Scatter(x=a.evid_taus, y=np.maximum(ev, -60.0), mode="lines", name="log p(b | τ) − Maximum", line=dict(color=C.COLOR_EVID, width=3)))
    fig.add_vline(x=a.settings.tau, line=dict(color=C.COLOR_MAP, width=2), annotation_text=f"gewählt: τ = {tau_text(a.settings.tau)}", annotation_position="top left")
    fig.add_vline(x=a.tau_hat, line=dict(color=C.COLOR_GREY, width=2, dash="dash"), annotation_text=f"Evidenz-Maximum: τ̂ = {tau_text(a.tau_hat)}", annotation_position="top right")
    fig.update_xaxes(title="Prior-Standardabweichung τ", type="log", tickformat=".0e")
    fig.update_yaxes(title="Log-Evidenz relativ zum Maximum (unter −60 abgeschnitten)", range=[-62, 2])
    return _fixed(fig, 360)


def build_calibration_chart(res: dict) -> go.Figure:
    taus = [f"τ = {de(r['tau'], 1)}" for r in res["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=taus, y=[100.0 * r["prior"]["coverage"] for r in res["rows"]], name="Wahrheit aus dem Prior", marker_color=C.COLOR_MAP,
                         text=[f"{100 * r['prior']['coverage']:.0f} %" for r in res["rows"]], textposition="outside"))
    fig.add_trace(go.Bar(x=taus, y=[100.0 * r["phantom"]["coverage"] for r in res["rows"]], name="Phantom", marker_color=C.COLOR_NAIVE,
                         text=[f"{100 * r['phantom']['coverage']:.0f} %" for r in res["rows"]], textposition="outside"))
    fig.add_hline(y=100.0 * C.NOMINAL_COVERAGE, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="nominal 95 %", annotation_position="bottom left")
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="Abdeckung der 95-%-Intervalle (%)", range=[0, 115])
    return _fixed(fig, 340)


def build_mmse_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    fig = go.Figure()
    colors = [C.COLOR_MAP] + [C.COLOR_TIK] * (len(rows) - 1)
    fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r["mse"] for r in rows], marker_color=colors, text=[num_text(r["mse"]) for r in rows], textposition="outside"))
    fig.add_hline(y=res["trace"], line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="spur Γ (Bayes-Risiko)", annotation_position="top right")
    fig.update_yaxes(title="Mittlerer quadratischer Fehler ‖m − x‖² (logarithmisch)", type="log")
    fig.update_xaxes(tickangle=-30)
    return _fixed(fig, 380, showlegend=False)
