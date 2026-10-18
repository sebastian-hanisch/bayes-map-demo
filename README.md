# Bayes-Sicht: Was das MAP verschweigt – Streamlit-Demo

Siebtes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo) und [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab.

## Kernfrage

Der Strafterm aus den früheren Stücken ist ein **Prior**. Im linear-gaußschen Fall ist der Posterior exakt bekannt: sein Mittel ist das MAP (für den weißen Prior die Tikhonov-Lösung), seine Kovarianz sagt, wie sicher man sein darf. Darf man ihm trauen, wenn die Wahrheit nicht aus dem Prior stammt, und kann man den Prior aus den Daten wählen?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen, zwei Datenarten):

- **Wahrheit** x(s) auf [0, 1]; **Messung** mit dem Gauß-Kern der Breite w, analytisch bekannt. Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴.
- **Prior** x ~ N(0, τ² C) mit der Korrelationsmatrix C einer Familie: weiß, quadratisch-exponentiell (SE), Matérn 3/2, exponentiell; Rauschen e ~ N(0, σ² I).
- **Posterior** in Datenraumformeln (K = τ² A C Aᵀ + σ² I): Mittel m = τ² C Aᵀ K⁻¹ b, Kovarianz Γ = τ² C − τ⁴ C Aᵀ K⁻¹ A C; **Evidenz** log p(b) = −½(log det 2πK + bᵀK⁻¹b).
- **Wahrheit:** das Phantom (ehrliche Daten oder aus demselben Gitter) oder eine **Ziehung aus dem Prior** (dann sind die Daten A x + e und das Modell stimmt).

Alles ist erfunden; die Wahrheit ist das simulierte Phantom oder die gezogene Funktion.

## Methodik

- **Kalibrierung:** Abdeckung des 95-%-Intervalls m ± 1,96 sd (Anteil der Gitterpunkte mit Wahrheit im Intervall) und das mittlere z² = E((m − x)²/Γᵢᵢ), das bei Kalibrierung 1 ist.
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 200 Wiederholungen des Rauschens und 400 Ziehungen aus dem Prior; gemeinsame Suchen über (τ, σ) und (ℓ, τ) mit je 100 Wiederholungen; nie ein einzelner Seed. Eine einzelne Stichprobe über 48 korrelierte Punkte schwankt (bei Wahrheit aus dem Prior liegt die Abdeckung meist zwischen 85 und 100 %).
- **Maßstab:** der Fehler ‖m − x‖/‖x‖ und Tikhonov beim besten λ derselben Stichprobe.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, ehrliche Daten), SE-Prior mit ℓ = 0,05, wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Ist das MAP die Tikhonov-Lösung? | Ja, mit λ = σ/τ: größte Abweichung unter 10⁻⁸ bei jeder Voreinstellung. | `test_map_equals_tikhonov_for_the_white_prior_at_every_preset` |
| Ist der Posterior kalibriert (Experiment 1)? | **Wahrheit aus dem Prior** (400 Ziehungen), τ = 0,1 / 0,3 / 1 / 3: Abdeckung **95,2 / 95,4 / 95,1 / 94,8 %**, z² 0,98 / 0,98 / 0,99 / 1,03. **Phantom** (200 Wdh.): Abdeckung **40 / 53 / 64 / 73 %**, z² **1.232 / 483 / 166 / 63**, bei einem Fehler von nur 13,6 / 12,7 / 12,2 / 11,7 %. Die Intervalle sind um den Faktor 8 bis 35 zu eng. | `test_calibration_numbers` |
| Ist das Posterior-Mittel optimal (Experiment 2)? | Wahrheit aus dem Prior (τ = 1, 400 Ziehungen): mittlerer quadratischer Fehler des Posterior-Mittels **8,5·10⁻⁴** (spur Γ = 8,6·10⁻⁴); das beste andere Verfahren (Tikhonov, λ = 10⁻²) **4,1·10⁻³**, also das 4,7-Fache; TSVD (Rang 24) 4,8·10⁻²; Tikhonov mit λ = 10⁻⁴ **12,9**. | `test_mmse_numbers` |
| Ist der richtige Prior der beste? | Nein: Weißer Prior mit τ gleich dem Effektivwert der Wahrheit (0,375): RMS **21,0 %**; der beste τ = 0,032 (λ = 0,011) gibt **11,7 %**. | `test_the_calibrated_white_prior_is_worse_than_the_best_one_in_the_mean` |
| Hilft ein glatter Prior? | Die Form ändert den Fehler kaum: bester RMS **11,6 %** (SE, ℓ = 0,03), **11,6 %** (Matérn 3/2, ℓ = 0,2), **11,7 %** (exponentiell, ℓ = 0,12) gegen 11,7 % beim weißen Prior. Der Boden kommt von den Daten. | `test_the_shape_of_the_prior_hardly_changes_the_error` |
| Wählt die Evidenz den Prior (Experiment 3)? | **Weißer Prior:** τ̂ = **0,40** in allen Wiederholungen (≈ Effektivwert der Wahrheit), Median-Fehler **21,5 %** (Optimum 11,7 %), **28 %** gescheitert (> Doppeltes des Optimums). Mit σ mitgeschätzt: σ̂/σ Median **0,68** (5 / 95 %: 0,46 / 1,47), Median-Fehler 26,9 %. **SE-Prior, (ℓ, τ) aus der Evidenz:** ℓ̂ = 0,03, Median-Fehler **11,7 %**. | `test_evidence_numbers` |
| Einzelstichproben (Presets) | Standard (SE, ℓ = 0,03, τ = 0,18): Fehler 11,6 %, Abdeckung 77 %, z² 20. Zu selbstsicher (SE, ℓ = 0,05, τ = 1): 12,1 %, **60 %**, z² 165. σ viermal zu klein: 12,7 %, **27 %**, z² 65. Kalibrierter weißer Prior: 16,5 %, Abdeckung 100 %, z² 0,3 (eher zu breit). Prior-Ziehung: 0,79 %, Abdeckung 85 %, z² 2,0. | `test_preset_help_numbers` |
| Mini-Instanz von Hand | 3 × 3-Matrix, τ = 1, σ = 0,1: MAP (1,153, 1,653, 1,153); Posterior-Eigenwerte 1/101, 1/26, 1/2; log-Evidenz **−2,939**. | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Unsicherheit ist das eigentliche Ergebnis.** MAP und Tikhonov sind dasselbe; der Posterior dazu ist es nicht, und er ist auf dem Phantom um den Faktor 8 bis 35 überzuversichtlich, während der Punktfehler gut ist.
- **Der richtige Prior ist nicht der beste.** Kalibrierung (τ = Effektivwert) kostet fast das Doppelte an Fehler (21,0 gegen 11,7 %); der beste λ entspricht einem Prior mit einem Zwölftel der Stärke der Wahrheit.
- **Die Evidenz ist keine Parameterwahl wie die aus Stück 4.** Sie bewertet die Daten unter dem Prior, nicht den Fehler; mit falscher Familie wählt sie den kalibrierten τ.
- **MAP ist nicht der Bayes-Schätzer außerhalb des Gaußschen Falls:** hier nur gesagt und mit dem Orakel „Laplace-Prior-MAP gleich L1“ belegt; die Messung MAP gegen Posterior-Mittel braucht das Sampling aus Stück 8 (ein naives Metropolis-Verfahren hatte in der Vormessung Annahmeraten von 0,00 bis 0,25 und war nicht belastbar).
- **Die Verdikt-Schwellen für eine einzelne Stichprobe** (Abdeckung ab 80 %, z² bis 3) stammen aus 200 gezogenen Wahrheiten im Modell (5-%-Quantil der Abdeckung 0,85, 95-%-Quantil von z² 1,9), nicht aus einer Konvention.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Wahrheit stammt aus dem Prior** | Auf dem Phantom decken die 95-%-Intervalle 64 % ab (z² = 166) statt 95 %. | Stück 16 (Abdeckungsrate als Maß) |
| **Die Prior-Familie passt** | Mit weißem Prior wählt die Evidenz τ̂ = 0,40 und liegt bei 21,5 % statt 11,7 %. | – |
| **σ ist bekannt** | Ein viermal zu kleines σ senkt die Abdeckung auf 27 %; gemeinsam mit τ geschätzt liegt σ̂/σ bei 0,68. | – |
| **Linear und Gaußsch** | Nur hier sind Mittel und Kovarianz geschlossen; sonst Sampling oder Näherung. | Stück 8, 10 |
| **MAP ist der Schätzer** | Für nicht-Gaußsche Priors ist das MAP nicht das Posterior-Mittel. | Stück 8 |
| **Stationäre Priors auf dem Gitter** | Kasten und Spitzen verlangen ortsabhängige Glätte. | Stück 6, 13 |
| **Ein Phantom, eine Matrix** | Die Zahlen gelten für dieses Beispiel. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo): Stück 2; das MAP des weißen Priors ist dessen Lösung, λ = σ/τ.
- [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo): Stück 6; Laplace-Prior und L1, Sprung-Prior und TV.
- [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo): Stück 4; die Evidenz ist eine weitere Regel zur Parameterwahl.
- [kalman-filter-demo](https://github.com/sebastian-hanisch/kalman-filter-demo): das sequentielle Update des Posteriors (im Orakeltest nachgerechnet).

## Tests

Mini-Instanz von Hand (3 × 3: Kernel, MAP, Posterior-Kovarianz, Evidenz, Grenzfälle, Abdeckung und z², Wurzel und Ziehungen), Datenraum- gegen Präzisionsmatrix-Formel, MAP gegen die Minimierung der negativen Log-Posterior, Evidenz gegen die scipy-Dichte und numerische Integration, Kalibrierung und Bayes-Risiko per Monte Carlo (4.000 Ziehungen), Optimalität des Posterior-Mittels gegen Störungen und Tikhonov, sequentielles gegen Gesamt-Update, Laplace-Prior-MAP gegen L1 (ISTA), Stichprobenkovarianz exakter Ziehungen, Generator, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jede Familie, Wahrheit und Datenart, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `bys_model.py` | Kernel, Posterior, Evidenz, exakte Ziehungen, Abdeckung und z² |
| `bys_evaluation.py` | Wahrheiten, Kennzahlen, Experimente, Verdikte |
| `bys_visualization.py`, `bys_pdf_export.py` | Figuren, PDF |
| `bys_constants.py`, `bys_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Nicht-Gaußsche Priors und Sampling** (MAP gegen Mittel, MCMC): Stück 8.
- **Hierarchische Priors und Hyperprior auf τ:** ändern die Aussage nicht.
- **Nichtlineare Vorwärtsmodelle (Laplace-Näherung):** Stück 9.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).
