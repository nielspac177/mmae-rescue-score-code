# MMA Embolization Rescue Risk Score — Analysis Code

Reproducible analysis pipeline for a pre-procedural risk score predicting **rescue surgery after middle meningeal artery (MMA) embolization for chronic subdural hematoma (cSDH)**.

> **Research use only.** Internally derived in 214 consecutive patients (36 events). External validation pending. Not a substitute for clinical judgment.

🌐 **Live calculator:** <https://nielspac177.github.io/mmae-rescue-score/>
📦 **Site repo:** <https://github.com/nielspac177/mmae-rescue-score>

---

## What this repo contains

Pure analysis code. No raw patient data — the source CSV (`mmaecsv.csv`) is gitignored.

| File | Purpose |
|---|---|
| `build_score_v2.py`     | Primary scores (Models 1 + 2): integer-score AUC (apparent + Harrell optimism-corrected via 1000 bootstraps), calibration. Also produces focal-deficit-included variants for sensitivity. |
| `build_socr_v2.py`      | Model 3 — alternative age-stratification specification (raises upper age threshold from > 80 to > 85). Re-fit on the same n = 214 cohort. |
| `build_model4_v2.py`    | Model 4 — data-driven L1 (lasso) logistic regression with EPV-5 cap; evaluated under three class-imbalance modes (natural / class\_weight=balanced / SMOTE). |
| `build_model5_v2.py`    | Model 5 — enhanced sensitivity bundle: restricted cubic splines, prespecified clinical interactions, Optuna-tuned elastic-net, stacked ensemble (logistic + GBM), Bayesian logistic with informative priors anchored to published MMA series, MICE imputation. |
| `build_table1_v2.py`    | Table 1 — baseline characteristics by rescue status (median IQR / mean SD; χ² / Fisher / Mann–Whitney). |
| `build_ml_v2.py`        | Random forest, gradient boosting, elastic-net, XGBoost benchmark on the same predictors (5×10 stratified CV, 1000-bootstrap CIs). |
| `build_nomogram_v2.py`  | Harrell-style nomogram for the primary Model 1. |
| `build_dca_v2.py`       | Decision curve analysis (Vickers) + operating-point metrics (sens / spec / PPV / NPV with Wilson CIs). |
| `build_fig0_v3.py`      | STROBE-style study-flow figure. |
| `build_3model_roc.py`   | Three-model ROC overlay (Models 1, 2, 3). |
| `build_figs_3model.py`  | Score-by-rate panels, calibration, decision threshold, DCA across the three integer scores. |
| `figs_v2.py`            | Forest plot of univariable ORs and supplementary figure rendering. |
| `build_supp_xlsx_v2.py` | Supplementary tables workbook. |
| `build_docx_full_v2.py` | Full IMRAD Word manuscript (JAMA Neurology format). |
| `build_html_full_v2.py` | Self-contained HTML manuscript. |
| `build_bedside_card_v2.py` | One-page printable PDF (consumed by the live site). |
| `build_slides_v2.py`    | Ten-slide PowerPoint deck (16:9, JAMA aesthetic). |
| `v2/`                   | De-identified outputs: scored cohort, summary JSON, references, integer-score coefficients, lasso paths, Bayesian/imbalance comparisons, literature priors. |

`v2/scored_cohort_v2.csv` contains the binary score components and predicted probabilities only — no MRN, no dates, no free-text.

---

## Models at a glance

The score family is organised into **two primary models** for clinical use and **three sensitivity analyses** that probe robustness to specification choices.

### Primary (knowledge-driven)

| | Variables | Max | Score AUC (corrected) | Cutoff | Sens / Spec | Low-risk → High-risk rate |
|---|---|---|---|---|---|---|
| **Model 1** (full)   | age + 5 others | 7 | **0.704** | ≥ 4 | 64 % / 78 % | 8.6 % vs 36.5 % (4.2-fold) |
| **Model 2** (simple) | age + 2 others | 4 | 0.636 | ≥ 3 | 36 % / 84 % | — |

Model 1 variables: age (< 65 / 65–80 / > 80 → 0/1/2), SDH volume ≥ 100 mL, oral anticoagulation, platelets < 150 × 10⁹/L, antiplatelet therapy, embolization of both anterior + posterior MMA branches.
Model 2 variables: age, SDH volume ≥ 100 mL, oral anticoagulation.

### Sensitivity

| | Description | AUC (corrected or 5×10 CV) | 95 % CI |
|---|---|---|---|
| **Model 3** | Alternative age stratification — raises upper age threshold from > 80 to > 85; predictors = age + SDH ≥ 100 mL + plt < 150 + antiplatelet | 0.704 | — |
| **Model 4** | Data-driven lasso (EPV-5 cap, natural class balance) | 0.66 | 0.56 – 0.75 |
| **Model 5** | Bundled enhancements (splines + interactions + tuned elastic-net + stacked ensemble + Bayesian-with-priors + MICE) — Bayesian variant | 0.67 | 0.57 – 0.77 |
| **Model 5** | Bundled enhancements — composite mean | 0.63 | 0.52 – 0.74 |
| Focal-deficit-included variants of Models 1, 2, 3 | Drop-in re-fit including focal deficit; reported for completeness — variable was non-significant in primary multivariable Model 1 (adjusted OR 0.80, P = 0.58) | within ± 0.02 of primary | — |

Optimism correction via 1000 nonparametric bootstrap replicates (Harrell). Calibration assessed by Hosmer–Lemeshow with non-significant *P* values for all primary scores. Cross-validated AUCs computed under 5×10 stratified repeated CV with 1000-bootstrap percentile CIs.

The integer scores match or exceed every machine-learning benchmark trained on the same inputs (random forest CV-AUC 0.58, gradient boosting 0.57, XGBoost 0.57, elastic-net 0.65). At 36 events, the events-per-variable constraint sets the AUC ceiling, not the modeling strategy.

---

## Reproducing the analysis

### 1. Clone and install
```bash
git clone https://github.com/nielspac177/mmae-rescue-score-code.git
cd mmae-rescue-score-code
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn statsmodels matplotlib scipy \
            openpyxl python-docx reportlab xgboost \
            optuna pymc imbalanced-learn python-pptx
```

### 2. Provide the source data
Drop your cohort CSV at the repo root as `mmaecsv.csv`. Required columns (case-sensitive):

```
casecodes, age, gender_num, hypertension, statins,
anticoagulation, antiplatelet, plt_num, inrlastbeforeprocedure,
focal_deficit, headache, nausea, seizures, gait, aphasia,
gcsonpresentation, baselinemrs, branches, useofliquidembolic,
sdhvolumebaseline, midlineshiftmeasureinmmbaseline, shift5baseline,
densitybaseline, structurebaseline, membranesbasline,
acute_subacutebasline, bilateral_num2, standalone1, rescue_surgery
```

`rescue_surgery` and binary string columns must be `"Yes"`/`"No"`. `branches` must contain `"Anterior + posterior"` for the dual-branch level. `gender_num` must be `"Male"`/`"Female"`.

### 3. Run the pipeline (order matters)
```bash
python build_score_v2.py        # Models 1 + 2 (primary) + focal-included variants
python build_socr_v2.py         # Model 3 (alternative age stratification, sensitivity)
python build_table1_v2.py       # baseline-characteristics table
python figs_v2.py               # forest plot + supplementary figures
python build_3model_roc.py      # 3-model ROC overlay (run after figs_v2 to win)
python build_figs_3model.py     # score-rate, calibration, decision-threshold, DCA
python build_fig0_v3.py         # study-flow figure
python build_ml_v2.py           # random forest / GBM / elastic-net / XGBoost benchmark
python build_nomogram_v2.py     # Harrell nomogram for Model 1
python build_dca_v2.py          # decision-curve + operating-point metrics
python build_model4_v2.py       # Model 4 (lasso, three imbalance modes)
python build_model5_v2.py       # Model 5 (enhanced bundle, Bayesian-with-priors)
python build_supp_xlsx_v2.py    # supplementary tables workbook
python build_docx_full_v2.py    # full Word manuscript
python build_html_full_v2.py    # full HTML manuscript
python build_bedside_card_v2.py # 1-page printable PDF (consumed by site)
python build_slides_v2.py       # 10-slide PowerPoint deck
```

Total runtime: about 3–5 minutes on a laptop (Model 5 dominates, ~90 s for the Bayesian PyMC fit inside CV).

### 4. Verify
After the pipeline completes:
- `v2/summary_v2.json` reports Model 1 corrected AUC ≈ 0.70 and Model 2 ≈ 0.64.
- `v2/m4_imbalance_comparison.csv` lists Model 4 across the three imbalance modes (natural / balanced / SMOTE).
- `v2/m5_results.csv` lists Model 5 across the four enhancement variants and their composite.
- `v2/operating_points.csv` reproduces the sensitivity / specificity table.
- `REPORT_v2.docx` opens cleanly in Word.

---

## Data privacy

This repo contains **no patient-identifying information**. The `.gitignore` excludes:

```
mmaecsv.csv
mmaefinalfinal.xlsx
*.RData
.Rhistory
```

`v2/scored_cohort_v2.csv` contains only the binary score components, integer scores, and predicted probabilities — no MRN, no dates, no free-text fields. This file is committed; it is what an external validator needs.

To validate externally:
1. Compute the same binary features on your cohort.
2. Apply the integer-score coefficients in `v2/m{1,2,3}_logit_coefs.csv` to predict probabilities (or sum the integer points).
3. Compare your AUC, calibration, and observed rescue rates by score stratum to the values reported here.

---

## Reporting standards

The accompanying manuscript follows the **TRIPOD-AI** extension to TRIPOD for prediction-model studies. The completed checklist is in `supp_tables.xlsx` (Sheet: TRIPOD-AI).

Statistical methods:
- **Discrimination:** AUC, computed from the integer score and from logistic regression on the components. Optimism correction via 1000 nonparametric bootstrap replicates (Harrell).
- **Calibration:** Visual + Hosmer–Lemeshow χ².
- **Stratum risks:** Wilson 95 % confidence intervals.
- **Operating points:** Sens / Spec / PPV / NPV at candidate cutoffs.
- **Decision-curve analysis:** Vickers' net-benefit framework over the clinically reasonable threshold range.
- **Cross-validation:** 5×10 stratified repeated CV with 1000-bootstrap percentile CIs (Models 4 and 5).
- **Bayesian priors:** Normal priors on log-odds ratios anchored to published MMA-embolization series (see `v2/literature_priors.md`).

---

## Citation

The accompanying manuscript is in preparation. In the interim, please cite the calculator and code as:

> *MMA Embolization Rescue Risk Calculator and analysis code (v2).*
> Calculator: <https://nielspac177.github.io/mmae-rescue-score/>
> Code: <https://github.com/nielspac177/mmae-rescue-score-code>

A DOI (Zenodo release) and a peer-reviewed citation will be added once the manuscript is published.

---

## License

Code: **MIT**. Figures and prose: **CC BY 4.0**.

External validation, pull requests, and bug reports are welcome.
