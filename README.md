# MMA Embolization Rescue Risk Score — Analysis Code

Reproducible analysis pipeline for a pre-procedural integer risk score predicting **rescue surgery after middle meningeal artery (MMA) embolization for chronic subdural hematoma (cSDH)**.

> **Research use only.** Internally derived in 214 consecutive patients (36 events). Externally not validated yet. Do not use as a substitute for clinical judgment.

🌐 **Live calculator** (consumer of this code): https://nielspac177.github.io/mmae-rescue-score/
📦 **Site repo**: https://github.com/nielspac177/mmae-rescue-score

---

## What this repo contains

Pure analysis code. No raw patient data — the source CSV (`mmaecsv.csv`) is gitignored.

| File | Purpose |
|---|---|
| `build_score_v2.py`     | Score derivation, AUC (apparent + Harrell optimism-corrected via 1000 bootstraps), calibration |
| `build_socr_v2.py`      | Co-author parallel SOCR score (Model 3) re-fit on the same n=214 cohort |
| `build_table1_v2.py`    | Table 1 — baseline characteristics by rescue status (median IQR / mean SD; χ² / Fisher / Mann–Whitney) |
| `build_ml_v2.py`        | Comparison vs random forest, gradient boosting, elastic-net, XGBoost (5×10 stratified CV, bootstrap CIs) |
| `build_nomogram_v2.py`  | Harrell-style nomogram for Model 1 |
| `build_dca_v2.py`       | Decision curve analysis (Vickers) + operating-point metrics (sens/spec/PPV/NPV with Wilson CIs) |
| `build_fig0_v3.py`      | STROBE-style study-flow figure |
| `build_3model_roc.py`   | Three-model ROC overlay (Models 1, 2, 3) |
| `build_figs_v2.py`      | Score-vs-risk bars, calibration, forest, score tables, decision-threshold figure |
| `build_supp_xlsx_v2.py` | 11-sheet supplementary tables workbook |
| `build_docx_full_v2.py` | Full IMRAD Word manuscript |
| `build_html_full_v2.py` | Self-contained HTML manuscript |
| `build_bedside_card_v2.py` | 1-page printable PDF (used by the live site) |
| `v2/`                   | De-identified outputs: scored cohort, summary JSON, references, intermediate CSVs, integer-score coefficients |

`v2/scored_cohort_v2.csv` contains de-identified feature vectors only (no MRN, no dates).

---

## The three scores at a glance

| | Variables | Max | Score AUC (corrected) | Cutoff | Sens / Spec at cutoff |
|---|---|---|---|---|---|
| **Model 1** (full)   | age + 6 others | 8 | **0.732** | ≥ 5 | 58.3 % / 84.3 % |
| **Model 3** (SOCR)   | age + 4 others | 6 | 0.724 | ≥ 4 | 58.3 % / 84.3 % |
| **Model 2** (simple) | age + 3 others | 5 | 0.681 | ≥ 4 | 30.6 % / 89.3 % |

All three are derived on the same n = 214 cohort (median imputation for 32 missing baseline volumes and 1 missing platelet value). Optimism correction via 1000 nonparametric bootstrap replicates (Harrell). Calibration assessed by Hosmer–Lemeshow with non-significant P values for all three.

The integer scores **outperform** random forest, gradient boosting, and XGBoost trained on the same inputs at this event count — a known consequence of an event-per-variable ratio in the single digits.

---

## Reproducing the analysis end-to-end

### 1. Clone and set up
```bash
git clone https://github.com/nielspac177/mmae-rescue-score-code.git
cd mmae-rescue-score-code
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn statsmodels matplotlib scipy \
            openpyxl python-docx reportlab xgboost
```

### 2. Provide the source data
Drop your cohort CSV at the repo root as `mmaecsv.csv`. Required columns (case-sensitive):

```
casecodes, age, gender_num, smoking, hypertension, diabetes, liver,
malignancy, anticoagulation, antiplatelet, hb_num, plt_num,
inrlastbeforeprocedure, focal_deficit, headache, fall, gcsonpresentation,
baselinemrs, branches, useofliquidembolic, sdhvolumebaseline,
midlineshiftmeasureinmmbaseline, rescue_surgery
```

`rescue_surgery` and binary string columns must be `"Yes"`/`"No"`. `branches` must contain `"Anterior + posterior"` for the dual-branch level. `gender_num` must be `"Male"`/`"Female"`.

### 3. Run the pipeline (order matters)
```bash
python build_score_v2.py        # → v2/scored_cohort_v2.csv, summary_v2.json, fig assets
python build_socr_v2.py          # → adds Model 3 columns + summary entries
python build_table1_v2.py        # → v2/table1_baseline.csv
python build_figs_v2.py          # → fig1..fig6 (figs 1 will be overwritten by 3-model ROC)
python build_3model_roc.py       # → fig1_roc.png with all three models
python build_fig0_v3.py          # → fig0_study_flow.png
python build_ml_v2.py            # → ml_comparison.csv, fig7_ml_roc, fig8_ml_bars
python build_nomogram_v2.py      # → fig9_nomogram.png
python build_dca_v2.py           # → fig10_dca.png, operating_points.csv
python build_supp_xlsx_v2.py     # → supp_tables.xlsx (11 sheets)
python build_docx_full_v2.py     # → REPORT_v2.docx
python build_html_full_v2.py     # → REPORT_v2.html
python build_bedside_card_v2.py  # → docs/bedside_card.pdf (consumed by site repo)
```

Total runtime on a laptop: about 90 seconds. Re-running with new data only requires placing a new `mmaecsv.csv` in the root and re-running the pipeline.

### 4. Verify
After the pipeline completes:
- `v2/summary_v2.json` should contain matching AUC numbers (Model 1 corrected ≈ 0.732, Model 2 ≈ 0.681, Model 3 ≈ 0.724).
- `v2/operating_points.csv` should reproduce the sens/spec table reported above.
- `REPORT_v2.docx` should open cleanly in Word.

---

## Data privacy

This repo contains **no patient-identifying information**. The `.gitignore` excludes:

```
mmaecsv.csv
mmaefinalfinal.xlsx
*.RData
.Rhistory
```

`v2/scored_cohort_v2.csv` contains only the binary score components and outcome (`y`, `score_m1`, `score_m2`, `score_m3`, etc.) — no MRN, no dates, no free-text fields. This file *is* committed and is what an external validator needs.

To validate externally, you can:
1. Compute the same binary features on your cohort.
2. Apply the integer-score coefficients in `v2/m{1,2,3}_logit_coefs.csv` to predict probabilities.
3. Compare your AUC and calibration to ours.

---

## Reporting standards

The accompanying manuscript follows the **TRIPOD** checklist for prediction-model studies.
A complete TRIPOD checklist is included in the supplementary tables workbook (`supp_tables.xlsx`).

Statistical methods:
- **Discrimination**: AUC, computed both from the integer score and via logistic regression on the components. Optimism corrected via 1000 nonparametric bootstrap replicates (Harrell).
- **Calibration**: Visual + Hosmer–Lemeshow χ².
- **Stratum risks**: Wilson 95% confidence intervals.
- **Operating points**: Sens / Spec / PPV / NPV at candidate cutoffs with Wilson CIs.
- **Decision-curve analysis**: Vickers' net-benefit framework over the clinically reasonable threshold range.
- **ML benchmark**: 5×10 stratified repeated cross-validation; AUCs with 1000-bootstrap percentile CIs.

---

## Citation

The manuscript is in preparation. In the interim, please cite the calculator and code as:

> *MMA Embolization Rescue Risk Calculator and analysis code (v2).*
> Calculator: <https://nielspac177.github.io/mmae-rescue-score/>
> Code: <https://github.com/nielspac177/mmae-rescue-score-code>

A DOI (Zenodo release) and a peer-reviewed citation will be added to this README when the manuscript is published.

---

## License

Code: **MIT**. Figures and prose: **CC BY 4.0**.

External validation, pull requests, and bug reports are welcome.
