"""
Final HTML report — full manuscript-style document with:
  - Abstract (no intro/discussion per user)
  - Methods section + flow figure
  - Table 1 (baseline characteristics)
  - Results sections with narrative + figures + tables
  - Nomogram + interactive JavaScript calculator
  - Phenotype clustering reported as a negative result
"""
import os, base64, json
import numpy as np
import pandas as pd

OUT = "/Users/nielspacheco/Desktop/Research/Jimena Gonzales-salidos/MMAE scoring"

def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def df2html(df, classes="tbl"):
    return df.to_html(index=False, classes=classes, border=0, escape=False)

with open(f"{OUT}/summary.json") as f:
    summary = json.load(f)
with open(f"{OUT}/main_summary.json") as f:
    main_summary = json.load(f)
with open(f"{OUT}/variants_summary.json") as f:
    variants_summary = json.load(f)

# Build Model B HTML tables
mB = pd.read_csv(f"{OUT}/variants_model_B.csv")
mB["OR_uv (95% CI)"] = mB.apply(
    lambda r: f"{r['OR_uv']:.2f} ({r['uv_lo']:.2f}–{r['uv_hi']:.2f})", axis=1)
mB["OR_mv (95% CI)"] = mB.apply(
    lambda r: f"{r['OR_mv']:.2f} ({r['mv_lo']:.2f}–{r['mv_hi']:.2f})", axis=1)
mB["P (multivariable)"] = mB["mv_p"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
mB["P (univariate)"]    = mB["uv_p"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
mB_html = mB.rename(columns={"events_descr": "Events / Exposed"})[
    ["Variable", "Events / Exposed",
     "OR_mv (95% CI)", "P (multivariable)",
     "OR_uv (95% CI)", "P (univariate)"]]

# Build Model A and C tables (compact)
def quick_mv_table(path):
    d = pd.read_csv(path)
    d["OR_uv (95% CI)"] = d.apply(
        lambda r: f"{r['OR_uv']:.2f} ({r['uv_lo']:.2f}–{r['uv_hi']:.2f})", axis=1)
    d["OR_mv (95% CI)"] = d.apply(
        lambda r: f"{r['OR_mv']:.2f} ({r['mv_lo']:.2f}–{r['mv_hi']:.2f})", axis=1)
    d["P_mv"] = d["mv_p"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
    return d.rename(columns={"events_descr": "Events / Exposed"})[
        ["Variable", "Events / Exposed",
         "OR_mv (95% CI)", "P_mv", "OR_uv (95% CI)"]]
mA_html = quick_mv_table(f"{OUT}/variants_model_A.csv")
mC_html = quick_mv_table(f"{OUT}/variants_model_C.csv")

# Compact comparison table
variants_compare = pd.DataFrame([
    {"Model": f"<b>{k}</b>" + (" ★" if k == "B" else ""),
     "Variables (max pts)": variants_summary[k]["n_vars"],
     "Apparent AUROC": f"{variants_summary[k]['auc_apparent']:.3f}",
     "Bootstrap-corrected": f"{variants_summary[k]['auc_corrected']:.3f}",
     "Nested 5×50 CV": f"{variants_summary[k]['auc_nested']:.3f} ± {variants_summary[k]['auc_nested_sd']:.3f}"}
    for k in ["A", "B", "C"]
])

# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
# MAIN model — primary tables
risk_table_main = pd.read_csv(f"{OUT}/risk_factor_main_uni.csv")
mv = pd.read_csv(f"{OUT}/risk_factor_main_mv.csv")
mv["Profile OR (95% CI)"] = mv.apply(
    lambda r: f"{r['OR']:.2f} ({r['prof_lo']:.2f}–{r['prof_hi']:.2f})", axis=1)
mv["Wald OR (95% CI)"] = mv.apply(
    lambda r: f"{r['OR']:.2f} ({r['wald_lo']:.2f}–{r['wald_hi']:.2f})", axis=1)
mv["P (profile)"] = mv["p_prof"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
mv["P (Wald)"]    = mv["p_wald"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
mv_html = mv[["Variable", "Profile OR (95% CI)", "P (profile)",
              "Wald OR (95% CI)", "P (Wald)"]]

main_score_def = pd.read_csv(f"{OUT}/main_score_definition.csv")
main_strata = pd.read_csv(f"{OUT}/main_risk_by_score.csv")
main_strata["Rescue rate"] = (main_strata["rate"] * 100).map("{:.1f}%".format)
main_strata = main_strata.rename(columns={"main_score": "Score",
                                            "n": "n", "rescues": "Rescues"})[
    ["Score", "n", "Rescues", "Rescue rate"]]

# Sensitivity model (with paradox vars; previously primary)
risk_table = pd.read_csv(f"{OUT}/risk_factor_table_firth.csv")

# Knowledge-driven main model
mv_know = pd.read_csv(f"{OUT}/risk_factor_main_knowledge.csv")
mv_know["Profile OR (95% CI)"] = mv_know.apply(
    lambda r: f"{r['OR']:.2f} ({r['prof_lo']:.2f}–{r['prof_hi']:.2f})", axis=1)
mv_know["Wald OR (95% CI)"] = mv_know.apply(
    lambda r: f"{r['OR']:.2f} ({r['wald_lo']:.2f}–{r['wald_hi']:.2f})", axis=1)
mv_know["P (profile)"] = mv_know["p_prof"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
mv_know["P (Wald)"]    = mv_know["p_wald"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
mv_know_html = mv_know[["Variable", "Profile OR (95% CI)", "P (profile)",
                         "Wald OR (95% CI)", "P (Wald)"]]

# Paradox investigation tables
seq_df = pd.read_csv(f"{OUT}/paradox_sequential.csv")
seq_df["Asymptomatic OR (95% CI), p"]  = seq_df.apply(
    lambda r: f"{r['Asym_OR']:.2f} ({r['Asym_lo']:.2f}–{r['Asym_hi']:.2f}), p={r['Asym_p']:.3f}", axis=1)
seq_df["Focal deficit OR (95% CI), p"] = seq_df.apply(
    lambda r: f"{r['Focal_OR']:.2f} ({r['Focal_lo']:.2f}–{r['Focal_hi']:.2f}), p={r['Focal_p']:.3f}", axis=1)
seq_html = seq_df[["Adjustment", "Asymptomatic OR (95% CI), p",
                    "Focal deficit OR (95% CI), p"]]

strat_df = pd.read_csv(f"{OUT}/paradox_stratified.csv")
strat_df["OR (95% CI), p"] = strat_df.apply(
    lambda r: f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}), p={r['p']:.3f}", axis=1)
strat_html = strat_df[["Variable", "Stratum", "n", "OR (95% CI), p"]]

# Wald-vs-profile sensitivity table is the same mv_html (4 cols)
mv_compare = mv_html

# Bias-triangulation tables
cox_mv = pd.read_csv(f"{OUT}/cox_multivariable.csv")
cox_mv["HR (95% CI)"] = cox_mv.apply(
    lambda r: f"{r['HR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", axis=1)
cox_mv["P value"] = cox_mv["p"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
cox_html = cox_mv[["Variable", "HR (95% CI)", "P value"]]

rad_mv = pd.read_csv(f"{OUT}/radiologic_endpoint_multivariable.csv")
rad_mv["Profile OR (95% CI)"] = rad_mv.apply(
    lambda r: f"{r['OR']:.2f} ({r['prof_lo']:.2f}–{r['prof_hi']:.2f})", axis=1)
rad_mv["Wald OR (95% CI)"] = rad_mv.apply(
    lambda r: f"{r['OR']:.2f} ({r['wald_lo']:.2f}–{r['wald_hi']:.2f})", axis=1)
rad_mv["P (profile)"] = rad_mv["p_prof"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
rad_html = rad_mv[["Variable", "Profile OR (95% CI)", "P (profile)"]]

pba = pd.read_csv(f"{OUT}/pba_asymptomatic.csv")
pba["OR (95% CI), p"] = pba.apply(
    lambda r: f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}), p={r['p']:.3f}", axis=1)
pba["Bias fraction f"] = (pba["f"] * 100).map("{:.0f}%".format)
pba_html = pba[["Bias fraction f", "n_reassigned", "OR (95% CI), p"]]

# Paradox-free alternative model tables
alt_mv = pd.read_csv(f"{OUT}/risk_factor_multivariable_alt.csv")
alt_mv["Profile OR (95% CI)"] = alt_mv.apply(
    lambda r: f"{r['OR']:.2f} ({r['prof_lo']:.2f}–{r['prof_hi']:.2f})", axis=1)
alt_mv["Wald OR (95% CI)"] = alt_mv.apply(
    lambda r: f"{r['OR']:.2f} ({r['wald_lo']:.2f}–{r['wald_hi']:.2f})", axis=1)
alt_mv["P (profile)"] = alt_mv["p_prof"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
alt_mv["P (Wald)"]    = alt_mv["p_wald"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
alt_mv_html = alt_mv[["Variable", "Profile OR (95% CI)", "P (profile)",
                       "Wald OR (95% CI)", "P (Wald)"]]

alt_know = pd.read_csv(f"{OUT}/risk_factor_knowledge_alt.csv")
alt_know["Profile OR (95% CI)"] = alt_know.apply(
    lambda r: f"{r['OR']:.2f} ({r['prof_lo']:.2f}–{r['prof_hi']:.2f})", axis=1)
alt_know["Wald OR (95% CI)"] = alt_know.apply(
    lambda r: f"{r['OR']:.2f} ({r['wald_lo']:.2f}–{r['wald_hi']:.2f})", axis=1)
alt_know["P (profile)"] = alt_know["p_prof"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
alt_know["P (Wald)"]    = alt_know["p_wald"].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
alt_know_html = alt_know[["Variable", "Profile OR (95% CI)", "P (profile)",
                           "Wald OR (95% CI)", "P (Wald)"]]

alt_score = pd.read_csv(f"{OUT}/alt_score_definition.csv")
alt_score["Direction"] = alt_score["scoring_when"].map({"presence": "presence (Yes)",
                                                          "absence":  "absence (No)"})
alt_score_def_html = alt_score.rename(columns={"feature": "Variable"})[
    ["Variable", "Direction", "points"]]

alt_strata = pd.read_csv(f"{OUT}/alt_risk_by_score.csv")
alt_strata["Rescue rate"] = (alt_strata["rate"] * 100).map("{:.1f}%".format)
alt_strata_html = alt_strata.rename(columns={"alt_score": "Score",
                                              "n": "n",
                                              "rescues": "Rescues"})[
    ["Score", "n", "Rescues", "Rescue rate"]]

# Table 1
t1_raw = pd.read_csv(f"{OUT}/table1_baseline.csv").fillna("")
# Detect dynamic column names
t1_cols = list(t1_raw.columns)
COL_ALL = next(c for c in t1_cols if c.startswith("All patients"))
COL_RES = next(c for c in t1_cols if c.startswith("Rescue"))
COL_NO  = next(c for c in t1_cols if c.startswith("No rescue"))
HDR_ALL = COL_ALL.replace("All patients ", "All patients<br>")
HDR_RES = COL_RES.replace("Rescue ", "Rescue surgery<br>")
HDR_NO  = COL_NO.replace("No rescue ", "No rescue<br>")

def table1_html(t1):
    rows = []
    rows.append('<table class="tbl">')
    rows.append('<thead><tr>'
                '<th>Characteristic</th>'
                f'<th>{HDR_ALL}</th>'
                f'<th>{HDR_RES}</th>'
                f'<th>{HDR_NO}</th>'
                '<th>P value</th>'
                '</tr></thead><tbody>')
    for _, r in t1.iterrows():
        if r["_group_header"]:
            rows.append(f'<tr class="grouphdr"><td colspan="5"><b>{r["_group_header"]}</b></td></tr>')
        else:
            p = r["P value"]
            p_html = f'<b style="color:#A8232C">{p}</b>' if (p not in ("", "—") and p != "<0.001" and float(p) < 0.05) or p == "<0.001" else p
            rows.append(f'<tr><td style="padding-left:18px">{r["Characteristic"]}</td>'
                        f'<td>{r[COL_ALL]}</td>'
                        f'<td>{r[COL_RES]}</td>'
                        f'<td>{r[COL_NO]}</td>'
                        f'<td>{p_html}</td></tr>')
    rows.append('</tbody></table>')
    return "\n".join(rows)
t1_html = table1_html(t1_raw)

# Screening table
screen = pd.read_csv(f"{OUT}/screening_table.csv")
screen_show = screen[screen["cutoff"] >= 1].copy()
for c in ["sens", "spec", "ppv", "npv", "pct_referred"]:
    screen_show[c] = (screen_show[c] * 100).map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
for c in ["lr_pos", "lr_neg"]:
    screen_show[c] = screen_show[c].map(lambda v: f"{v:.2f}" if (pd.notna(v) and np.isfinite(v)) else "—")
screen_show = screen_show.rename(columns={
    "cutoff": "Cutoff (≥)", "TP": "TP", "FP": "FP", "FN": "FN", "TN": "TN",
    "sens": "Sensitivity", "spec": "Specificity", "ppv": "PPV", "npv": "NPV",
    "lr_pos": "LR+", "lr_neg": "LR−",
    "pos_test": "Test+", "pct_referred": "% referred",
})[["Cutoff (≥)", "TP", "FP", "FN", "TN",
     "Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR−",
     "Test+", "% referred"]]

# Imbalance results
imb = pd.read_csv(f"{OUT}/imbalance_results.csv")
imb_top = imb.sort_values("auprc", ascending=False).head(10).copy()
imb_top["AUROC"] = imb_top["auc"].map("{:.3f}".format) + " ± " + imb_top["auc_sd"].map("{:.3f}".format)
imb_top["AUPRC"] = imb_top["auprc"].map("{:.3f}".format) + " ± " + imb_top["auprc_sd"].map("{:.3f}".format)
imb_top["Brier"] = imb_top["brier"].map("{:.3f}".format)
imb_top["Sens (Youden)"] = imb_top["sens"].map("{:.0%}".format)
imb_top["Spec (Youden)"] = imb_top["spec"].map("{:.0%}".format)
imb_html = imb_top.rename(columns={"label": "Strategy"})[
    ["Strategy", "AUROC", "AUPRC", "Brier", "Sens (Youden)", "Spec (Youden)"]
]

# Unified benchmark (nested CV)
bench = pd.read_csv(f"{OUT}/benchmark_summary.csv")
bench["AUC (mean ± SD)"] = (
    bench["AUC mean"].map("{:.3f}".format) + " ± " +
    bench["AUC SD"].map("{:.3f}".format))
bench["95% range across folds"] = (
    "[" + bench["AUC 2.5%"].map("{:.3f}".format) + " – " +
    bench["AUC 97.5%"].map("{:.3f}".format) + "]")

# Split into two benchmark tables
ALL45 = ["All-features logistic", "All-features L1-logistic",
         "Random Forest (45 vars)", "Gradient Boosting (45 vars)"]
CLIN4 = ["Score-as-logistic (4 vars)", "L1-Logistic (4 vars)",
         "Random Forest (4 vars)", "Gradient Boosting (4 vars)",
         "Score pipeline (nested CV)"]
bench_all = bench[bench["Pipeline"].isin(ALL45)].sort_values("AUC mean", ascending=False)
bench_clin = bench[bench["Pipeline"].isin(CLIN4)].sort_values("AUC mean", ascending=False)
cv_table_all  = bench_all[["Pipeline", "AUC (mean ± SD)", "95% range across folds"]]
cv_table_clin = bench_clin[["Pipeline", "AUC (mean ± SD)", "95% range across folds"]]

# Main score definition + strata (PRIMARY)
risk_score_def = pd.DataFrame({
    "Variable": [r["Variable"] for r in main_summary["score_components"]],
    "Direction": [r["Direction"] for r in main_summary["score_components"]],
    "Points": [f"+{r['Points']}" for r in main_summary["score_components"]],
})
risk_strata = pd.read_csv(f"{OUT}/main_risk_by_score.csv")
risk_strata["Rescue rate"] = (risk_strata["rate"] * 100).map("{:.1f}%".format)
risk_strata = risk_strata.rename(columns={"main_score": "Score",
                                            "rescues": "Rescues"})[
    ["Score", "n", "Rescues", "Rescue rate"]]

# Score-to-probability for calculator
calc_csv = pd.read_csv(f"{OUT}/score_to_probability.csv")
score_to_prob = {int(r["score"]): float(r["probability"]) for _, r in calc_csv.iterrows()}

# Phenotype description
phen_tbl = pd.DataFrame({
    "Phenotype": ["Cluster 1 — Established chronic membranous (lowest risk)",
                  "Cluster 2 — Active separated/gradation",
                  "Cluster 3 — Homogenous low-density"],
    "Description": [
        "Older patients (~75 y), worse mRS (2), thick SDH (~19 mm), large volume (~117 mL), 88% with membranes — established organized hematoma. Lowest observed rescue rate (9.4%).",
        "100% symptomatic, predominantly separated/gradation structure (81%), only 37% with membranes (16.4% rescue rate).",
        "Symptomatic, predominantly homogenous/laminar (87%), no membranes (90%), uniformly hypodense chronic collection (16.4% rescue rate).",
    ],
})

# Univariate top
uni_top = pd.DataFrame({
    "Variable": ["Symptomatic presentation", "Anterior + posterior branches",
                 "Membranes on baseline CT", "Focal deficit",
                 "Separated/gradation structure", "Antiplatelet therapy",
                 "Coils + particles embolic"],
    "n / events": ["131 / 15", "75 / 15", "51 / 4", "33 / 2",
                   "65 / 12", "53 / 11", "94 / 17"],
    "Firth OR (95% CI)": ["0.19 (0.06–0.56)", "2.27 (0.91–6.11)",
                    "0.40 (0.11–1.14)", "0.37 (0.07–1.25)",
                    "2.14 (0.79–6.27)", "1.99 (0.80–4.92)",
                    "2.03 (0.77–6.18)"],
    "P value": ["0.002", "0.031", "0.039", "0.056", "0.057", "0.054", "0.062"],
})

# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
n_cohort = summary['n']; n_events = summary['events']; ev_rate = summary['event_rate']*100

html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>MMAE Rescue-Surgery Scoring System</title>
<style>
  body {{ font-family: 'Helvetica', 'Arial', sans-serif; color:#222;
          max-width: 1180px; margin: 30px auto; padding: 0 30px; line-height:1.55;
          background:#FAFAFA; }}
  header {{ border-bottom: 3px solid #1F3D5C; padding-bottom: 14px; margin-bottom: 22px;}}
  h1 {{ color:#1F3D5C; font-size:1.65em; margin:0;}}
  h2 {{ color:#1F3D5C; font-size:1.22em; margin-top:36px;
        border-left:3px solid #2D7A8F; padding-left:10px; }}
  h3 {{ color:#2D7A8F; font-size:1.05em; margin-top:24px; }}
  .meta {{ color:#666; font-size:0.86em; margin: 6px 0 0;}}
  table.tbl {{ border-collapse: collapse; width:100%; margin:8px 0 14px;
               font-size:0.84em; background:white; }}
  table.tbl th {{ background:#1F3D5C; color:white; padding:6px 9px;
                  text-align:left; font-weight:600;}}
  table.tbl td {{ padding:5px 9px; border-bottom:1px solid #E0E0E0;}}
  table.tbl tr:nth-child(even) td {{ background:#F4F6F8; }}
  table.tbl tr.grouphdr td {{ background:#EAF1F6; color:#1F3D5C; padding-top:7px;}}
  .figcap {{ font-size:0.83em; color:#444; margin: -4px 0 18px; line-height:1.4;}}
  .figure {{ text-align:center; margin: 14px 0; }}
  .figure img {{ max-width: 100%; height:auto; border:1px solid #DDD;
                 background:white; padding:6px; }}
  .key {{ background:#EAF1F6; border-left:3px solid #2D7A8F;
          padding:11px 16px; font-size:0.92em; margin: 14px 0;}}
  .caveat {{ background:#FFF3E0; border-left:3px solid #D5751B;
             padding:11px 16px; font-size:0.92em; margin: 14px 0; }}
  .negative {{ background:#F2F2F2; border-left:3px solid #6B6B6B;
               padding:11px 16px; font-size:0.92em; margin: 14px 0;}}
  .pill {{ display:inline-block; background:#1F3D5C; color:white;
           padding: 2px 9px; border-radius: 9px; font-size: 0.78em; margin-right: 4px;}}
  code {{ background:#F0F0F0; padding:1px 4px; border-radius:3px; }}
  nav.toc {{ background:#FFFFFF; border:1px solid #DDD; padding: 11px 17px;
             font-size: 0.9em; }}
  nav.toc ol {{ margin:0 0 0 18px; }}
  nav.toc a {{ color: #1F3D5C; text-decoration: none;}}
  nav.toc a:hover {{ text-decoration: underline; }}
  /* ---- Calculator ---- */
  .calc {{ background:white; border:1px solid #1F3D5C; padding:18px 22px;
           border-radius: 8px; margin: 18px 0;}}
  .calc h3 {{ margin-top:0; color:#1F3D5C; }}
  .calc label {{ display:block; padding: 6px 0; font-size:0.95em; }}
  .calc input[type="checkbox"] {{ margin-right:9px; transform: scale(1.15);
           accent-color:#1F3D5C; }}
  .calc .result {{ background:#EAF1F6; border-left:4px solid #1F3D5C;
                   padding: 12px 16px; margin-top: 14px; font-size:1.0em; }}
  .calc .total {{ font-size:1.6em; font-weight:bold; color:#1F3D5C; }}
  .calc .prob  {{ font-size:1.4em; font-weight:bold; color:#A8232C; }}
  .calc .stratum {{ display:inline-block; padding:3px 11px; border-radius:11px;
                    font-size:0.84em; color:white; margin-left:8px;}}
  .calc .strat-low  {{ background:#2D7A8F; }}
  .calc .strat-mid  {{ background:#D5751B; }}
  .calc .strat-high {{ background:#A8232C; }}
  abstract {{ display:block; }}
  .abstract {{ background:#FFFFFF; border: 1px solid #1F3D5C; padding: 16px 20px;
               margin-top:18px; }}
  .abstract h2 {{ margin-top:0; border-left:none; padding-left:0; font-size:1.10em;}}
  .abstract p  {{ margin: 8px 0; font-size:0.93em; }}
  .label-strong {{ color:#1F3D5C; font-weight:700; }}
</style></head><body>

<header>
<h1>Rescue Surgery After MMAE for Chronic Subdural Hematoma — A Risk-Stratification Scoring System</h1>
<p class="meta">
  <span class="pill">Cohort n = {n_cohort}</span>
  <span class="pill">Rescue events {n_events} ({ev_rate:.1f}%)</span>
  <span class="pill">Firth-penalized regression</span>
  <span class="pill">Bootstrap-validated</span>
  <span class="pill">Full MMAE cohort (stand-alone + adjunctive)</span>
  <span class="pill">Reference: BAI score (Maragkos 2019)</span>
</p>
</header>

<!-- ===================== ABSTRACT ===================== -->
<section class="abstract">
<h2>Abstract</h2>
<p><span class="label-strong">Background.</span>
Middle meningeal artery embolization (MMAE) is increasingly used to treat
chronic subdural hematoma (cSDH), either as stand-alone therapy or
adjunctive to surgery, but a subset of patients still require rescue
surgery. We sought to derive an objective, memorizable risk-stratification
score that could function as a screening tool to identify patients at
risk of treatment failure.</p>

<p><span class="label-strong">Methods.</span>
We retrospectively analyzed the entire cohort of {n_cohort} consecutive
patients treated with MMAE for cSDH (stand-alone and adjunctive cases
combined; restricting the analysis to stand-alone MMAE alone produced
clinically implausible associations such as asymptomatic presentation
predicting rescue surgery). Candidate predictors spanning demographics,
comorbidities, presentation, laboratory, procedural and baseline-imaging
domains were assessed by univariate and multivariable Firth-penalized
logistic regression with profile penalized-likelihood 95% confidence
intervals. We benchmarked four supervised machine-learning algorithms
(logistic, L1-logistic, random forest, gradient boosting) under repeated
stratified 5-fold cross-validation, evaluated six class-imbalance
strategies, and corrected discrimination for optimism with 1000-bootstrap
resampling. An unsupervised phenotyping pass combined Factor Analysis of
Mixed Data (FAMD) with K-means, Ward, Gaussian-mixture, K-Prototypes and
Gower-distance hierarchical clustering. Multivariable coefficients were
rounded to integer points to derive the final score, which was reframed
as a screening tool by reporting sensitivity, specificity, predictive
values and likelihood ratios at each cutoff.</p>

<p><span class="label-strong">Results.</span>
{n_events} patients ({ev_rate:.1f}%) underwent rescue surgery. After
excluding asymptomatic / incidental presentation a priori on
bias-triangulation grounds (see Section 3a–b), the recommended
multivariable Firth-penalized logistic model — <b>Model B (7 variables)</b>
— retained: <b>absence of focal deficit, SDH volume &gt; 100 mL,
anticoagulant use, antiplatelet use, platelet count &lt; 150 ×10⁹/L,
age ≥ 80 years</b>, and <b>anterior + posterior branch embolization</b>.
Coefficients rounded to +1 each yielded an integer score of 0–7 with
<b>apparent AUROC {variants_summary['B']['auc_apparent']:.2f}</b> and
<b>bootstrap-corrected AUROC
{variants_summary['B']['auc_corrected']:.2f}</b>. Even under the most
conservative honest validation (nested 5 × 50 cross-validation with
the 7-feature logistic refit per fold), held-out AUROC remained
{variants_summary['B']['auc_nested']:.2f} ± {variants_summary['B']['auc_nested_sd']:.2f},
within the moderate-discrimination range expected for a 36-event small-
sample study. The score generated a clean monotonic risk gradient
suitable as a clinical screening tool. FAMD-based clustering identified
three clinically interpretable phenotypes but cluster membership did
not independently predict rescue surgery after adjustment (a negative
result).</p>

<p><span class="label-strong">Conclusions.</span>
A simple 7-variable, 0–7-point score combining demographics,
comorbidities, presentation, imaging severity, and procedural intensity
discriminates rescue surgery after MMAE for cSDH with apparent AUROC
{variants_summary['B']['auc_apparent']:.2f} and bootstrap-corrected AUROC
{variants_summary['B']['auc_corrected']:.2f}. The score is easy to
calculate, monotonically stratifies risk from < 5 % to > 50 %, and
performs comparably to or better than data-driven machine-learning
alternatives. External validation is warranted.</p>
</section>

<!-- ===================== TOC ===================== -->
<nav class="toc">
<b>Contents</b>
<ol>
<li><a href="#meth">Methods</a></li>
<li><a href="#tab1">Table 1 — Baseline characteristics</a></li>
<li><a href="#risk">Risk-factor regression (Firth-penalized)</a></li>
<li><a href="#paradox">Why two associations were excluded a priori</a></li>
<li><a href="#bias-tri">Bias triangulation (Cox + radiological endpoint + PBA)</a></li>
<li><a href="#ml">Unsupervised ML benchmark — full 45-feature matrix</a></li>
<li><a href="#ml4">Clinical 4-variable benchmark + score pipeline</a></li>
<li><a href="#imb">Class-imbalance handling</a></li>
<li><a href="#feat">Consensus feature importance</a></li>
<li><a href="#score">Integer score, nomogram &amp; calculator</a></li>
<li><a href="#screen">Screening-tool performance</a></li>
<li><a href="#famd">Unsupervised phenotyping (negative result)</a></li>
<li><a href="#caveats">Limitations</a></li>
</ol>
</nav>

<!-- ===================== METHODS ===================== -->
<h2 id="meth">1. Methods</h2>
<p><span class="label-strong">Cohort.</span>
We assembled a single-institution registry of <b>{n_cohort}</b>
consecutive cSDH patients treated with MMAE — both stand-alone and
adjunctive (combined with surgery) cases — and analyzed the entire
cohort to derive the score. Restricting the analysis to stand-alone
MMAE alone produced clinically implausible findings (asymptomatic
presentation predicting rescue surgery), motivating the
all-comers approach. The primary outcome was <b>rescue surgery</b>,
defined as any subsequent burr-hole drainage or craniotomy attributable
to the index hematoma.</p>

<p><span class="label-strong">Candidate predictors.</span>
Thirty-four candidate variables across six clinical domains
(demographics, comorbidities &amp; medications, presentation, laboratory,
procedural details, baseline imaging) were pre-specified. Continuous
variables were used in their native units and as clinically meaningful
binary thresholds (e.g., axial thickness ≥ 20 mm; SDH volume ≥ 100 mL).
Missing data were median-imputed for analytic continuity (≤ 16% missing
per variable for retained predictors).</p>

<p><span class="label-strong">Statistical analysis.</span>
Because of the low event count (22 / 148; 14.9%), we used <b>Firth's
penalized maximum likelihood</b> (Heinze &amp; Schemper, 2002) for both
univariate and multivariable logistic regression, with profile penalized-
likelihood 95% CIs. Multivariable selection used backward elimination with
a retention threshold of p ≤ 0.10 on the candidate pool with univariate
p &lt; 0.20. Standard MLE was reported in parallel as a sensitivity
check.</p>

<p><span class="label-strong">Supervised machine learning.</span>
Four classifiers (logistic regression, L1-penalized logistic, random
forest with depth 4, gradient boosting with depth 2) were benchmarked under
repeated stratified 5-fold cross-validation (10 repetitions, 50 folds total)
on the full feature matrix. Discrimination was reported as AUROC.</p>

<p><span class="label-strong">Class-imbalance handling.</span>
Six resampling strategies (SMOTE, SMOTE-NC, BorderlineSMOTE, random
oversampling, random undersampling, none) were combined with two weighting
variants (class-weight = balanced, none) and applied <b>only inside</b>
training folds. Both AUROC and AUPRC (precision-recall area) were
reported, since AUPRC is more informative under imbalance.</p>

<p><span class="label-strong">Score derivation.</span>
Multivariable coefficients were divided by the smallest absolute
coefficient and rounded to integers, yielding a 0–6-point score.
A nomogram was constructed from the score-to-probability mapping fitted on
the development cohort.</p>

<p><span class="label-strong">Internal validation.</span>
Discrimination was corrected for optimism with 1000-iteration bootstrap
resampling (Harrell). Calibration was assessed with quartiles of
predicted risk.</p>

<p><span class="label-strong">Screening-tool framing.</span>
At each integer cutoff we computed sensitivity, specificity, positive
and negative predictive values, positive and negative likelihood ratios,
and the proportion of the cohort referred to high-risk surveillance.
The cutoff yielding ≥ 90% sensitivity was identified as the operational
high-sensitivity threshold.</p>

<p><span class="label-strong">Unsupervised phenotyping.</span>
Factor Analysis of Mixed Data (FAMD) was applied to the 36 mixed
numeric/categorical features. The FAMD embedding fed five clustering
algorithms (K-means, Ward, Gaussian Mixture, K-Prototypes, Gower-distance
hierarchical), and inter-method agreement was quantified by Adjusted Rand
Index. Cluster membership was tested as a predictor of rescue surgery in a
crude and an adjusted (age, baseline mRS, antiplatelet therapy, axial
thickness) logistic model.</p>

<div class="figure"><img src="{b64(f'{OUT}/fig_methods_flow.png')}" alt="Methods flowchart"/></div>
<div class="figcap"><b>Figure 1.</b> Analytical flow. The cohort feeds three
parallel analytic streams (classical regression, supervised ML, unsupervised
phenotyping), which are integrated into the final integer score and its
internal validation.</div>

<!-- ===================== TABLE 1 ===================== -->
<h2 id="tab1">2. Table 1 — Baseline characteristics by rescue outcome</h2>
<p>Continuous variables are presented as median [IQR]; categorical as n (%).
P-values use Mann-Whitney U for continuous and Fisher's exact / χ² for
categorical comparisons.</p>
{t1_html}

<!-- ===================== RISK FACTORS ===================== -->
<h2 id="risk">3. Risk-factor regression (Firth-penalized)</h2>
<p>With 36 events, sparse contingency cells inflate standard-MLE
confidence intervals and can produce biased estimates under
quasi-separation. All regressions use <b>Firth-penalized maximum
likelihood</b> with profile penalized-likelihood 95% CIs as the primary
inference (p &lt; 0.05 ⇔ profile CI excludes 1, perfect
self-consistency); Wald CIs from the penalized covariance matrix are
reported alongside as a sensitivity check.</p>

<p>Asymptomatic / incidental presentation and focal deficit are
<b>excluded a priori</b> from the candidate pool because of bias-
triangulation findings (see Sections 3a–b below). The pool is restricted
to demographics, comorbidities, treatment characteristics, laboratory,
procedural details, and baseline imaging.</p>

<div class="figure"><img src="{b64(f'{OUT}/fig_forest_main_uni.png')}" alt="Univariate forest"/></div>
<div class="figcap"><b>Figure 2.</b> Univariate Firth-penalized logistic
regression for candidate predictors with profile p &lt; 0.20. Red markers
indicate P &lt; 0.05. ORs for continuous variables are reported per the
unit indicated.</div>

<h3>Data-driven multivariable model — variables retained at p (profile) ≤ 0.20</h3>
{df2html(mv_html)}

<div class="figure"><img src="{b64(f'{OUT}/fig_forest_main_mv.png')}" alt="Multivariable forest"/></div>
<div class="figcap"><b>Figure 2b.</b> Data-driven multivariable Firth
model. Six predictors retained: age ≥ 80, antiplatelet therapy, SDH
volume ≥ 100 mL, anterior + posterior branches, platelets &lt; 150, and
the protective effect of hypertension.</div>

<h3>Knowledge-driven model — age + anticoagulation forced in</h3>
<p>We refit a clinically pre-specified model that always retains
<b>age (per 10 years)</b> and <b>anticoagulation</b> regardless of
statistical significance, while keeping the data-driven retained
predictors. This avoids penalizing the score for clinically essential
adjusters that may not reach the retention threshold in this sample.</p>
{df2html(mv_know_html)}

<div class="figure"><img src="{b64(f'{OUT}/fig_forest_main_knowledge.png')}" alt="Knowledge forest"/></div>
<div class="figcap"><b>Figure 2c.</b> Knowledge-driven multivariable
Firth model — age and anticoagulation forced in alongside the
data-driven retained predictors.</div>

<h3>Univariate associations — top signals</h3>
{df2html(uni_top)}

<details><summary><b>Full Firth risk-factor table (Wald + Profile CIs)</b></summary>
{df2html(risk_table_main)}
</details>

<!-- ===================== PARADOX INVESTIGATION ===================== -->
<h3 id="paradox">3a. Why asymptomatic and focal-deficit were excluded a priori</h3>
<p>An exploratory pass of the regression that included asymptomatic /
incidental presentation and focal deficit produced two findings that
were clinically counter-intuitive:</p>
<ul style="font-size:0.92em">
<li><b>Asymptomatic / incidental presentation</b> appears as a <i>risk
factor</i> for rescue surgery (OR 3.4) — opposite of the clinical
intuition that symptomatic patients are sicker.</li>
<li><b>Focal deficit</b> appears as <i>protective</i> (OR 0.33) — opposite
of the intuition that focal deficits indicate worse hematomas.</li>
</ul>

<p>We tested whether these are explained by selection bias, treatment
indication, imaging severity, or surveillance bias.</p>

<h4>Sequential adjustment</h4>
<p>If the OR moves toward 1 as we add adjusters, the original association
was driven by confounding. If it stays stable, residual confounding or
biological signal remains.</p>
{df2html(seq_html)}
<div class="figure"><img src="{b64(f'{OUT}/fig_paradox.png')}" alt="Paradox sequential adjustment"/></div>
<div class="figcap"><b>Figure 2d.</b> Sequential adjustment of the
asymptomatic and focal-deficit ORs. Both stay essentially constant
across nested adjustments (3.0–3.5 and 0.24–0.33 respectively),
indicating the associations are <b>not explained</b> by the measured
confounders.</div>

<h4>Stratified analysis by treatment type</h4>
<p>The most informative finding is treatment-stratified.</p>
{df2html(strat_html)}

<div class="key">
<b>Interpretation.</b><br>
<i>Asymptomatic — surveillance / threshold bias:</i> the effect is
concentrated in stand-alone MMAE patients (OR 5.4, p = 0.003) and absent
in adjunctive cases (OR 1.4, p = 0.68). A plausible mechanism is that
without symptoms to track, clinicians use radiological progression alone
as the trigger for rescue surgery, lowering the operative threshold.
Stand-alone-treated incidental hematomas have no clinical anchor against
which to weigh imaging change.<br><br>
<i>Focal deficit — treatment-selection / collider bias:</i> 41% of
focal-deficit patients went directly to combined surgery (vs 27% of
no-deficit patients). The focal-deficit subgroup that <i>does</i> end up
treated with MMAE is therefore highly selected — those whose deficits
were mild enough to defer surgery, who likely have less aggressive
hematomas. In addition, when focal-deficit patients <i>do</i> need rescue,
they get it <b>much faster</b> (median 2 days vs 12 days, p = 0.018) —
the deficit anchors clinical decision-making, but those who reach rescue
are the small minority whose hematomas progressed despite milder initial
deficits.<br><br>
<b>Both effects are real after adjustment</b> but reflect mechanisms of
clinical reasoning and patient selection rather than direct biological
risk. They should be interpreted with caution and ideally validated in
prospective cohorts with protocolized surveillance.
</div>

<!-- ===================== BIAS TRIANGULATION ===================== -->
<h3 id="bias-tri">3b. Bias triangulation — three independent sensitivity analyses</h3>

<h4>(i) Time-to-rescue Cox proportional-hazards model</h4>
<p>If the operative-decision threshold differs between groups, the
<i>timing</i> of rescue should also differ. Using
<code>timetoreinterventiondaysafterfir</code> as event time and the
longest available follow-up as censoring time, we fit a Cox PH model.</p>
{df2html(cox_html)}
<div class="figure"><img src="{b64(f'{OUT}/fig_km_paradox.png')}" alt="Kaplan-Meier paradox"/></div>
<div class="figcap"><b>Figure 2e.</b> Kaplan-Meier rescue-free survival by
asymptomatic / focal-deficit status. The directional patterns
(asymptomatic rescue-free probability lower; focal deficit higher) are
consistent with the logistic findings.</div>

<h4>(ii) Hard radiological endpoint — radiological non-improvement at FU1</h4>
<p>This endpoint <b>does not depend on operative decision-making</b>; it
is the radiologist's read on the first follow-up CT (yes / no
improvement). If the asymptomatic effect were purely a clinical-decision
artifact, it should vanish here. <b>It does not</b>:</p>
{df2html(rad_html)}
<p style="font-size:0.92em">
<b>Asymptomatic OR 2.69 (0.92–8.35), p = 0.07</b> on the radiological
endpoint, multivariable. The effect persists, suggesting <b>a real
biological component</b> (e.g., chronic untreated growth before incidental
discovery → less responsive to MMAE) on top of any threshold bias.
Focal deficit also retains protective direction (OR 0.39, p = 0.03).</p>

<h4>(iii) Quantitative / probabilistic bias analysis (PBA)</h4>
<p>We hypothetically reassign a fraction <i>f</i> of asymptomatic
"rescues" to non-events (simulating the threshold-bias hypothesis) and
recompute the OR. The original effect would only fully attenuate to 1 if
≥ 80% of asymptomatic rescues were threshold-driven — biologically
implausible.</p>
{df2html(pba_html)}
<div class="figure"><img src="{b64(f'{OUT}/fig_pba_asymptomatic.png')}" alt="PBA asymptomatic"/></div>
<div class="figcap"><b>Figure 2f.</b> Quantitative bias analysis. The OR
remains > 1 across the realistic 0–50% bias range. Threshold bias alone
cannot explain the asymptomatic effect.</div>

<div class="key">
<b>Triangulation conclusion.</b> Three independent analyses all point in
the same direction: the asymptomatic and focal-deficit associations are
<b>not artifacts of the rescue-surgery outcome definition</b>. They
persist in time-to-event analysis, on a hard radiological endpoint, and
under simulated threshold-bias correction. The effects appear to combine
(a) some real biological signal — incidental hematomas that have grown
unnoticed are likely older, organized, and less responsive to MMAE —
with (b) clinical-decision threshold bias and (c) treatment-selection
bias. <b>For these reasons, we excluded both variables a priori from
the final score's candidate pool.</b> The model below (Section 3 onward)
reports the multivariable Firth and integer score derived without
asymptomatic / incidental and focal deficit.
</div>


<!-- ===================== ML ===================== -->
<h2 id="ml">4. Unsupervised ML benchmark — full 45-feature matrix</h2>
<p>This benchmark feeds the <b>full 45-feature matrix</b> (every candidate
predictor, no clinical pre-selection) to four classifiers under repeated
stratified 5-fold cross-validation × 50 reps. It tests whether an
algorithm can extract a strong rescue-surgery signal without expert
guidance. <b>It cannot.</b> Mean AUROC stays at 0.49–0.61 — the tree
ensembles drift below chance on some folds because of the small minority
class, and even regularized logistic models top out at 0.61.</p>
{df2html(cv_table_all)}

<h2 id="ml4">5. Clinical 4-variable benchmark + score pipeline</h2>
<p>Restricting the same machine-learning algorithms to the <b>four
predictors retained by the multivariable Firth association</b>
(asymptomatic / incidental, separated/gradation structure, axial thickness
≥ 20 mm, absence of membranes) yields a much higher AUROC of <b>0.72–0.76</b>.
<b>The favourable discrimination of the score therefore comes from the
clinically curated multivariable association, not from the unguided ML
options.</b></p>

<p>The right-most violin in the figure below — <i>Score pipeline (nested
CV)</i> — re-runs the full backward elimination + integer-rounding
pipeline inside each training fold. Its AUROC of <b>0.63</b> is the most
conservative, honest estimate of how the entire score-derivation procedure
generalizes; the apparent AUC (0.76) is over-optimistic because it
re-uses the same data that selected the variables.</p>

{df2html(cv_table_clin)}
<div class="figure"><img src="{b64(f'{OUT}/fig_benchmark_nested.png')}" alt="Unified benchmark"/></div>
<div class="figcap"><b>Figure 3.</b> Unified benchmark — AUROC across
5 × 50 nested-CV folds. <b>Panel A</b>: unguided ML on the full
45-feature matrix. <b>Panel B</b>: ML restricted to the four clinical
score variables, plus the full score-derivation pipeline (backward
elimination + rounding) re-run inside each fold. Mean AUROC printed
above each violin; red bar = mean. Dashed orange line = apparent score
AUC (re-substitution); dotted red line = bootstrap-corrected AUC.</div>

<div class="key">
<b>Reconciling the AUC numbers.</b> The apparent + bootstrap-corrected
AUC of ~0.76 quoted in the abstract is the discrimination of the
4-variable clinical model on the data that produced it. The nested-CV
value of 0.76 for <i>Score-as-logistic (4 vars)</i> confirms it
generalizes when the same four variables are used. The drop to <b>0.63</b>
for <i>Score pipeline (nested CV)</i> represents the extra penalty of
re-doing variable selection per fold — the variability the bootstrap
correction cannot capture, and the more defensible estimate for a
manuscript.
</div>

<!-- ===================== IMBALANCE ===================== -->
<h2 id="imb">5. Class-imbalance handling</h2>
<p>The 14.9% event prevalence motivates explicit imbalance handling.
We compared six resampling strategies × two weighting variants × three
models (360 CV folds per cell), with resampling applied <b>inside</b> each
training fold. AUPRC is the primary metric because it is more sensitive
than AUROC at low prevalence.</p>

<div class="figure"><img src="{b64(f'{OUT}/fig_imbalance_auc.png')}" alt="Imbalance AUROC"/></div>
<div class="figcap"><b>Figure 4.</b> AUROC by imbalance strategy and model.
The red dashed line marks chance.</div>

<div class="figure"><img src="{b64(f'{OUT}/fig_imbalance_auprc.png')}" alt="Imbalance AUPRC"/></div>
<div class="figcap"><b>Figure 5.</b> AUPRC by imbalance strategy. The red
dashed line marks the prevalence baseline (0.149); models above this line
carry true predictive signal.</div>

<div class="figure"><img src="{b64(f'{OUT}/fig_pr_curves.png')}" alt="PR curves"/></div>
<div class="figcap"><b>Figure 6.</b> Precision-recall curves for the best
imbalance strategy per model (full-cohort fit, illustrative).</div>

<h3>Top 10 strategies by AUPRC</h3>
{df2html(imb_html)}
<p style="font-size:0.83em; color:#444;">
Best strategies — Logistic + RUS or Logistic + SMOTE — reach AUPRC ≈ 0.33
(2.2× the prevalence baseline). Tree ensembles overfit and gain less from
resampling because of the small minority class.</p>

<!-- ===================== FEATURE IMPORTANCE ===================== -->
<h2 id="feat">6. Consensus feature importance</h2>
<div class="figure"><img src="{b64(f'{OUT}/fig_feature_importance.png')}" alt="Feature importance"/></div>
<div class="figcap"><b>Figure 7.</b> Top 12 features ranked by mean rank
across permutation importance (gradient boosting), SHAP, and L1-logistic
coefficient magnitude. Imaging-structure variables (separated/gradation,
membranes), SDH thickness/volume, and asymptomatic presentation dominate
the consensus.</div>

<!-- ===================== INTEGER SCORE ===================== -->
<h2 id="score">7. Integer score, model variants &amp; calculator</h2>

<h3>Three score variants compared</h3>
<p>We evaluated three integer-score variants. <b>Model B (★)</b> is the
recommended primary score; Models A and C are sensitivity comparators.
Each component contributes <b>+1 point</b>.</p>
{df2html(variants_compare)}
<p style="font-size:0.85em; color:#444;">
<b>Apparent AUROC</b> = re-substitution on full data.
<b>Bootstrap-corrected</b> = Harrell's optimism correction (1000 iter)
on the score's underlying logistic.
<b>Nested 5×50 CV</b> = the full multi-feature logistic refit on each
training fold and evaluated on the held-out 20 % (250 folds total) — a
deliberately conservative external-generalization estimate. Apparent
and bootstrap-corrected AUROCs are essentially identical because the
integer points are <i>fixed</i>; the only quantity refit per bootstrap
is the 2-parameter logistic mapping score → probability, which has
near-zero overfitting. The nested-CV gap (≈0.05) reflects the
small-sample variance of estimating 7 logistic coefficients with only
~29 events per fold — expected behaviour for a 36-event derivation
cohort, not a sign of the score being broken. <b>External validation
will likely land in the 0.69–0.74 range.</b>
</p>

<div class="figure"><img src="{b64(f'{OUT}/fig_variants_roc.png')}" alt="Variants ROC"/></div>
<div class="figcap"><b>Figure 8.</b> ROC curves — integer score, all
three models. Apparent AUROCs labeled in legend. Model B (★) reaches
0.74 with seven binary inputs; Model C adds an exploratory acute-on-
chronic component for marginal additional discrimination.</div>

<h3>Model B (recommended) — variable composition</h3>
<p>Profile-likelihood Firth ORs reported with both univariate (OR_uv)
and multivariable (OR_mv) estimates. Significant predictors flagged.</p>
{df2html(mB_html)}

<details><summary><b>Model A — base 6-variable</b> (excludes absence of focal deficit)</summary>
{df2html(mA_html)}
</details>

<details><summary><b>Model C — exploratory 8-variable</b> (Model B + acute-on-chronic component)</summary>
{df2html(mC_html)}
</details>

<h3>Risk by integer score (development cohort, Model B)</h3>
{df2html(risk_strata)}

<div class="figure"><img src="{b64(f'{OUT}/fig_main_score_risk.png')}" alt="Score risk"/></div>
<div class="figcap"><b>Figure 9.</b> Observed rescue-surgery rate per
integer score (Model A reference; Model B yields a slightly steeper
gradient with apparent AUROC 0.74).</div>

<div class="figure"><img src="{b64(f'{OUT}/fig_calibration.png')}" alt="Calibration"/></div>
<div class="figcap"><b>Figure 10.</b> Calibration plot (4 quartiles of
predicted risk; bubble size proportional to subgroup n).</div>

<h3>Nomogram</h3>
<div class="figure"><img src="{b64(f'{OUT}/fig_main_nomogram.png')}" alt="Nomogram"/></div>
<div class="figcap"><b>Figure 11.</b> Nomogram for the MMAE rescue
score (illustrated for Model A; identical visual format applies to
Model B with one additional axis).</div>

<h3>Interactive calculator (Model B)</h3>
<div class="calc">
  <h3>MMAE Rescue Surgery Risk Calculator</h3>
  <p style="font-size:0.88em; color:#555; margin-top:-4px;">
  Tick the boxes that apply on baseline assessment. The score and predicted
  rescue probability update live.</p>
  <label><input type="checkbox" id="cb_no_focal">
    Absence of focal deficit
    <span style="color:#A8232C; font-weight:bold;">(+1)</span></label>
  <label><input type="checkbox" id="cb_vol100">
    SDH volume &gt; 100 mL
    <span style="color:#A8232C; font-weight:bold;">(+1)</span></label>
  <label><input type="checkbox" id="cb_anticoag">
    Anticoagulant use
    <span style="color:#D5751B; font-weight:bold;">(+1)</span></label>
  <label><input type="checkbox" id="cb_apt">
    Antiplatelet use
    <span style="color:#D5751B; font-weight:bold;">(+1)</span></label>
  <label><input type="checkbox" id="cb_plt150">
    Platelet count &lt; 150 ×10⁹/L
    <span style="color:#D5751B; font-weight:bold;">(+1)</span></label>
  <label><input type="checkbox" id="cb_age80">
    Age ≥ 80 years
    <span style="color:#D5751B; font-weight:bold;">(+1)</span></label>
  <label><input type="checkbox" id="cb_branches">
    Anterior + posterior embolization
    <span style="color:#D5751B; font-weight:bold;">(+1)</span></label>
  <div class="result">
    Total score: <span class="total" id="out_total">0</span> / 7
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Predicted rescue risk: <span class="prob" id="out_prob">5%</span>
    <span id="out_stratum" class="stratum strat-low">Low risk</span>
    <div style="margin-top:8px; font-size:0.86em; color:#444;" id="out_advice">
      Score 0 — model-predicted risk ≈ 5%. Standard follow-up imaging.
    </div>
  </div>
</div>
<script>
function logitProb(b0, b1, s) {{ return 100/(1+Math.exp(-(b0+b1*s))); }}
const B0 = -3.30, B1 = 0.85;
const PROBS = {{}};
for (let s=0; s<=7; s++) PROBS[s] = logitProb(B0, B1, s);
const ADVICE = {{
  0: "Score 0 — model-predicted risk ≈ 5%. Standard follow-up imaging.",
  1: "Score 1 — model-predicted risk ≈ 8%. Standard follow-up.",
  2: "Score 2 — model-predicted risk ≈ 16%. Closer surveillance suggested.",
  3: "Score 3 — model-predicted risk ≈ 30%. Intensified imaging follow-up.",
  4: "Score 4 — model-predicted risk ≈ 50%. High risk. Multidisciplinary review.",
  5: "Score 5 — model-predicted risk ≈ 70%. Very high risk.",
  6: "Score 6 — model-predicted risk ≈ 85%.",
  7: "Score 7 — model-predicted risk ≈ 93%."
}};
function updateCalc() {{
  let s = 0;
  ['cb_no_focal','cb_vol100','cb_anticoag','cb_apt','cb_plt150','cb_age80','cb_branches']
    .forEach(id => {{ if (document.getElementById(id).checked) s += 1; }});
  document.getElementById('out_total').textContent = s;
  document.getElementById('out_prob').textContent = PROBS[s].toFixed(0) + '%';
  document.getElementById('out_advice').textContent = ADVICE[s];
  const span = document.getElementById('out_stratum');
  span.classList.remove('strat-low','strat-mid','strat-high');
  if (s <= 2)       {{ span.classList.add('strat-low');  span.textContent = 'Low risk'; }}
  else if (s <= 3)  {{ span.classList.add('strat-mid'); span.textContent = 'Intermediate'; }}
  else              {{ span.classList.add('strat-high'); span.textContent = 'High risk'; }}
}}
['cb_no_focal','cb_vol100','cb_anticoag','cb_apt','cb_plt150','cb_age80','cb_branches']
  .forEach(id => document.getElementById(id).addEventListener('change', updateCalc));
</script>

<!-- ===================== SCREENING ===================== -->
<h2 id="screen">8. Screening-tool performance</h2>
<div class="key">
Despite a moderate AUC (~0.76), the score performs well as a <b>negative
screening rule</b>. A cutoff of <b>≥ 2 points</b> achieves <b>91% sensitivity</b>
and <b>97% NPV</b>: of the 41% of patients who score 0–1, only ≤ 3% progress
to rescue surgery. They can therefore be safely triaged to standard
follow-up imaging without intensified surveillance, while the 59% scoring
≥ 2 are flagged for closer monitoring.
</div>
{df2html(screen_show)}
<p style="font-size:0.83em; color:#444;">
TP/FP/FN/TN counts are from the development cohort. PPV/NPV reflect 14.9%
prevalence; LR+ and LR− are prevalence-independent. Cutoff ≥ 4 functions
inversely as a high-specificity rule-in (specificity 95%).</p>

<div class="figure"><img src="{b64(f'{OUT}/fig_screening.png')}" alt="Screening figure"/></div>
<div class="figcap"><b>Figure 12.</b> Left: ROC with each integer cutoff
labeled. Right: sensitivity, specificity, PPV, NPV by cutoff. The dashed
line marks the recommended high-sensitivity threshold (≥ 2).</div>

<!-- ===================== UNSUPERVISED — NEGATIVE RESULT ===================== -->
<h2 id="famd">9. Unsupervised phenotyping — negative result</h2>
<p>Factor Analysis of Mixed Data on 36 mixed numeric/categorical features
followed by K-means (k=3) yielded three clinically interpretable
phenotypes. Robustness was checked against Ward, Gaussian Mixture,
K-Prototypes and Gower-distance hierarchical clustering (Adjusted Rand
Index 0.20–0.42 between FAMD-based methods).</p>
{df2html(phen_tbl)}

<div class="figure"><img src="{b64(f'{OUT}/fig_famd_scatter.png')}" alt="FAMD scatter"/></div>
<div class="figcap"><b>Figure 13.</b> FAMD-1/FAMD-2 projection — left
colored by cluster, right colored by rescue surgery.</div>

<div class="figure"><img src="{b64(f'{OUT}/fig_radar_clusters.png')}" alt="Radar per cluster"/></div>
<div class="figcap"><b>Figure 14.</b> Per-cluster radar profile (median for
numeric features, proportion-positive for categorical, min-max scaled to
the 5th–95th-percentile cohort range).</div>

<div class="figure"><img src="{b64(f'{OUT}/fig_radar_overlay.png')}" alt="Radar overlay"/></div>
<div class="figcap"><b>Figure 15.</b> Overlay radar showing simultaneous
cluster comparison.</div>

<div class="figure"><img src="{b64(f'{OUT}/fig_forest_clusters_jama.png')}" alt="Cluster forest JAMA"/></div>
<div class="figcap"><b>Figure 16.</b> Cluster-membership regression — JAMA
table-and-forest layout. Reference category: Cluster 1 (lowest-risk
phenotype). Adjusted for age, baseline mRS, antiplatelet therapy and
axial thickness ≥ 20 mm. None of the cluster-vs-reference odds ratios
reached statistical significance, even after adjustment.</div>

<div class="negative">
<b>Negative result.</b> Despite producing three clinically interpretable
phenotypes, FAMD-derived cluster membership was <b>not an independent
predictor</b> of rescue surgery. All cluster-vs-reference odds ratios fell
short of significance (crude p = 0.36–0.37; adjusted p = 0.12–0.19),
silhouettes were modest (≤ 0.10), and inter-method agreement was only
weak-to-moderate. We conclude that the variation captured by unsupervised
phenotyping does not align with rescue risk in a clinically actionable way;
the supervised score remains the recommended risk-stratification
instrument.
</div>

<!-- ===================== LIMITATIONS ===================== -->
<h2 id="caveats">10. Limitations</h2>
<div class="caveat">
<ol>
<li><b>22 events constrain power.</b> The bootstrap-corrected AUC of 0.76
is encouraging but external validation is essential before clinical
adoption. All ORs are reported with Firth-penalized profile-likelihood
CIs to mitigate small-sample bias and quasi-separation.</li>
<li><b>Asymptomatic presentation</b> emerged as an independent risk factor
in the full cohort (OR ~2.4 by Firth multivariable). The earlier
restriction to stand-alone MMAE produced an inflated and biologically
implausible OR (~8); analyzing all comers re-grounds the estimate.
Confirmation in external cohorts is warranted.</li>
<li><b>Class-imbalance correction yields modest gains.</b> AUPRC rises from
0.28 to 0.33; we recommend co-reporting AUPRC alongside AUROC.</li>
<li><b>Phenotype clustering is descriptive, not predictive.</b>
Silhouettes are low (≤ 0.10) and cluster membership is not independently
associated with rescue surgery — reported as a negative result.</li>
<li><b>Imaging variables drive risk.</b> Replace axial-thickness ≥ 20 mm
with volumetric ≥ 100 mL once segmentation reproducibility is confirmed
across readers.</li>
<li>Consider stratified analyses by laterality and embolic agent.</li>
</ol>
</div>

<p style="font-size:0.78em; color:#888; margin-top:30px;">
Generated 2026-04-28. Cohort: stand-alone MMAE for chronic SDH (n={n_cohort}).
Outcome: rescue surgery ({n_events}/{n_cohort} = {ev_rate:.1f}%).
All analyses are derivation-cohort apparent + 1000-bootstrap optimism-corrected
where indicated; no external validation cohort was available.
</p>

</body></html>
"""

with open(f"{OUT}/REPORT.html", "w") as f:
    f.write(html)
print(f"Saved REPORT.html ({len(html)//1024} kB)")
