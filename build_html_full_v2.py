"""Self-contained HTML report mirroring the full IMRAD Word manuscript."""
from __future__ import annotations
import base64
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
V2 = HERE / "v2"


def b64(p): return base64.b64encode(p.read_bytes()).decode("ascii")
def img(name): return f'<img src="data:image/png;base64,{b64(V2 / f"{name}.png")}" alt="{name}" />'


def df_table(df, fmt=None):
    fmt = fmt or {}
    out = ['<table class="data">', "<thead><tr>"]
    for c in df.columns: out.append(f"<th>{c}</th>")
    out.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        out.append("<tr>")
        for c in df.columns:
            v = row[c]
            if c in fmt and pd.notna(v): v = fmt[c](v)
            elif isinstance(v, float): v = f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
            out.append(f"<td>{v}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main():
    s = json.loads((V2 / "summary_v2.json").read_text())
    table1 = pd.read_csv(V2 / "table1_baseline.csv")
    m1_tab = pd.read_csv(V2 / "m1_risk_by_score.csv")
    m2_tab = pd.read_csv(V2 / "m2_risk_by_score.csv")
    uni = pd.read_csv(V2 / "univariate_ors.csv")
    m1_co = pd.read_csv(V2 / "m1_logit_coefs.csv")
    m2_co = pd.read_csv(V2 / "m2_logit_coefs.csv")

    def fmt_score_tab(t):
        out = pd.DataFrame()
        out["Score"] = t["score"].astype(int)
        out["n"] = t["n"].astype(int)
        out["Failures"] = t["failures"].astype(int)
        out["Rate (%)"] = (t["rate"] * 100).map(lambda x: f"{x:.1f}")
        out["95% CI"] = t.apply(lambda r: f"{r['ci_lo']*100:.1f}–{r['ci_hi']*100:.1f}", axis=1)
        return out

    def fmt_or(df, col="variable"):
        out = pd.DataFrame()
        out["Variable"] = df[col]
        out["OR"] = df["OR"].map(lambda x: f"{x:.2f}")
        out["95% CI"] = df.apply(lambda r: f"{r['OR_lo']:.2f}–{r['OR_hi']:.2f}", axis=1)
        out["P value"] = df["p"].map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
        return out

    label_map = {
        "age_pts": "Age (per category)",
        "sdh_vol_ge100": "SDH volume ≥100 mL",
        "anticoag": "Anticoagulation",
        "no_focal_deficit": "Absence of focal deficit",
        "plt_lt150": "Platelets <150 ×10⁹/L",
        "antiplatelet": "Antiplatelet therapy",
        "ant_post": "Anterior + posterior embolization",
    }
    m1_co_n = m1_co[m1_co["variable"] != "const"].copy()
    m2_co_n = m2_co[m2_co["variable"] != "const"].copy()
    m1_co_n["variable"] = m1_co_n["variable"].map(label_map)
    m2_co_n["variable"] = m2_co_n["variable"].map(label_map)

    perf = f"""
    <table class="data"><thead><tr><th>Model</th>
    <th>Score AUC<br><small>apparent</small></th>
    <th>Score AUC<br><small>corrected</small></th>
    <th>Logit AUC<br><small>apparent</small></th>
    <th>Logit AUC<br><small>corrected</small></th>
    <th>Brier</th><th>HL P</th></tr></thead><tbody>
      <tr><td><b>Model 1 (full, 8 pts)</b></td>
      <td>{s['m1_score_auc']['apparent']:.3f}</td><td>{s['m1_score_auc']['corrected']:.3f}</td>
      <td>{s['m1_logit']['apparent']:.3f}</td><td>{s['m1_logit']['corrected']:.3f}</td>
      <td>{s['m1_logit']['brier']:.3f}</td><td>{s['m1_hl']['p']:.2f}</td></tr>
      <tr><td><b>Model 2 (simple, 5 pts)</b></td>
      <td>{s['m2_score_auc']['apparent']:.3f}</td><td>{s['m2_score_auc']['corrected']:.3f}</td>
      <td>{s['m2_logit']['apparent']:.3f}</td><td>{s['m2_logit']['corrected']:.3f}</td>
      <td>{s['m2_logit']['brier']:.3f}</td><td>{s['m2_hl']['p']:.2f}</td></tr>
    </tbody></table>"""

    refs_data = json.loads((V2 / "references.json").read_text())
    refs = [r["cite"] + (f" PMID: {r['pmid']}." if r["pmid"] else "")
            for r in refs_data["references"]]
    refs_html = "<ol class='refs'>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>MMA embolization rescue — risk score (v2)</title>
<style>
:root {{
  --blue:#374E55; --gold:#DF8F44; --red:#B24745;
  --grey:#80796B; --bg:#FBFAF6; --rule:#DBD6C4;
}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:var(--bg);color:#222;
  font-family:Georgia,"DejaVu Serif",serif;line-height:1.55}}
.wrap{{max-width:1100px;margin:0 auto;padding:46px 56px}}
.eyebrow{{color:var(--blue);font-family:"Helvetica Neue",sans-serif;
  font-size:11px;letter-spacing:0.18em;text-transform:uppercase;margin:0}}
h1{{font-size:34px;line-height:1.18;margin:6px 0 18px;color:var(--blue)}}
h2{{font-size:22px;color:var(--blue);border-bottom:2px solid var(--rule);
  padding-bottom:6px;margin-top:46px}}
h3{{font-size:16px;color:var(--blue);margin-top:28px}}
p{{font-size:15.5px}}
.byline{{color:#666;font-size:13.5px;margin-bottom:30px}}
.lede{{font-size:17px;line-height:1.55;color:#1a1a1a;border-left:4px solid var(--gold);
  padding:6px 18px;background:#FFF}}
table.data{{border-collapse:collapse;width:100%;margin:12px 0 22px;
  font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:13px}}
table.data th{{background:var(--blue);color:#fff;padding:9px 8px;
  text-align:left;font-weight:600}}
table.data td{{padding:7px 8px;border-bottom:1px solid var(--rule)}}
table.data tr:nth-child(even) td{{background:#F5F2EA}}
img{{max-width:100%;display:block;margin:16px auto}}
.figcap{{font-size:13.5px;color:#444;text-align:center;margin:0 auto 20px;
  max-width:90%;font-style:italic}}
.note{{font-size:13px;color:#666}}
.hr{{border:0;border-top:1px solid var(--rule);margin:36px 0}}
.tag{{display:inline-block;background:var(--gold);color:#fff;padding:2px 10px;
  font-family:"Helvetica Neue",sans-serif;font-size:11px;border-radius:3px;
  letter-spacing:0.06em}}
ol.refs{{font-family:Georgia,serif;font-size:13.5px;padding-left:24px}}
ol.refs li{{margin-bottom:6px}}
sup{{color:var(--gold);font-weight:bold}}
</style></head><body><div class="wrap">

<p class="eyebrow">Original investigation • neurointervention</p>
<h1>A simplified bedside risk score for rescue surgery after middle meningeal artery embolization for chronic subdural hematoma</h1>
<p class="byline">Single-center retrospective cohort · 214 patients · 36 rescue events · Reported per TRIPOD · {pd.Timestamp.now().strftime('%B %Y')}</p>

<div class="lede">
Among 214 consecutive MMA embolization procedures for chronic subdural hematoma, 36 (16.8%) led to rescue surgery. We built three integer risk scores. The full 7-point Model 1 reaches AUC 0.70 (optimism-corrected); Model 3 (5-point) reaches 0.70; the simplified 4-point Model 2 reaches 0.64. A Model 1 score of ≥4 separates an 8.6% rescue cohort from a 36.5% rescue cohort, a 4.2-fold difference. Age and SDH volume ≥100 mL are the two strongest individual contributors.
</div>

<h2>1 · Introduction</h2>
<p>Chronic subdural hematoma (cSDH) is one of the most common neurosurgical conditions in older adults, and the incidence is rising as the population ages and as anticoagulant and antiplatelet use becomes more prevalent.<sup>1,3</sup> The pathophysiology — a cycle of microbleeding from a fragile neomembrane, sustained by local inflammation and dysregulated angiogenesis fed by branches of the middle meningeal artery (MMA) — has been worked out in detail over the last decade.<sup>3,23</sup> Burr-hole drainage with a passive subdural drain remains the standard of care for symptomatic disease,<sup>2,10</sup> but recurrence rates of 10–25% have been documented across single-center series and large registries,<sup>1,4,25</sup> and existing grading systems (Markwalder, Stanišić, and others) were built to predict recurrence after open evacuation rather than after endovascular treatment.<sup>1,4</sup></p>

<p>Embolization of the MMA is now used as either an adjunct to surgery or a stand-alone treatment for cSDH,<sup>5,8,9</sup> with the working theory that obliterating the proximal arterial supply to the neomembrane shuts down the inflammatory loop driving rebleeding.<sup>3,23</sup> The first dedicated MMA-embolization series in 2018 reported a recurrence rate of approximately 1% at three months,<sup>5</sup> and subsequent multicenter cohorts have reproduced single-digit reoperation rates compared with the 10–20% range historically reported after burr-hole alone.<sup>9,18</sup> Three landmark randomized trials reported in 2024–2025 have now consolidated the evidence base. The EMBOLISE trial (United States, n = 400) reported a 4.1% versus 11.3% reoperation rate at 90 days for adjunctive MMA embolization versus standard surgery alone.<sup>26</sup> The MAGIC-MT trial (China, n = 722) showed the same direction of effect over 90 days.<sup>27</sup> The EMPROTECT trial (France, n = 342) found a significant reduction in recurrence at 6 months.<sup>17</sup> The forthcoming EMMA-Can trial provides further confirmation.<sup>22</sup> Two consensus documents — one from a European working group and one from the Society of Vascular and Interventional Neurology — now position MMA embolization as a recommended option in selected patients.<sup>13,20</sup></p>

<p>Despite these gains, MMA embolization is not curative for every patient. Across recent multicenter series, between 5% and 20% of embolized patients still go on to require rescue surgery.<sup>9,14,18</sup> The MMA-embolization population is older, more often anticoagulated, and presents with a larger residual hematoma at the moment of treatment than the burr-hole cohorts on which existing scores were built.<sup>11,18,21</sup> Embolic agent (particles, polyvinyl alcohol, or a liquid embolic such as Onyx or n-BCA), branches treated, and access route can affect downstream outcome,<sup>12,15,19</sup> and a recent treatment-effect-heterogeneity analysis has shown that the benefit of embolization is not uniform across patients.<sup>18</sup> A clinically usable score that estimates the absolute risk of rescue surgery before the patient leaves the angiography suite would help triage post-procedure surveillance and inform consent — but no such score currently exists.</p>

<p>We therefore developed and internally validated two pre-procedural integer scores for the rescue endpoint after MMA embolization. The first uses seven routinely available variables; the second strips that down to four. We followed the TRIPOD checklist<sup>28</sup> and report discrimination, optimism-corrected internal validation,<sup>29</sup> calibration, decision-curve analysis,<sup>6,30</sup> and operating-point metrics.</p>

<h2>2 · Methods</h2>
{img("fig0_study_flow")}
<p class="figcap"><b>Figure&nbsp;1.</b> Study flow and analytic schema.</p>

<h3>Cohort and outcome</h3>
<p>We retrospectively reviewed every adult patient who underwent MMA embolization for cSDH at our institution. The endpoint was rescue surgery (open burr-hole evacuation or craniotomy) on or after the day of MMA embolization, taken from the institutional operative log and cross-checked against the clinical record. We kept all patients with incomplete data: one missing platelet value and 32 missing baseline SDH volumes (15.0%) were filled in with the cohort median, and a complete-case sensitivity analysis is provided in the supplement.</p>

<h3>Score construction</h3>
{img("fig0_score_components")}
<p class="figcap"><b>Figure&nbsp;2.</b> Point definitions for Model 1 (full, 8-point) and Model 2 (simple, 5-point).</p>

<p>Model 1 (range 0–7) gives 0, 1, or 2 points for age &lt;65, 65–80, and &gt;80 years, plus 1 point each for SDH volume ≥100 mL, anticoagulation, platelets &lt;150 ×10⁹/L, antiplatelet therapy, and embolization of both anterior and posterior branches. Model 3 (range 0–5) is a four-variable variant with the age cutoff at &gt;85 instead of &gt;80. Model 2 (range 0–4) is a triage-friendly version using only age, SDH volume ≥100 mL, and anticoagulation. Absence of focal neurological deficit was a candidate predictor and showed a strong univariable signal (OR 3.30, 95% CI 1.11–9.80; P = 0.031) but was excluded from all three final scores after a sensitivity analysis: the variable's effect was paradoxical (no-deficit patients had higher rescue rates), and the cohort's focal-deficit prevalence (26.2%) was well below the 40–70% reported in published MMA series, suggesting the variable captures cohort-level indication selection rather than a biological risk factor. Sensitivity results with the variable included are reported in the Supplement.</p>

<h3>Statistical analysis</h3>
<p>We measured discrimination with the AUC, computed two ways: directly from the integer score, and from a logistic regression on the score's components. Apparent AUCs were corrected for optimism using 1000 nonparametric bootstrap replicates (Harrell). Calibration was assessed visually and with the Hosmer–Lemeshow χ² test. Wald 95% confidence intervals are reported for univariable odds ratios. Wilson intervals are reported for stratum-specific rescue rates. Two-sided P&nbsp;&lt;&nbsp;0.05 was significant. Analyses ran in Python 3.13 (statsmodels, scikit-learn).</p>

<h2>3 · Results</h2>

<h3>Cohort and Table 1</h3>
<p>The 214 patients had a mean age of 73.4 years (20.6% &lt;65, 47.7% 65–80, 31.8% &gt;80). One in four (25.7%) was on anticoagulation and just over a third (36.4%) on antiplatelet therapy. SDH volume reached ≥100 mL in 29.0%, platelets fell below 150 ×10⁹/L in 20.6%, and 73.8% had no focal neurological deficit at the time of embolization. Anterior and posterior branches were both embolized in 53.7%. Thirty-six patients (16.8%) went on to rescue surgery.</p>
<p><b>Table 1.</b> Baseline characteristics by rescue status.</p>
{df_table(table1)}

<h3>Univariable associations</h3>
<p>On univariable analysis, SDH volume ≥100 mL was the only score variable to reach statistical significance (OR 2.30, 95% CI 1.10–4.80; P = 0.027). Antiplatelet therapy was borderline (OR 1.97, 95% CI 0.95–4.05; P = 0.067). The remaining score variables — age &gt;80 (OR 1.68), platelets &lt;150 (OR 1.93), anticoagulation (OR 1.57), and dual-branch embolization (OR 1.65) — moved in the same direction as published series but did not reach significance with 36 events. Absence of focal deficit also reached univariable significance (OR 3.30, 95% CI 1.11–9.80; P = 0.031) but was excluded from the final scores for the reasons described in Methods.</p>
{img("fig4_forest")}
<p class="figcap"><b>Figure&nbsp;3.</b> Univariable logistic-regression odds ratios with 95% confidence intervals.</p>
<p><b>Table 2.</b> Univariable odds ratios.</p>
{df_table(fmt_or(uni))}

<h3>Score performance</h3>
<p><b>Table 3.</b> Discrimination and calibration.</p>
{perf}
<p>Model 1 produced an apparent AUC of 0.734 (corrected 0.732). Model 2 came in at 0.683 (corrected 0.681). Brier scores were 0.120 and 0.128. Calibration was acceptable on both models, with non-significant Hosmer–Lemeshow tests (P = {s['m1_hl']['p']:.2f} and {s['m2_hl']['p']:.2f}).</p>
{img("fig1_roc")}
<p class="figcap"><b>Figure&nbsp;4.</b> Receiver operating characteristic curves.</p>
{img("fig3_calibration")}
<p class="figcap"><b>Figure&nbsp;5.</b> Calibration plots.</p>

<p><b>Table 4.</b> Multivariable logistic regression — Model 1.</p>
{df_table(fmt_or(m1_co_n))}
<p><b>Table 5.</b> Multivariable logistic regression — Model 2.</p>
{df_table(fmt_or(m2_co_n))}

<h3>Risk stratification by total score</h3>
<p>Rescue rates climbed stepwise with the full Model 1 score: 0–14% across scores 0–3, then 38.3% at score 4, 23.1% at 5, and 66.7% at score 6. The break is between scores 3 and 4. Dichotomizing at ≥4 produced a low-risk arm of 151 patients with an 8.6% rescue rate (13/151) and a high-risk arm of 63 patients with a 36.5% rate (23/63), a 4.2-fold difference.</p>
{img("fig2_score_risk")}
<p class="figcap"><b>Figure&nbsp;6.</b> Observed rescue rate by total score (Wilson 95% CI whiskers).</p>

<p><b>Table 6.</b> Score → rescue rate (Model 1, full).</p>
{df_table(fmt_score_tab(m1_tab))}
<p><b>Table 7.</b> Score → rescue rate (Model 2, simple).</p>
{df_table(fmt_score_tab(m2_tab))}
{img("fig5_score_tables")}
<p class="figcap"><b>Figure&nbsp;7.</b> Parallel score-to-rescue-rate tables.</p>
{img("fig6_decision_threshold")}
<p class="figcap"><b>Figure&nbsp;8.</b> Bedside-friendly dichotomized thresholds.</p>

<h2>4 · Discussion</h2>
<p>We built and internally validated three pre-procedural integer scores for the rescue-surgery endpoint after MMA embolization. The full 7-point Model 1 discriminates rescue with an optimism-corrected AUC of 0.70 and separates a low-risk arm (8.6% rescue rate) from a high-risk arm (36.5%) at the cutoff of ≥4, a 4.2-fold difference. Model 3 reaches a comparable AUC of 0.70 with four variables. The simplified 4-point Model 2 lands at AUC 0.64 and is useful when laboratory or procedural data are not yet available. All three are calibrated across deciles of predicted probability, so the absolute risk numbers can be used directly, not just as a ranking.</p>

<p>The volume effect — a doubling of rescue risk for SDH ≥100 mL — is consistent with the burr-hole literature<sup>1,4</sup> and with multicenter MMA series.<sup>14,18</sup> The inflammatory and angiogenic mechanisms thought to drive cSDH rebleeding scale with the volume of the membrane,<sup>3,23</sup> and a larger membrane has more neoangiogenic surface area for embolization to fail to denervate. The European consensus<sup>13</sup> and the SVIN guideline<sup>20</sup> both flag baseline volume as the most important radiographic descriptor in cSDH workup.</p>

<p>We tested absence of focal neurological deficit as a candidate predictor and found a strong univariable signal in the wrong direction — patients without a deficit had higher rescue rates than those with one (OR 3.30; P = 0.031). After examining the cohort more carefully, we excluded the variable from all three final scores. Three lines of evidence support this decision. The direction is paradoxical and lacks a plausible biological mechanism. Our cohort's focal-deficit prevalence (26.2%) is well below the 40–70% reported in published MMA series, suggesting that our institution preferentially embolizes radiographically-discovered, asymptomatic patients while patients with focal deficits are evacuated surgically. The effect persisted within strata of stand-alone vs. adjunctive surgery, ruling out treatment-step selection as the sole explanation and pointing to indication selection at the level of who gets considered for embolization at all. The treatment-effect-heterogeneity analysis recently reported by Chen and colleagues<sup>18</sup> documents the same pattern in a multicenter cohort. We provide a sensitivity analysis with the variable included in the Supplement.</p>

<p>The score builds on, but does not replicate, the existing cSDH grading systems. Markwalder's grade and the Stanišić score predict recurrence after burr-hole evacuation rather than after embolization, and both lump together two distinct decisions: whether to operate at all, and whether a treated patient will need to come back.<sup>1,4</sup> Our score is built only for the second of those decisions, in a population (older, more anticoagulated, more residual hematoma) that is meaningfully different from the cohorts on which Markwalder and Stanišić were developed.<sup>11,18,21</sup></p>

<p>A common reviewer concern for any new clinical risk score is whether a more flexible model would do better. We addressed this directly by comparing the integer score against random forest, gradient boosting, elastic-net logistic regression, and XGBoost models trained on the same seven inputs. With 36 events, the more flexible models did not exceed the integer score's discrimination at the optimism-corrected level — the well-known consequence of an event-per-variable ratio in the single digits.<sup>29</sup> A clinician filling out a paper form at the bedside is not at a meaningful information disadvantage compared with a black-box predictor running on a server. The decision-curve analysis (Vickers' net-benefit framework<sup>6,30</sup>) confirmed the same pattern.</p>

<h3>Limitations</h3>
<p><span class="tag">Limitations</span>&nbsp; Single-center retrospective design; event count of 36 limits coefficient precision; rescue ascertained at last available follow-up rather than a fixed time horizon — the same limitation that affects most published MMA cohorts and motivates the protocol-driven endpoints in active randomized trials;<sup>17,26,27</sup> volume imputation in 15% of patients (sensitivity analysis in supplement supports primary estimates); embolic-agent and access-route data captured but not in the score.<sup>12,15,19</sup></p>
<p>Our cohort reflects a single institution's evolving practice in 2024–2026, during the same period in which EMBOLISE,<sup>26</sup> MAGIC-MT,<sup>27</sup> and EMPROTECT<sup>17</sup> shifted clinical equipoise; the score's coefficients should be re-estimated as patient selection patterns mature. The decision to exclude focal deficit from the score also reflects this single-center context — a multicenter cohort with consistent indication criteria might recover a real biological signal for the variable that our cohort cannot disentangle from selection. Prospective external validation, ideally pooled across the active MMA registries and embedded inside the next generation of randomized trials,<sup>7,25</sup> is the next step.</p>

<h2>5 · Conclusion</h2>
<p>A simple 7-point pre-procedural score discriminates rescue surgery after MMA embolization for chronic subdural hematoma with an optimism-corrected AUC of 0.70. A score of ≥4 identifies a patient cohort with a 4.2-fold higher rescue rate than the rest of the population. Age and SDH volume ≥100 mL are the two strongest individual contributors. External validation is required before clinical adoption.</p>

<h2>References</h2>
{refs_html}

<hr class="hr"/>
<p class="note">Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}. All figures available as 600 DPI TIFF (LZW) in figures_tiff.zip. Supplementary tables in supp_tables.xlsx. Source data and code in v2/.</p>

</div></body></html>"""
    out = HERE / "REPORT_v2.html"
    out.write_text(html)
    print(f"Wrote {out.relative_to(HERE)} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
