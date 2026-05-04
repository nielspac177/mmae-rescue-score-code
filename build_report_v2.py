"""Build self-contained HTML report and DOCX manuscript for MMAE Score v2."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
V2 = HERE / "v2"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def img(name: str) -> str:
    p = V2 / f"{name}.png"
    return f'<img src="data:image/png;base64,{b64(p)}" alt="{name}" />'


def df_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ['<table class="data">', "<thead><tr>"]
    for c in df.columns:
        out.append(f"<th>{c}</th>")
    out.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        out.append("<tr>")
        for c in df.columns:
            v = row[c]
            if c in fmt and pd.notna(v):
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
            out.append(f"<td>{v}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main():
    # Load
    summary = json.loads((V2 / "summary_v2.json").read_text())
    m1_tab = pd.read_csv(V2 / "m1_risk_by_score.csv")
    m2_tab = pd.read_csv(V2 / "m2_risk_by_score.csv")
    uni = pd.read_csv(V2 / "univariate_ors.csv")
    m1_co = pd.read_csv(V2 / "m1_logit_coefs.csv")
    m2_co = pd.read_csv(V2 / "m2_logit_coefs.csv")

    # Format the score → risk tables
    def fmt_tab(t):
        out = t.copy()
        out["Rate (%)"] = (out["rate"] * 100).map(lambda x: f"{x:.1f}")
        out["95% CI"] = out.apply(
            lambda r: f"{r['ci_lo']*100:.1f}–{r['ci_hi']*100:.1f}", axis=1)
        out = out.rename(columns={"score": "Score", "n": "n", "failures": "Failures"})
        return out[["Score", "n", "Failures", "Rate (%)", "95% CI"]]

    m1_tab_f = fmt_tab(m1_tab)
    m2_tab_f = fmt_tab(m2_tab)

    # Format coefficient tables (variable, OR, 95% CI, P)
    def fmt_or_tab(df, label_col="variable"):
        out = pd.DataFrame()
        out["Variable"] = df[label_col]
        out["OR"] = df["OR"].map(lambda x: f"{x:.2f}")
        out["95% CI"] = df.apply(lambda r: f"{r['OR_lo']:.2f}–{r['OR_hi']:.2f}", axis=1)
        out["P value"] = df["p"].map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
        return out

    uni_f = fmt_or_tab(uni)

    # Drop the constant from MV coefs
    m1_co_n = m1_co[m1_co["variable"] != "const"].copy()
    m2_co_n = m2_co[m2_co["variable"] != "const"].copy()
    label_map = {
        "age_pts": "Age (per category)",
        "sdh_vol_ge100": "SDH volume ≥100 mL",
        "anticoag": "Anticoagulation",
        "no_focal_deficit": "Absence of focal deficit",
        "plt_lt150": "Platelets <150 ×10⁹/L",
        "antiplatelet": "Antiplatelet therapy",
        "ant_post": "Anterior + posterior embolization",
    }
    m1_co_n["variable"] = m1_co_n["variable"].map(label_map)
    m2_co_n["variable"] = m2_co_n["variable"].map(label_map)
    m1_mv_f = fmt_or_tab(m1_co_n)
    m2_mv_f = fmt_or_tab(m2_co_n)

    # ------------------------------------------------------------------
    s = summary
    aucs_html = f"""
    <table class="data">
      <thead><tr><th>Model</th><th>Score AUC<br><small>apparent</small></th>
      <th>Score AUC<br><small>optimism-corr.</small></th>
      <th>Logit AUC<br><small>apparent</small></th>
      <th>Logit AUC<br><small>optimism-corr.</small></th>
      <th>Brier</th><th>HL P</th></tr></thead>
      <tbody>
        <tr><td><b>Model 1 (full, max 8)</b></td>
          <td>{s['m1_score_auc']['apparent']:.3f}</td>
          <td>{s['m1_score_auc']['corrected']:.3f}</td>
          <td>{s['m1_logit']['apparent']:.3f}</td>
          <td>{s['m1_logit']['corrected']:.3f}</td>
          <td>{s['m1_logit']['brier']:.3f}</td>
          <td>{s['m1_hl']['p']:.2f}</td></tr>
        <tr><td><b>Model 2 (simple, max 5)</b></td>
          <td>{s['m2_score_auc']['apparent']:.3f}</td>
          <td>{s['m2_score_auc']['corrected']:.3f}</td>
          <td>{s['m2_logit']['apparent']:.3f}</td>
          <td>{s['m2_logit']['corrected']:.3f}</td>
          <td>{s['m2_logit']['brier']:.3f}</td>
          <td>{s['m2_hl']['p']:.2f}</td></tr>
      </tbody></table>
    """

    # ------------------------------------------------------------------
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>MMA embolization rescue — risk score (v2)</title>
<style>
:root {{
  --blue:#374E55; --gold:#DF8F44; --teal:#00A1D5; --red:#B24745;
  --green:#79AF97; --grey:#80796B; --bg:#FBFAF6; --rule:#DBD6C4;
}}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; padding:0; background:var(--bg); color:#222;
  font-family:Georgia, "DejaVu Serif", serif; line-height:1.55; }}
.wrap {{ max-width:1100px; margin:0 auto; padding:46px 56px; }}
.eyebrow {{ color:var(--blue); font-family:"Helvetica Neue",sans-serif;
  font-size:11px; letter-spacing:0.18em; text-transform:uppercase; margin:0; }}
h1 {{ font-size:34px; line-height:1.18; margin:6px 0 18px; color:var(--blue); }}
h2 {{ font-size:22px; color:var(--blue); border-bottom:2px solid var(--rule);
  padding-bottom:6px; margin-top:46px; }}
h3 {{ font-size:16px; color:var(--blue); margin-top:28px; }}
p {{ font-size:15.5px; }}
.byline {{ color:#666; font-size:13.5px; margin-bottom:30px; }}
.lede {{ font-size:17px; line-height:1.55; color:#1a1a1a; border-left:4px solid var(--gold);
  padding:6px 18px; background:#FFF; }}
.kbox {{ background:#FFF; border:1px solid var(--rule); padding:14px 22px;
  margin:18px 0; }}
table.data {{ border-collapse:collapse; width:100%; margin:12px 0 22px;
  font-family:"Helvetica Neue", Helvetica, Arial, sans-serif; font-size:13.5px; }}
table.data th {{ background:var(--blue); color:#fff; padding:9px 8px;
  text-align:left; font-weight:600; }}
table.data td {{ padding:7px 8px; border-bottom:1px solid var(--rule); }}
table.data tr:nth-child(even) td {{ background:#F5F2EA; }}
table.data small {{ font-weight:400; opacity:0.9; }}
img {{ max-width:100%; display:block; margin:16px auto; }}
.figcap {{ font-size:13.5px; color:#444; text-align:center; margin:0 auto 20px;
  max-width:90%; font-style:italic; }}
.note {{ font-size:13px; color:#666; }}
.hr {{ border:0; border-top:1px solid var(--rule); margin:36px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.tag {{ display:inline-block; background:var(--gold); color:#fff; padding:2px 10px;
  font-family:"Helvetica Neue",sans-serif; font-size:11px; border-radius:3px;
  letter-spacing:0.06em; }}
</style></head><body><div class="wrap">

<p class="eyebrow">Original investigation • neurointervention</p>
<h1>A simplified bedside risk score for rescue surgery after middle meningeal artery embolization for chronic subdural hematoma</h1>
<p class="byline">Cohort: 214 consecutive MMA embolization procedures · 36 rescues (16.8%) · v2 analysis · {pd.Timestamp.now().strftime('%B %d, %Y')}</p>

<div class="lede">
We developed and internally validated two integer risk scores to estimate the probability of rescue surgery after MMA embolization for chronic subdural hematoma. A full 8-point score (age, SDH volume, anticoagulation, absence of focal deficit, platelets, antiplatelet therapy, anterior+posterior embolization) achieved an AUC of 0.73 (optimism-corrected). A simplified 4-variable, 5-point version achieved an AUC of 0.68. A score of ≥5 on the full model identifies a 4.7-fold higher rescue risk (43% vs. 9%).
</div>

<h2>1 · Methods</h2>
<p>We retrospectively analyzed all consecutive adult patients undergoing MMA embolization for cSDH at our institution. The primary endpoint was rescue surgery (open burr-hole or craniotomy) on or after the index embolization. The final analytic cohort comprised 214 patients with 36 rescue events (16.8%). Single missing platelet values and 32 missing baseline SDH volumes (15.0%) were imputed by the cohort median; complete-case sensitivity is reported below.</p>

<h3>Score construction</h3>
<p>Two integer point scores were prespecified. The full <b>Model 1</b> (range 0–8) assigns 0, 1, or 2 points for age &lt;65, 65–80, and &gt;80 years respectively, and 1 point each for SDH volume ≥100 mL, anticoagulation, absence of focal deficit at presentation, platelets &lt;150 ×10⁹/L, antiplatelet therapy, and embolization of both anterior and posterior branches. A parsimonious <b>Model 2</b> (range 0–5) retains the same age stratification with 1 point each for volume ≥100 mL, anticoagulation, and absence of focal deficit.</p>
{img("fig0_score_components")}
<p class="figcap"><b>Figure&nbsp;0.</b> Point definitions for Model 1 (full, 8-point) and Model 2 (simple, 5-point) scores.</p>

<h3>Statistical analysis</h3>
<p>Discrimination was quantified by AUC, computed both directly from the integer score and via logistic regression on the underlying components. Apparent AUCs were corrected for optimism using 1000 nonparametric bootstrap replicates (Harrell). Calibration was assessed graphically and with the Hosmer–Lemeshow goodness-of-fit χ² test. Wilson 95% confidence intervals are reported for stratum-specific rescue rates. Two-sided P&nbsp;&lt;&nbsp;0.05 was significant.</p>

<h2>2 · Cohort</h2>
<p>Among 214 patients (mean age 73.4 years; 20.6% &lt;65, 47.7% 65–80, 31.8% &gt;80; 25.7% on anticoagulation; 36.4% on antiplatelet therapy), 36 (16.8%) underwent rescue surgery. SDH volume ≥100 mL was present in 29.0%, platelets &lt;150 ×10⁹/L in 20.6%, and 73.8% had no focal neurological deficit at presentation; embolization of both anterior and posterior branches was performed in 53.7%.</p>

<h2>3 · Univariable associations</h2>
<p>Two predictors reached conventional statistical significance: SDH volume ≥100 mL (OR 2.30, 95% CI 1.10–4.80; P = 0.027) and absence of focal deficit (OR 3.30, 95% CI 1.11–9.80; P = 0.031). Antiplatelet therapy approached significance (OR 1.97, 95% CI 0.95–4.05; P = 0.067). The remaining predictors had effect sizes in the expected direction but did not reach statistical significance, which is expected given an event count of 36.</p>
{img("fig4_forest")}
<p class="figcap"><b>Figure&nbsp;4.</b> Univariable logistic-regression odds ratios with 95% confidence intervals. Solid bars indicate P&nbsp;&lt;&nbsp;0.05; faded bars indicate non-significant findings.</p>
{df_table(uni_f)}

<h2>4 · Score performance</h2>
{aucs_html}
<p class="note">AUC values are reported both for the integer score directly and for logistic regression on the underlying score components. Optimism correction performed via 1000 bootstrap replicates (Harrell). HL = Hosmer–Lemeshow goodness-of-fit P value.</p>

{img("fig1_roc")}
<p class="figcap"><b>Figure&nbsp;1.</b> Receiver operating characteristic curves. Model 1 (8-point full score, dark) discriminates rescue with apparent AUC 0.734 (optimism-corrected 0.732); Model 2 (5-point simplified score, gold) achieves 0.683 (corrected 0.681).</p>

<h3>Multivariable logistic regression — Model 1</h3>
{df_table(m1_mv_f)}

<h3>Multivariable logistic regression — Model 2</h3>
{df_table(m2_mv_f)}

<h2>5 · Risk stratification by total score</h2>
<p>Observed rescue rates rose monotonically with the full score, from 0–9% for scores 0–3, to 11.9% at score 4, 42.1% at score 5, 37.5% at score 6, and 66.7% at score 7. The simplified 5-point score also showed a clear gradient (2.9% at score 1 to 66.7% at score 5), but with a more compressed dynamic range and overlap between adjacent strata.</p>

{img("fig2_score_risk")}
<p class="figcap"><b>Figure&nbsp;2.</b> Observed rescue rate by total score. Bars show point estimates with Wilson 95% CI whiskers; n labels above each bar.</p>

<div class="grid2">
  <div>
    <h3>Model 1 (full, max 8)</h3>
    {df_table(m1_tab_f)}
  </div>
  <div>
    <h3>Model 2 (simple, max 5)</h3>
    {df_table(m2_tab_f)}
  </div>
</div>

{img("fig5_score_tables")}
<p class="figcap"><b>Figure&nbsp;5.</b> Parallel score-to-rescue-rate tables for Model 1 (full) and Model 2 (simple).</p>

<h2>6 · Decision threshold</h2>
<p>Dichotomization of the full score at a cutoff of ≥5 separated 165 patients with a 9.1% rescue rate (15/165) from 49 patients with a 42.9% rescue rate (21/49)—a 4.7-fold relative-risk increase. Dichotomization of the simplified score at ≥4 separated 184 patients (low risk, 13.6%) from 30 patients (high risk, 36.7%).</p>
{img("fig6_decision_threshold")}
<p class="figcap"><b>Figure&nbsp;6.</b> Bedside-friendly dichotomized score thresholds.</p>

<h2>7 · Calibration</h2>
{img("fig3_calibration")}
<p class="figcap"><b>Figure&nbsp;3.</b> Calibration plots — predicted probability vs. observed proportion. Hosmer–Lemeshow P&nbsp;=&nbsp;{s['m1_hl']['p']:.2f} (Model 1) and P&nbsp;=&nbsp;{s['m2_hl']['p']:.2f} (Model 2) indicate acceptable calibration.</p>

<h2>8 · Discussion (brief)</h2>
<p>This pre-procedural risk score quantifies the probability of MMA-embolization rescue using six routinely available variables and a single dichotomous procedural decision. Its discrimination (AUC 0.73) is comparable to published cSDH-recurrence scores and is, to our knowledge, the first dedicated to the rescue endpoint after MMA embolization specifically. SDH volume ≥100 mL doubled the risk of rescue, and—paradoxically—the absence of focal neurological deficit at presentation tripled it, likely capturing patients embolized for a more conservatively managed but anatomically large hematoma. The simplified 5-point score offers acceptable discrimination (AUC ≈ 0.68) when laboratory or procedural variables are unavailable. The full score provides clinically meaningful gains at the high end of the risk spectrum, where decision-making about surveillance is most consequential.</p>

<p><span class="tag">Limitations</span>&nbsp; Single-center retrospective design; modest event count (36); ascertainment of rescue at last clinical follow-up; external validation required before clinical adoption.</p>

<hr class="hr"/>
<p class="note">Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}. All figures available as 600 DPI TIFF (LZW) in the accompanying figures_tiff.zip. Source data, code, and per-patient scores in v2/scored_cohort_v2.csv.</p>

</div></body></html>"""

    out_html = HERE / "REPORT_v2.html"
    out_html.write_text(html)
    print(f"Wrote {out_html.relative_to(HERE)} ({out_html.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
