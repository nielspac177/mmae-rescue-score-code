"""Build Table 1 — baseline characteristics by rescue surgery status."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
V2 = HERE / "v2"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def cont_row(name: str, vals_g0, vals_g1, vals_all, prefer_median=False):
    """Continuous variable row. Returns dict with summary statistics."""
    g0 = pd.Series(vals_g0).dropna().astype(float)
    g1 = pd.Series(vals_g1).dropna().astype(float)
    all_v = pd.Series(vals_all).dropna().astype(float)

    # Choose representation
    if prefer_median or stats.shapiro(all_v.sample(min(len(all_v), 500),
                                                   random_state=0))[1] < 0.05:
        # median (IQR)
        def f(s):
            return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"
        try:
            p = stats.mannwhitneyu(g0, g1, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        repr_kind = "Median (IQR)"
    else:
        def f(s):
            return f"{s.mean():.1f} ± {s.std():.1f}"
        try:
            p = stats.ttest_ind(g0, g1, equal_var=False).pvalue
        except Exception:
            p = np.nan
        repr_kind = "Mean ± SD"

    return dict(Variable=f"{name}, {repr_kind.lower()}",
                Overall=f(all_v), NoRescue=f(g0), Rescue=f(g1), P=fmt_p(p))


def cat_row(name: str, vals_g0, vals_g1, vals_all, level=None):
    """Categorical variable row — single level (e.g., 'Yes')."""
    if level is None:
        # binary, default to "Yes"
        level = 1
    g0 = pd.Series(vals_g0)
    g1 = pd.Series(vals_g1)
    all_v = pd.Series(vals_all)

    n0 = (g0 == level).sum()
    n1 = (g1 == level).sum()
    n_all = (all_v == level).sum()

    def pct(num, den):
        return f"{num} ({100 * num / den:.1f}%)" if den else "—"

    # 2x2 chi-square / Fisher
    try:
        tab = np.array([[(g0 == level).sum(), (g0 != level).sum()],
                        [(g1 == level).sum(), (g1 != level).sum()]])
        if (tab < 5).any():
            p = stats.fisher_exact(tab).pvalue
        else:
            p = stats.chi2_contingency(tab, correction=False).pvalue
    except Exception:
        p = np.nan

    return dict(Variable=name,
                Overall=pct(n_all, len(all_v)),
                NoRescue=pct(n0, len(g0)),
                Rescue=pct(n1, len(g1)),
                P=fmt_p(p))


def main():
    df = pd.read_csv(HERE / "mmaecsv.csv")
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")

    df["y"] = sc["y"].values
    g0 = df[df["y"] == 0]
    g1 = df[df["y"] == 1]

    rows = []

    # Demographics
    rows.append(cont_row("Age, years", g0["age"], g1["age"], df["age"], prefer_median=False))
    rows.append(cat_row("Age <65 years", sc.loc[df.index, "age_lt65"][df["y"] == 0],
                        sc.loc[df.index, "age_lt65"][df["y"] == 1],
                        sc["age_lt65"]))
    rows.append(cat_row("Age 65–80 years", sc.loc[df.index, "age_65_80"][df["y"] == 0],
                        sc.loc[df.index, "age_65_80"][df["y"] == 1],
                        sc["age_65_80"]))
    rows.append(cat_row("Age >80 years", sc.loc[df.index, "age_gt80"][df["y"] == 0],
                        sc.loc[df.index, "age_gt80"][df["y"] == 1],
                        sc["age_gt80"]))
    # Sex
    if "gender_num" in df.columns:
        male0 = (g0["gender_num"] == "Male").astype(int)
        male1 = (g1["gender_num"] == "Male").astype(int)
        male_all = (df["gender_num"] == "Male").astype(int)
        rows.append(cat_row("Male sex", male0, male1, male_all))

    # Smoking — current
    if "smoking" in df.columns:
        smk0 = (g0["smoking"] == "Current smoker").astype(int)
        smk1 = (g1["smoking"] == "Current smoker").astype(int)
        smk_all = (df["smoking"] == "Current smoker").astype(int)
        rows.append(cat_row("Current smoker", smk0, smk1, smk_all))
        # Former
        smk0 = (g0["smoking"] == "Former smoker").astype(int)
        smk1 = (g1["smoking"] == "Former smoker").astype(int)
        smk_all = (df["smoking"] == "Former smoker").astype(int)
        rows.append(cat_row("Former smoker", smk0, smk1, smk_all))

    # Comorbidities
    yes_no = lambda s: (s.astype(str).str.strip() == "Yes").astype(int)
    for col, label in [
        ("hypertension", "Hypertension"),
        ("diabetes", "Diabetes mellitus"),
        ("liver", "Liver disease"),
        ("malignancy", "Malignancy"),
        ("anticoagulation", "Anticoagulation use"),
        ("antiplatelet", "Antiplatelet use"),
    ]:
        if col in df.columns:
            v0 = yes_no(g0[col]); v1 = yes_no(g1[col]); va = yes_no(df[col])
            rows.append(cat_row(label, v0, v1, va))

    # Lab values
    for col, label in [
        ("hb_num", "Hemoglobin, g/dL"),
        ("plt_num", "Platelets, ×10⁹/L"),
        ("inrlastbeforeprocedure", "INR pre-procedure"),
    ]:
        if col in df.columns:
            try:
                rows.append(cont_row(label, g0[col], g1[col], df[col],
                                     prefer_median=False))
            except Exception:
                pass

    # Imaging
    rows.append(cont_row("SDH volume baseline, mL",
                          g0["sdhvolumebaseline"], g1["sdhvolumebaseline"],
                          df["sdhvolumebaseline"], prefer_median=True))
    if "midlineshiftmeasureinmmbaseline" in df.columns:
        rows.append(cont_row("Midline shift baseline, mm",
                              g0["midlineshiftmeasureinmmbaseline"],
                              g1["midlineshiftmeasureinmmbaseline"],
                              df["midlineshiftmeasureinmmbaseline"],
                              prefer_median=True))

    # Clinical presentation
    rows.append(cat_row("Focal neurological deficit",
                        yes_no(g0["focal_deficit"]),
                        yes_no(g1["focal_deficit"]),
                        yes_no(df["focal_deficit"])))
    if "headache" in df.columns:
        rows.append(cat_row("Headache",
                            yes_no(g0["headache"]),
                            yes_no(g1["headache"]),
                            yes_no(df["headache"])))
    if "fall" in df.columns:
        rows.append(cat_row("History of fall",
                            yes_no(g0["fall"]),
                            yes_no(g1["fall"]),
                            yes_no(df["fall"])))
    if "gcsonpresentation" in df.columns:
        rows.append(cont_row("GCS on presentation",
                              g0["gcsonpresentation"], g1["gcsonpresentation"],
                              df["gcsonpresentation"], prefer_median=True))
    if "baselinemrs" in df.columns:
        rows.append(cont_row("Baseline mRS",
                              g0["baselinemrs"], g1["baselinemrs"],
                              df["baselinemrs"], prefer_median=True))

    # Procedure
    if "branches" in df.columns:
        v0 = (g0["branches"] == "Anterior + posterior").astype(int)
        v1 = (g1["branches"] == "Anterior + posterior").astype(int)
        va = (df["branches"] == "Anterior + posterior").astype(int)
        rows.append(cat_row("Anterior + posterior embolization", v0, v1, va))
    if "useofliquidembolic" in df.columns:
        liq0 = (g0["useofliquidembolic"] == 1).astype(int)
        liq1 = (g1["useofliquidembolic"] == 1).astype(int)
        liq_all = (df["useofliquidembolic"] == 1).astype(int)
        rows.append(cat_row("Liquid embolic used", liq0, liq1, liq_all))

    tab = pd.DataFrame(rows)
    tab = tab[["Variable", "Overall", "NoRescue", "Rescue", "P"]]
    tab.columns = ["Variable",
                   f"Overall (n={len(df)})",
                   f"No rescue (n={(df['y']==0).sum()})",
                   f"Rescue (n={(df['y']==1).sum()})",
                   "P"]
    tab.to_csv(V2 / "table1_baseline.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nWrote {(V2 / 'table1_baseline.csv').relative_to(HERE)}")


if __name__ == "__main__":
    main()
