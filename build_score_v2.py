"""
MMAE rescue prediction — Score v2 (Model 1 full / Model 2 simplified).

Model 1 (max 8):
  Age:               <65 → 0,  65–80 → 1,  >80 → 2
  SDH volume ≥ 100:  +1
  Anticoagulation:   +1
  Focal deficit:     +1
  Platelets < 150:   +1
  Antiplatelet:      +1
  Anterior+posterior embolization: +1

Model 2 (max 5, simplified):
  Age (0/1/2)
  SDH volume ≥ 100
  Anticoagulation
  Focal deficit
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
RNG = 42
HERE = Path(__file__).parent


# ----------------------------------------------------------------------
def load_and_score(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame(index=df.index)

    # Outcome — failure / reintervention
    out["y"] = (df["rescue_surgery"].astype(str).str.strip() == "Yes").astype(int)

    # Age — 0 / 1 / 2
    age = df["age"].astype(float)
    age_pts = np.where(age < 65, 0, np.where(age <= 80, 1, 2))
    out["age_pts"] = age_pts
    out["age_lt65"] = (age < 65).astype(int)
    out["age_65_80"] = ((age >= 65) & (age <= 80)).astype(int)
    out["age_gt80"] = (age > 80).astype(int)

    # SDH volume ≥ 100 mL  (impute median for missing then threshold)
    vol = df["sdhvolumebaseline"].astype(float)
    vol_imputed = vol.fillna(vol.median())
    out["sdh_vol_ge100"] = (vol_imputed >= 100).astype(int)
    out["_sdh_volume_missing"] = vol.isna().astype(int)
    out["sdh_volume_raw"] = vol

    # Anticoagulation Yes (impute "No" for missing, conservative)
    out["anticoag"] = (df["anticoagulation"].astype(str).str.strip() == "Yes").astype(int)

    # Focal deficit at presentation (corrected coding — prevalence 59.8%)
    out["focal_deficit"] = (df["focal_deficit"].astype(str).str.strip() == "Yes").astype(int)
    out["no_focal_deficit"] = (df["focal_deficit"].astype(str).str.strip() == "No").astype(int)

    # Platelets < 150
    plt = df["plt_num"].astype(float)
    plt_imputed = plt.fillna(plt.median())
    out["plt_lt150"] = (plt_imputed < 150).astype(int)

    # Antiplatelet Yes
    out["antiplatelet"] = (df["antiplatelet"].astype(str).str.strip() == "Yes").astype(int)

    # Anterior + posterior embolization
    out["ant_post"] = (df["branches"].astype(str).str.strip() == "Anterior + posterior").astype(int)

    # focal_deficit re-included after data correction (prevalence 59.8%
    # now matches published MMA-embolization series; previous 26.2% was
    # a coding error).
    m1_cols = ["age_pts", "sdh_vol_ge100", "anticoag", "focal_deficit",
               "plt_lt150", "antiplatelet", "ant_post"]
    m2_cols = ["age_pts", "sdh_vol_ge100", "anticoag", "focal_deficit"]
    out["score_m1"] = out[m1_cols].sum(axis=1).astype(int)
    out["score_m2"] = out[m2_cols].sum(axis=1).astype(int)

    # Carry raw fields used downstream
    out["age"] = age
    return out


# ----------------------------------------------------------------------
def auc_and_optimism(score: np.ndarray, y: np.ndarray, n_boot: int = 1000) -> dict:
    """Apparent + Harrell optimism-corrected AUC for an integer/continuous score."""
    apparent = roc_auc_score(y, score)
    rng = np.random.default_rng(RNG)
    n = len(y)
    optimisms = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_apparent = roc_auc_score(y[idx], score[idx])
        boot_on_orig = roc_auc_score(y, score)  # score already fixed
        optimisms.append(boot_apparent - boot_on_orig)
    optimism = float(np.mean(optimisms))
    return dict(apparent=float(apparent),
                optimism=optimism,
                corrected=float(apparent - optimism))


def logistic_fit_eval(X: pd.DataFrame, y: np.ndarray, n_boot: int = 1000) -> dict:
    """Fit logistic regression, return apparent + optimism-corrected AUC,
    coefficients, and predicted probabilities."""
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.Logit(y, Xc).fit(disp=False, method="bfgs", maxiter=500)
    pred = res.predict(Xc)
    apparent = roc_auc_score(y, pred)

    rng = np.random.default_rng(RNG)
    n = len(y)
    optimisms = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        Xb = Xc.iloc[idx].values
        yb = y[idx]
        try:
            r_b = sm.Logit(yb, Xb).fit(disp=False, method="bfgs", maxiter=200)
            pb_boot = r_b.predict(Xb)
            pb_orig = r_b.predict(Xc.values)
            opt = roc_auc_score(yb, pb_boot) - roc_auc_score(y, pb_orig)
            optimisms.append(opt)
        except Exception:
            continue
    optimism = float(np.mean(optimisms))

    coefs = pd.DataFrame({
        "variable": Xc.columns,
        "coef": res.params.values,
        "se": res.bse.values,
        "OR": np.exp(res.params.values),
        "OR_lo": np.exp(res.conf_int().iloc[:, 0].values),
        "OR_hi": np.exp(res.conf_int().iloc[:, 1].values),
        "p": res.pvalues.values,
    })
    return dict(
        apparent=float(apparent),
        optimism=optimism,
        corrected=float(apparent - optimism),
        coefs=coefs,
        pred=np.asarray(pred),
        loglik=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        nobs=int(res.nobs),
        brier=float(brier_score_loss(y, pred)),
    )


# ----------------------------------------------------------------------
def score_risk_table(score: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in sorted(np.unique(score)):
        mask = score == s
        n = int(mask.sum())
        ev = int(y[mask].sum())
        # Wilson 95% CI for proportion
        if n > 0:
            p = ev / n
            z = 1.96
            denom = 1 + z**2 / n
            centre = p + z**2 / (2 * n)
            spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
            lo = (centre - spread) / denom
            hi = (centre + spread) / denom
        else:
            p = lo = hi = np.nan
        rows.append(dict(score=int(s), n=n, failures=ev,
                         rate=p, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def hl_test(prob: np.ndarray, y: np.ndarray, q: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit."""
    df = pd.DataFrame({"p": prob, "y": y})
    df["q"] = pd.qcut(df["p"], q=min(q, df["p"].nunique()), duplicates="drop")
    grp = df.groupby("q", observed=True).agg(n=("y", "size"),
                                              o=("y", "sum"),
                                              e=("p", "sum"))
    grp["nm_o"] = grp["n"] - grp["o"]
    grp["nm_e"] = grp["n"] - grp["e"]
    chi = ((grp["o"] - grp["e"]) ** 2 / grp["e"] +
           (grp["nm_o"] - grp["nm_e"]) ** 2 / grp["nm_e"]).sum()
    from scipy.stats import chi2
    dof = max(1, len(grp) - 2)
    p = 1 - chi2.cdf(chi, dof)
    return dict(chi2=float(chi), dof=int(dof), p=float(p))


# ----------------------------------------------------------------------
def main():
    csv = HERE / "mmaecsv.csv"
    sc = load_and_score(csv)
    n = len(sc)
    events = int(sc["y"].sum())
    print(f"n={n}, events={events} ({events/n:.1%})")

    out = HERE / "v2"
    out.mkdir(exist_ok=True)
    sc.to_csv(out / "scored_cohort_v2.csv", index=False)

    # ------------------------------------------------------------------
    results = {"n": n, "events": events}

    # Score-only AUCs (score itself as discriminator)
    results["m1_score_auc"] = auc_and_optimism(sc["score_m1"].values, sc["y"].values)
    results["m2_score_auc"] = auc_and_optimism(sc["score_m2"].values, sc["y"].values)
    print(f"Model 1 score AUC: apparent={results['m1_score_auc']['apparent']:.3f} corrected={results['m1_score_auc']['corrected']:.3f}")
    print(f"Model 2 score AUC: apparent={results['m2_score_auc']['apparent']:.3f} corrected={results['m2_score_auc']['corrected']:.3f}")

    # Multivariable logistic — use the per-variable encoding
    m1_X = sc[["age_pts", "sdh_vol_ge100", "anticoag", "focal_deficit",
               "plt_lt150", "antiplatelet", "ant_post"]]
    m2_X = sc[["age_pts", "sdh_vol_ge100", "anticoag", "focal_deficit"]]

    m1_logit = logistic_fit_eval(m1_X, sc["y"].values)
    m2_logit = logistic_fit_eval(m2_X, sc["y"].values)

    results["m1_logit"] = {k: v for k, v in m1_logit.items() if k not in ("coefs", "pred")}
    results["m2_logit"] = {k: v for k, v in m2_logit.items() if k not in ("coefs", "pred")}
    print(f"Model 1 logit AUC: apparent={m1_logit['apparent']:.3f} corrected={m1_logit['corrected']:.3f}")
    print(f"Model 2 logit AUC: apparent={m2_logit['apparent']:.3f} corrected={m2_logit['corrected']:.3f}")

    m1_logit["coefs"].to_csv(out / "m1_logit_coefs.csv", index=False)
    m2_logit["coefs"].to_csv(out / "m2_logit_coefs.csv", index=False)

    # Score → failure tables
    m1_tab = score_risk_table(sc["score_m1"].values, sc["y"].values)
    m2_tab = score_risk_table(sc["score_m2"].values, sc["y"].values)
    m1_tab.to_csv(out / "m1_risk_by_score.csv", index=False)
    m2_tab.to_csv(out / "m2_risk_by_score.csv", index=False)

    # Calibration (logit predictions)
    results["m1_hl"] = hl_test(m1_logit["pred"], sc["y"].values, q=8)
    results["m2_hl"] = hl_test(m2_logit["pred"], sc["y"].values, q=6)

    # Save predictions
    sc["pred_m1"] = m1_logit["pred"]
    sc["pred_m2"] = m2_logit["pred"]
    sc.to_csv(out / "scored_cohort_v2.csv", index=False)

    # Univariate ORs (for forest plot)
    uni_rows = []
    feat_labels = {
        "age_pts": "Age (per category, <65/65–80/>80)",
        "age_lt65": "Age <65",
        "age_65_80": "Age 65–80",
        "age_gt80": "Age >80",
        "sdh_vol_ge100": "SDH volume ≥100 mL",
        "anticoag": "Anticoagulation",
        "focal_deficit": "Focal deficit at presentation",
        "plt_lt150": "Platelets <150 ×10⁹/L",
        "antiplatelet": "Antiplatelet therapy",
        "ant_post": "Anterior + posterior embolization",
    }
    for c, lbl in feat_labels.items():
        x = sc[c].values.astype(float)
        if np.unique(x).size < 2:
            continue
        try:
            X = sm.add_constant(x.reshape(-1, 1))
            r = sm.Logit(sc["y"].values, X).fit(disp=False, method="bfgs", maxiter=200)
            ci = np.asarray(r.conf_int())  # shape (k, 2): col0=lower, col1=upper
            uni_rows.append(dict(
                variable=lbl, OR=float(np.exp(r.params[1])),
                OR_lo=float(np.exp(ci[1, 0])), OR_hi=float(np.exp(ci[1, 1])),
                p=float(r.pvalues[1])))
        except Exception:
            pass
    uni_df = pd.DataFrame(uni_rows)
    uni_df.to_csv(out / "univariate_ors.csv", index=False)

    # Save full results JSON
    with open(out / "summary_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\nWrote v2/ artifacts.")
    print("\n--- Model 1 score → risk ---")
    print(m1_tab.to_string(index=False))
    print("\n--- Model 2 score → risk ---")
    print(m2_tab.to_string(index=False))


if __name__ == "__main__":
    main()
