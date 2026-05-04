"""Fit the SOCR score (co-author's parallel analysis) on our 214-patient
cohort so all three scores are evaluated on identical data.

SOCR score (max 6):
  Age <65 / 65–85 / >85   →   0 / 1 / 2 pts
  SDH volume ≥100 mL       →   +1
  Platelets <150 ×10⁹/L    →   +1
  Antiplatelet therapy     →   +1
  Absence of focal deficit →   +1
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = Path(__file__).parent
V2 = HERE / "v2"
RNG = 42


def main():
    df = pd.read_csv(HERE / "mmaecsv.csv")
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    n = len(sc)

    # SOCR uses age cutoff at >85 (not >80)
    age = df["age"].astype(float).values
    age_pts_socr = np.where(age < 65, 0, np.where(age <= 85, 1, 2))

    sc["age_pts_socr"] = age_pts_socr
    sc["score_m3"] = (
        age_pts_socr
        + sc["sdh_vol_ge100"].values
        + sc["plt_lt150"].values
        + sc["antiplatelet"].values
    ).astype(int)

    feats = ["age_pts_socr", "sdh_vol_ge100", "plt_lt150",
             "antiplatelet"]
    X = sc[feats]
    y = sc["y"].values

    # Multivariable logistic
    Xc = sm.add_constant(X)
    res = sm.Logit(y, Xc).fit(disp=False, method="bfgs", maxiter=500)
    ci = np.asarray(res.conf_int())
    coefs = pd.DataFrame({
        "variable": Xc.columns,
        "coef": res.params.values,
        "se": res.bse.values,
        "OR": np.exp(res.params.values),
        "OR_lo": np.exp(ci[:, 0]),
        "OR_hi": np.exp(ci[:, 1]),
        "p": res.pvalues.values,
    })
    coefs.to_csv(V2 / "m3_logit_coefs.csv", index=False)

    pred = res.predict(Xc).values
    sc["pred_m3"] = pred
    sc.to_csv(V2 / "scored_cohort_v2.csv", index=False)

    # AUCs
    auc_score = roc_auc_score(y, sc["score_m3"].values)
    auc_logit = roc_auc_score(y, pred)
    brier = brier_score_loss(y, pred)

    # Bootstrap optimism (Harrell)
    rng = np.random.default_rng(RNG)
    n_boot = 1000
    opts_score, opts_logit = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        # score AUC: simple
        try:
            opts_score.append(roc_auc_score(y[idx], sc["score_m3"].values[idx]) -
                              roc_auc_score(y, sc["score_m3"].values))
        except Exception:
            pass
        # logit AUC
        try:
            r_b = sm.Logit(y[idx], Xc.iloc[idx].values).fit(disp=False,
                                                            method="bfgs",
                                                            maxiter=200)
            pb_boot = r_b.predict(Xc.iloc[idx].values)
            pb_orig = r_b.predict(Xc.values)
            opts_logit.append(roc_auc_score(y[idx], pb_boot) -
                              roc_auc_score(y, pb_orig))
        except Exception:
            pass
    auc_score_corrected = auc_score - float(np.mean(opts_score))
    auc_logit_corrected = auc_logit - float(np.mean(opts_logit))

    # Score → risk table
    rows = []
    for s in sorted(np.unique(sc["score_m3"])):
        mask = sc["score_m3"] == s
        nn = int(mask.sum())
        ev = int(y[mask].sum())
        if nn:
            p = ev / nn
            z = 1.96
            denom = 1 + z**2 / nn
            centre = p + z**2 / (2 * nn)
            spread = z * np.sqrt(p * (1 - p) / nn + z**2 / (4 * nn**2))
            lo = (centre - spread) / denom
            hi = (centre + spread) / denom
        else:
            p = lo = hi = np.nan
        rows.append(dict(score=int(s), n=nn, failures=ev,
                         rate=p, ci_lo=lo, ci_hi=hi))
    risk = pd.DataFrame(rows)
    risk.to_csv(V2 / "m3_risk_by_score.csv", index=False)

    # Update summary JSON
    summary = json.loads((V2 / "summary_v2.json").read_text())
    summary["m3_score_auc"] = dict(apparent=float(auc_score),
                                   corrected=float(auc_score_corrected))
    summary["m3_logit"] = dict(apparent=float(auc_logit),
                                corrected=float(auc_logit_corrected),
                                brier=float(brier),
                                aic=float(res.aic),
                                bic=float(res.bic),
                                nobs=int(res.nobs))
    (V2 / "summary_v2.json").write_text(json.dumps(summary, indent=2,
                                                    default=float))

    print(f"SOCR (Model 3) on our cohort (n={n}, events={int(y.sum())}):")
    print(f"  Score AUC: apparent={auc_score:.3f}, corrected={auc_score_corrected:.3f}")
    print(f"  Logit AUC: apparent={auc_logit:.3f}, corrected={auc_logit_corrected:.3f}")
    print(f"  Brier   : {brier:.3f}")
    print()
    print("Score → rescue rate:")
    print(risk.to_string(index=False))
    print()
    print("Multivariable ORs:")
    print(coefs.to_string(index=False))


if __name__ == "__main__":
    main()
