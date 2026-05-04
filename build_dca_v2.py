"""Decision curve analysis (Vickers 2006) + operating-point metrics
(sens / spec / PPV / NPV with Wilson CIs) at the recommended cutoffs."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"

C = dict(blue="#374E55", gold="#DF8F44", teal="#00A1D5", red="#B24745",
         green="#79AF97", grey="#80796B", lt="#EFE9DB")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600,
})


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0, (centre - spread) / denom), min(1, (centre + spread) / denom)


def net_benefit(y, prob, threshold):
    """Vickers net-benefit at given threshold probability."""
    n = len(y)
    pred = prob >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    if threshold >= 1:
        return 0
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def operating_point(y, score, cutoff):
    """At a given cutoff (score >= cutoff predicts positive),
    return sens/spec/PPV/NPV with Wilson CIs."""
    pred = score >= cutoff
    TP = int(((y == 1) & pred).sum())
    FN = int(((y == 1) & ~pred).sum())
    TN = int(((y == 0) & ~pred).sum())
    FP = int(((y == 0) & pred).sum())
    sens, sens_lo, sens_hi = wilson_ci(TP, TP + FN)
    spec, spec_lo, spec_hi = wilson_ci(TN, TN + FP)
    ppv,  ppv_lo,  ppv_hi  = wilson_ci(TP, TP + FP)
    npv,  npv_lo,  npv_hi  = wilson_ci(TN, TN + FN)
    return dict(cutoff=int(cutoff), TP=TP, FN=FN, TN=TN, FP=FP,
                sens=sens, sens_ci=(sens_lo, sens_hi),
                spec=spec, spec_ci=(spec_lo, spec_hi),
                ppv=ppv,   ppv_ci=(ppv_lo, ppv_hi),
                npv=npv,   npv_ci=(npv_lo, npv_hi))


def main():
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    y = sc["y"].values.astype(int)

    # Decision curve analysis
    thresholds = np.linspace(0.02, 0.55, 60)
    p1 = sc["pred_m1"].values
    p2 = sc["pred_m2"].values
    nb_m1 = [net_benefit(y, p1, t) for t in thresholds]
    nb_m2 = [net_benefit(y, p2, t) for t in thresholds]
    nb_all = [net_benefit(y, np.ones_like(p1, dtype=float), t) for t in thresholds]
    nb_none = [0 for _ in thresholds]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(thresholds, nb_m1, color=C["blue"], lw=2.4, label="Model 1 (full)")
    ax.plot(thresholds, nb_m2, color=C["gold"], lw=2.4, label="Model 2 (simple)")
    ax.plot(thresholds, nb_all, color=C["red"], lw=1.4, ls="--",
            label="Treat all", alpha=0.85)
    ax.plot(thresholds, nb_none, color=C["grey"], lw=1.4, ls=":",
            label="Treat none", alpha=0.85)
    ax.fill_between(thresholds, np.minimum(nb_m1, nb_m2), nb_m1,
                    where=(np.array(nb_m1) > np.array(nb_m2)),
                    facecolor=C["blue"], alpha=0.10, interpolate=True)
    ax.set_xlabel("Threshold probability of rescue surgery")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.set_xlim(0.02, 0.55)
    ax.set_ylim(min(0, min(nb_m1) - 0.01), max(nb_all + nb_m1) * 1.1)
    ax.axhline(0, color="#888", lw=0.6)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right")
    fig.savefig(V2 / "fig10_dca.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(TIFF / "fig10_dca.tif", dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Operating-point metrics for all three scores (no focal deficit)
    rows = []
    for label, score, cutoffs in [
        ("Model 1 (full, 7 pts)",   sc["score_m1"].values, [3, 4, 5]),
        ("Model 3 (5 pts)",          sc["score_m3"].values, [2, 3, 4]),
        ("Model 2 (simple, 4 pts)",  sc["score_m2"].values, [2, 3]),
    ]:
        for cut in cutoffs:
            r = operating_point(y, score, cut)
            rows.append(dict(
                model=label, cutoff=f"≥{cut}",
                TP=r["TP"], FN=r["FN"], TN=r["TN"], FP=r["FP"],
                sensitivity=f"{r['sens']*100:.1f} ({r['sens_ci'][0]*100:.1f}–{r['sens_ci'][1]*100:.1f})",
                specificity=f"{r['spec']*100:.1f} ({r['spec_ci'][0]*100:.1f}–{r['spec_ci'][1]*100:.1f})",
                PPV=f"{r['ppv']*100:.1f} ({r['ppv_ci'][0]*100:.1f}–{r['ppv_ci'][1]*100:.1f})",
                NPV=f"{r['npv']*100:.1f} ({r['npv_ci'][0]*100:.1f}–{r['npv_ci'][1]*100:.1f})",
            ))
    op_df = pd.DataFrame(rows)
    op_df.to_csv(V2 / "operating_points.csv", index=False)
    print(op_df.to_string(index=False))
    print("\nWrote v2/operating_points.csv and fig10_dca.png")


if __name__ == "__main__":
    main()
