"""ROC overlay for the three integer scores (Model 1 / Model 2 / Model 3 SOCR).
Replaces the older 2-model fig1_roc.png."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"

C = dict(blue="#374E55", gold="#DF8F44", red="#B24745",
         green="#79AF97", grey="#80796B", teal="#00A1D5", purple="#6A6599")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600,
})


def main():
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    y = sc["y"].values

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for name, col, color, ls in [
        ("Model 1 (primary, max 7)",  "score_m1", C["blue"],   "-"),
        ("Model 2 (primary, max 4)",  "score_m2", C["gold"],   "-"),
        ("Model 3 (sensitivity, max 5)", "score_m3", C["purple"], "--"),
    ]:
        s = sc[col].values.astype(float)
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, lw=2.4, color=color, linestyle=ls,
                label=f"{name} — AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1.2, alpha=0.7)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9.0, title="Discrimination",
              title_fontsize=9.5)
    ax.grid(alpha=0.18)
    fig.savefig(V2 / "fig1_roc.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(TIFF / "fig1_roc.tif", dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("Wrote v2/fig1_roc.png  (now overlays Models 1, 2, 3)")


if __name__ == "__main__":
    main()
