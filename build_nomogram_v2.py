"""Nomogram for Model 1 — converts each variable to "points" on a 0–100 axis,
then maps total points to predicted probability of MMA-rescue surgery.
Standard Harrell-style nomogram, JAMA-styled."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"

C = dict(blue="#374E55", gold="#DF8F44", red="#B24745", green="#79AF97",
         grey="#80796B", lt="#EFE9DB")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.linewidth": 0.0,
    "savefig.dpi": 600,
})


def main():
    coefs = pd.read_csv(V2 / "m1_logit_coefs.csv")
    coefs = coefs[coefs["variable"] != "const"]
    intercept = float(pd.read_csv(V2 / "m1_logit_coefs.csv")
                      .query('variable == "const"')["coef"].values[0])

    # Predictor levels (primary M1 — no focal_deficit)
    items = [
        ("Age category",       "age_pts",         [0, 1, 2],  ["<65", "65–80", ">80"]),
        ("SDH volume ≥100 mL", "sdh_vol_ge100",   [0, 1],     ["No", "Yes"]),
        ("Anticoagulation",    "anticoag",        [0, 1],     ["No", "Yes"]),
        ("Platelets <150",     "plt_lt150",       [0, 1],     ["No", "Yes"]),
        ("Antiplatelet therapy", "antiplatelet",  [0, 1],     ["No", "Yes"]),
        ("Anterior + posterior", "ant_post",      [0, 1],     ["No", "Yes"]),
    ]

    # Compute points per level using the largest single-variable coefficient
    # max contribution → 100 points
    contrib = {}
    for label, key, levels, _ in items:
        b = float(coefs.loc[coefs.variable == key, "coef"].values[0])
        contrib[key] = [b * lv for lv in levels]
    all_max = max(max(v) - min(v) for v in contrib.values())
    factor = 100.0 / all_max

    # Compute total max points
    max_total_pts = sum(max(contrib[k]) - min(contrib[k])
                         for _, k, _, _ in items) * factor

    # Visual scaling so the total-points/probability axes fit the same width
    # (standard Harrell-nomogram trick: bottom axes use a tighter tick density)
    SCALE_TOTAL = 100.0 / max_total_pts   # 1 visual unit ↔ this many total-pts

    # ---- Spacing constants (more vertical breathing room) ----
    ROW = 1.55             # vertical distance between predictor rows
    GAP_TOP = 2.6          # gap between Points axis and first predictor
    GAP_BOT = 2.6          # gap between last predictor and Total-points axis
    GAP_AXES = 1.8         # gap between Total-points and Pr(rescue) axes
    H_top_axis = 0.6       # space above Points axis for ticks/labels
    H_footer = 1.1         # space below Pr(rescue) for footer

    n = len(items)
    total_h = (H_top_axis + GAP_TOP + (n - 1) * ROW + GAP_BOT
               + GAP_AXES + H_footer + 1.5)

    fig, ax = plt.subplots(figsize=(12, total_h * 0.58))
    ax.set_xlim(-26, 108)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ---- Title ----
    title_y = total_h - 0.55
    ax.text(50, title_y,
            "Model 1 nomogram — predicted probability of rescue surgery",
            ha="center", fontsize=13, fontweight="bold", color=C["blue"])
    ax.text(50, title_y - 0.50,
            "Read points for each predictor → sum → read probability on the bottom axis",
            ha="center", fontsize=10, color="#555", style="italic")

    # ---- Points axis (top, label 0–100, visual 0–100) ----
    y_top = title_y - 1.55
    ax.plot([0, 100], [y_top, y_top], color=C["blue"], lw=2)
    for x in range(0, 101, 10):
        ax.plot([x, x], [y_top - 0.18, y_top + 0.18], color=C["blue"], lw=1.5)
        ax.text(x, y_top + 0.45, str(x), ha="center", va="bottom",
                fontsize=9.5, color="#222")
    ax.text(-2, y_top, "Points", ha="right", va="center",
            fontsize=11, fontweight="bold", color=C["blue"])

    # ---- Per-predictor lines (with extra row spacing) ----
    y0 = y_top - GAP_TOP
    for i, (label, key, levels, level_labels) in enumerate(items):
        y = y0 - i * ROW
        b = float(coefs.loc[coefs.variable == key, "coef"].values[0])
        xs = [b * lv * factor for lv in levels]
        xs = [x - min(xs) for x in xs]
        ax.plot([min(xs), max(xs)], [y, y], color=C["blue"], lw=1.5)
        for x_, lvl_lbl in zip(xs, level_labels):
            ax.plot([x_, x_], [y - 0.18, y + 0.18], color=C["blue"], lw=1.4)
            ax.text(x_, y - 0.55, lvl_lbl, ha="center", va="top",
                    fontsize=9.5, color="#222")
        ax.text(-2, y, label, ha="right", va="center",
                fontsize=10.5, color="#222")

    # ---- Total points axis ----
    last_pred_y = y0 - (n - 1) * ROW
    y_tot = last_pred_y - GAP_BOT
    ax.plot([0, 100], [y_tot, y_tot], color=C["red"], lw=2.2)
    step = 50
    for pts in range(0, int(max_total_pts) + 1, step):
        x_vis = pts * SCALE_TOTAL
        ax.plot([x_vis, x_vis], [y_tot - 0.18, y_tot + 0.18],
                color=C["red"], lw=1.5)
        ax.text(x_vis, y_tot + 0.45, str(pts), ha="center", va="bottom",
                fontsize=9.5, color="#222")
    ax.text(-2, y_tot, "Total points", ha="right", va="center",
            fontsize=11, fontweight="bold", color=C["red"])

    # ---- Predicted probability axis ----
    y_p = y_tot - GAP_AXES
    ax.plot([0, 100], [y_p, y_p], color=C["red"], lw=2.2)
    p_ticks = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    for p in p_ticks:
        lp = np.log(p / (1 - p))
        pts = (lp - intercept) * factor
        if 0 <= pts <= max_total_pts:
            x_vis = pts * SCALE_TOTAL
            ax.plot([x_vis, x_vis], [y_p - 0.18, y_p + 0.18],
                    color=C["red"], lw=1.5)
            ax.text(x_vis, y_p - 0.50, f"{p:.2f}", ha="center", va="top",
                    fontsize=9.5, color="#222")
    ax.text(-2, y_p, "Pr(rescue)", ha="right", va="center",
            fontsize=11, fontweight="bold", color=C["red"])

    # Footer
    ax.text(50, y_p - 1.20,
            "Probabilities derived from the multivariable logistic regression of Model 1 (statsmodels).  "
            "Internal validation only.",
            ha="center", fontsize=9, color="#555", style="italic")

    fig.savefig(V2 / "fig9_nomogram.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(TIFF / "fig9_nomogram.tif", dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("Wrote v2/fig9_nomogram.png")


if __name__ == "__main__":
    main()
