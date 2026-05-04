"""Figure 0 v3 — STROBE-style top-down study flow with three clean lanes."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"
TIFF.mkdir(exist_ok=True, parents=True)

C = dict(blue="#374E55", gold="#DF8F44", teal="#00A1D5", red="#B24745",
         green="#79AF97", grey="#80796B", lt="#EFE9DB", pale="#F5F2EA",
         purple="#6A6599")

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "savefig.dpi": 600,
    "figure.dpi": 110,
})

# ---- Lane geometry ----
# Figure 16 wide × 9 tall. Three lanes of width ~4.8 each, gap 0.5.
LANE_W = 4.8
LANE_C1 = 2.7   # center of lane 1
LANE_C2 = 8.0
LANE_C3 = 13.3


def box(ax, x, y, w, h, text, fill, edge="#222", text_color="white",
        font_size=10, bold=True, lw=1.4):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=font_size, color=text_color,
            fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, color="#666", lw=1.3):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=12,
                        lw=lw, color=color, zorder=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    # ---- Title ----
    ax.text(8.0, 9.30,
            "Pre-procedural risk score for rescue surgery after MMA embolization for cSDH",
            ha="center", va="top", fontsize=15, fontweight="bold", color=C["blue"])
    ax.text(8.0, 8.95,
            "Single-center retrospective cohort  ·  three integer scoring models  ·  internal validation",
            ha="center", va="top", fontsize=11, color="#555", style="italic")

    # ---- Lane headers ----
    ax.text(LANE_C1, 8.30, "STUDY POPULATION", ha="center", fontsize=10.5,
            fontweight="bold", color=C["blue"])
    ax.text(LANE_C2, 8.30, "PREDICTORS  &  SCORING MODELS", ha="center", fontsize=10.5,
            fontweight="bold", color=C["blue"])
    ax.text(LANE_C3, 8.30, "VALIDATION", ha="center", fontsize=10.5,
            fontweight="bold", color=C["blue"])
    # Lane separator lines
    for x in (5.35, 10.65):
        ax.plot([x, x], [0.6, 8.6], color=C["lt"], lw=0.8, zorder=0)
    ax.plot([0.4, 15.6], [8.10, 8.10], color=C["lt"], lw=1.0, zorder=0)

    # ============== LANE 1 — Population pipeline ==============
    box(ax, LANE_C1, 7.55, 4.55, 0.70,
        "Adults undergoing MMA embolization\nfor chronic SDH at our institution",
        C["blue"], font_size=11)
    arrow(ax, LANE_C1, 7.18, LANE_C1, 6.85)

    # Exclusion side-note placed inside Lane 1, between cohort and analytic boxes
    ax.text(LANE_C1, 6.97,
            "Excluded:  acute SDH  ·  no follow-up  ·  no baseline imaging",
            fontsize=8.6, color="#666", ha="center", va="center",
            style="italic")

    box(ax, LANE_C1, 6.50, 4.55, 0.55,
        "n = 214 analytic cohort",
        C["pale"], edge=C["blue"], text_color="#222", font_size=11.5)
    arrow(ax, LANE_C1, 6.20, LANE_C1, 5.85)

    box(ax, LANE_C1, 5.50, 4.55, 0.60,
        "Outcome: rescue surgery\n36 events  /  16.8%",
        C["red"], font_size=11.5)
    arrow(ax, LANE_C1, 5.18, LANE_C1, 4.85)

    box(ax, LANE_C1, 4.50, 4.55, 0.60,
        "Endpoint ascertainment\noperative log + last clinical follow-up",
        C["grey"], font_size=10.5)

    # Predictor list panel (lane 1, lower)
    ax.text(LANE_C1, 3.85, "PRESPECIFIED PREDICTORS",
            ha="center", fontsize=10, fontweight="bold", color=C["blue"])
    pred_box = FancyBboxPatch((0.45, 0.85), 4.6, 2.85,
                              boxstyle="round,pad=0.02,rounding_size=0.04",
                              linewidth=1.0, edgecolor="#888",
                              facecolor=C["pale"], zorder=2)
    ax.add_patch(pred_box)
    preds = [
        "Age  (<65 / 65–80 / >80)",
        "SDH volume ≥ 100 mL",
        "Anticoagulation",
        "Platelets < 150 ×10⁹/L",
        "Antiplatelet therapy",
        "Anterior + posterior embolization",
    ]
    for i, line in enumerate(preds):
        ax.text(0.70, 3.50 - i * 0.38, "▸  " + line, ha="left", va="center",
                fontsize=10.2, color="#222")

    # ============== LANE 2 — Predictors & Scoring ==============
    # Top "two scoring models" header
    ax.text(LANE_C2, 7.65, "Total points → predicted probability of rescue",
            ha="center", fontsize=10, color="#444", style="italic")

    box(ax, LANE_C2, 7.20, 4.85, 0.65,
        "Model 1 — Full   (max 7 pts)\nall 6 predictors above",
        C["blue"], font_size=10.5)
    box(ax, LANE_C2, 6.40, 4.85, 0.65,
        "Model 3   (max 5 pts)\nage (>85), vol., plt, antiplatelet",
        C["purple"], font_size=10.5)
    box(ax, LANE_C2, 5.60, 4.85, 0.65,
        "Model 2 — Simple   (max 4 pts)\nage, volume, anticoagulation",
        C["gold"], font_size=10.5)
    arrow(ax, LANE_C2, 6.85, LANE_C2, 6.75, color="#888")
    arrow(ax, LANE_C2, 6.05, LANE_C2, 5.95, color="#888")

    # Score → risk mini-chart
    ax.text(LANE_C2, 4.85, "Observed rescue rate by Model 1 total score",
            ha="center", fontsize=10.5, fontweight="bold", color=C["blue"])
    # Updated for Model 1 without focal deficit (6 vars, max 7)
    # rates from v2/m1_risk_by_score.csv
    bar_centers = [LANE_C2 - 1.95 + i * 0.65 for i in range(7)]
    bar_v = [0.000, 0.138, 0.060, 0.094, 0.383, 0.231, 0.667]
    bar_lbl = ["0", "1", "2", "3", "4", "5", "≥6"]
    base_y = 3.50
    chart_h = 1.05
    for x, v, lab in zip(bar_centers, bar_v, bar_lbl):
        h = max(v * chart_h * 1.2, 0.025)
        col = C["red"] if v >= 0.30 else (C["gold"] if v >= 0.10 else C["green"])
        ax.add_patch(Rectangle((x - 0.22, base_y), 0.44, h,
                                 facecolor=col, edgecolor="white",
                                 lw=1.0, zorder=2))
        ax.text(x, base_y + h + 0.06, f"{v*100:.0f}%",
                ha="center", va="bottom", fontsize=9.5, color="#333")
        ax.text(x, base_y - 0.10, lab, ha="center", va="top",
                fontsize=10, color="#333")
    ax.plot([bar_centers[0] - 0.30, bar_centers[-1] + 0.30],
            [base_y, base_y], color="#444", lw=0.8)
    ax.text(bar_centers[0] - 0.40, base_y - 0.10, "Score:",
            ha="right", va="top", fontsize=10, color="#444", style="italic")

    # Cutoff annotation
    ax.text(LANE_C2, 2.70,
            "Clear break between scores 3 and 4",
            ha="center", fontsize=10, color="#555", style="italic")

    # Cutoff strip
    cut_box = FancyBboxPatch((LANE_C2 - 2.40, 1.00), 4.80, 1.45,
                              boxstyle="round,pad=0.02,rounding_size=0.04",
                              linewidth=1.0, edgecolor=C["blue"],
                              facecolor="#FFFFFF", zorder=2)
    ax.add_patch(cut_box)
    ax.text(LANE_C2, 2.20, "Recommended cutoff:  Model 1 ≥ 4",
            ha="center", fontsize=11, fontweight="bold", color=C["blue"])

    # Two horizontal split bars: low-risk vs high-risk
    # Updated for cutoff ≥ 4 (no focal deficit)
    bar_x0 = LANE_C2 - 2.20
    bar_w = 4.40
    n_low = 8 + 29 + 50 + 64       # scores 0-3
    n_high = 47 + 13 + 3            # scores 4-6
    seg1 = bar_w * (n_low / 214)
    seg2 = bar_w * (n_high / 214)
    ax.add_patch(Rectangle((bar_x0, 1.60), seg1, 0.20,
                             facecolor=C["green"], edgecolor="white", zorder=3))
    ax.add_patch(Rectangle((bar_x0 + seg1, 1.60), seg2, 0.20,
                             facecolor=C["red"], edgecolor="white", zorder=3))
    # Low (≤3): 13/151 = 8.6%; High (≥4): 23/63 = 36.5%
    ax.text(bar_x0 + seg1 / 2, 1.40,
            "Low risk\nn = 151   8.6% rescue",
            ha="center", va="top", fontsize=9.4, color=C["green"],
            fontweight="bold", linespacing=1.1)
    ax.text(bar_x0 + seg1 + seg2 / 2, 1.40,
            "High risk\nn = 63   36.5% rescue",
            ha="center", va="top", fontsize=9.4, color=C["red"],
            fontweight="bold", linespacing=1.1)

    # ============== LANE 3 — Validation ==============
    val_items = [
        ("Discrimination", "AUC, apparent + Harrell-corrected"),
        ("Bootstrap optimism", "1000 nonparametric replicates"),
        ("Calibration", "Hosmer–Lemeshow + visual"),
        ("Per-stratum risk", "Wilson 95% CIs at each score"),
        ("ML comparison", "RF · GBM · Elastic-Net · XGB"),
        ("Decision Curve", "Net benefit vs threshold prob."),
        ("Operating point", "Sens / Spec / PPV / NPV at ≥4"),
        ("Bedside card", "1-page printable PDF"),
    ]
    val_box = FancyBboxPatch((10.95, 0.85), 4.65, 7.05,
                              boxstyle="round,pad=0.02,rounding_size=0.04",
                              linewidth=1.0, edgecolor="#888",
                              facecolor=C["pale"], zorder=2)
    ax.add_patch(val_box)
    n = len(val_items)
    top_y = 7.55
    step = (top_y - 1.30) / (n - 1)
    for i, (k, v) in enumerate(val_items):
        y = top_y - i * step
        ax.add_patch(Rectangle((11.15, y - 0.18), 0.10, 0.36,
                                 facecolor=C["blue"], zorder=3))
        ax.text(11.40, y + 0.08, k, ha="left", va="center",
                fontsize=10.3, fontweight="bold", color=C["blue"])
        ax.text(11.40, y - 0.18, v, ha="left", va="center",
                fontsize=9.4, color="#333")

    # Connectors from models to validation
    arrow(ax, LANE_C2 + 2.45, 7.00, 10.95, 7.00, color="#888")
    arrow(ax, LANE_C2 + 2.45, 6.05, 10.95, 6.05, color="#888")

    # ============== Result strip at bottom ==============
    res_box = FancyBboxPatch((0.4, 0.10), 15.2, 0.55,
                              boxstyle="round,pad=0.02,rounding_size=0.04",
                              linewidth=1.5, edgecolor=C["blue"],
                              facecolor="#FFFFFF", zorder=2)
    ax.add_patch(res_box)
    ax.text(0.7, 0.375, "Result", ha="left", va="center",
            fontsize=10, fontweight="bold", color=C["blue"], style="italic")
    ax.plot([1.55, 1.55], [0.20, 0.55], color=C["lt"], lw=1)
    ax.text(3.4, 0.375, "AUC 0.70 (corrected)", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=C["blue"])
    ax.plot([5.30, 5.30], [0.20, 0.55], color=C["lt"], lw=1)
    ax.text(8.85, 0.375, "Score ≥ 4  →  4.2-fold higher rescue rate (37% vs 9%)",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=C["red"])
    ax.plot([12.40, 12.40], [0.20, 0.55], color=C["lt"], lw=1)
    ax.text(13.95, 0.375, "Calibrated  (HL P > 0.6)",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=C["green"])

    out_png = V2 / "fig0_study_flow.png"
    out_tif = TIFF / "fig0_study_flow.tif"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_tif, dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Wrote {out_png.relative_to(HERE)}")


if __name__ == "__main__":
    main()
