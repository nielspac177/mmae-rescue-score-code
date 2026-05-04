"""Figure 0 — Study flow + analysis schema (graphical abstract / STROBE-style)."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"
TIFF.mkdir(exist_ok=True, parents=True)

# JAMA palette
C = dict(blue="#374E55", gold="#DF8F44", teal="#00A1D5", red="#B24745",
         green="#79AF97", grey="#80796B", lt="#EFE9DB", bg="#FFFFFF")

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "savefig.dpi": 600,
    "figure.dpi": 110,
})


def draw_box(ax, x, y, w, h, text, fill, edge="#222", text_color="white",
             font_size=10, bold=True):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.018,rounding_size=0.04",
                         linewidth=1.4, edgecolor=edge, facecolor=fill,
                         zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=font_size, color=text_color,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        lw=1.4, color=color, zorder=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Header
    ax.text(6, 6.65, "Pre-procedural risk score for rescue surgery after MMA embolization for cSDH",
            ha="center", va="top", fontsize=13.5, fontweight="bold", color=C["blue"])
    ax.text(6, 6.30, "Study cohort, predictors, and analytic flow",
            ha="center", va="top", fontsize=10.5, color="#555", style="italic")

    # ----- Cohort -----
    draw_box(ax, 2.0, 5.4, 3.5, 0.7,
             "MMA embolization for cSDH\n(214 consecutive patients)",
             C["blue"], font_size=10.5)

    # ----- Outcome -----
    draw_box(ax, 6.0, 5.4, 3.0, 0.7,
             "Rescue surgery (Yes/No)\n36 events / 16.8%",
             C["red"], font_size=10.5)

    # ----- Follow-up -----
    draw_box(ax, 9.7, 5.4, 3.0, 0.7,
             "Last clinical follow-up\nendpoint ascertainment",
             C["grey"], font_size=10.5)

    arrow(ax, 3.78, 5.4, 4.5, 5.4)
    arrow(ax, 7.5, 5.4, 8.2, 5.4)

    # ----- Predictors -----
    ax.text(2.0, 4.5, "PREDICTORS (prespecified)", ha="center",
            fontsize=10, fontweight="bold", color=C["blue"])
    pred_lines = [
        "• Age (<65 / 65–80 / >80)",
        "• SDH volume ≥100 mL",
        "• Anticoagulation",
        "• Absence of focal deficit",
        "• Platelets <150 ×10⁹/L",
        "• Antiplatelet therapy",
        "• Anterior + posterior emboli.",
    ]
    box = FancyBboxPatch((0.4, 1.2), 3.3, 3.0,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.0, edgecolor="#888", facecolor=C["lt"],
                         zorder=2)
    ax.add_patch(box)
    for i, line in enumerate(pred_lines):
        ax.text(0.6, 3.95 - i * 0.36, line, ha="left", va="center",
                fontsize=10, color="#222")

    # ----- Models -----
    ax.text(6.0, 4.5, "TWO SCORING MODELS", ha="center",
            fontsize=10, fontweight="bold", color=C["blue"])
    draw_box(ax, 6.0, 3.55, 3.0, 0.65,
             "Model 1 — Full (8 pts)\nall 7 predictors",
             C["blue"], font_size=10)
    draw_box(ax, 6.0, 2.65, 3.0, 0.65,
             "Model 2 — Simple (5 pts)\nage, vol., anticoag, deficit",
             C["gold"], font_size=10)

    arrow(ax, 3.7, 2.7, 4.5, 3.55)
    arrow(ax, 3.7, 2.7, 4.5, 2.65)

    # ----- Outputs / metrics -----
    ax.text(9.7, 4.5, "INTERNAL VALIDATION", ha="center",
            fontsize=10, fontweight="bold", color=C["blue"])
    metrics = [
        "AUC (apparent + Harrell-corrected)",
        "1000-bootstrap optimism correction",
        "Hosmer–Lemeshow calibration",
        "Wilson CIs per score stratum",
        "Score → rescue-rate table",
    ]
    box = FancyBboxPatch((8.05, 1.2), 3.3, 3.0,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.0, edgecolor="#888", facecolor=C["lt"],
                         zorder=2)
    ax.add_patch(box)
    for i, line in enumerate(metrics):
        ax.text(8.25, 3.95 - i * 0.36, "• " + line, ha="left", va="center",
                fontsize=10, color="#222")

    arrow(ax, 7.5, 3.55, 8.05, 3.6)
    arrow(ax, 7.5, 2.65, 8.05, 2.6)

    # ----- Result strip at bottom -----
    box = FancyBboxPatch((0.4, 0.15), 11.0, 0.85,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.4, edgecolor=C["blue"],
                         facecolor="#F5F2EA", zorder=2)
    ax.add_patch(box)
    ax.text(5.9, 0.78, "Result", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["blue"])
    ax.text(5.9, 0.42,
            "Model 1 AUC 0.73 (corrected 0.73)   |   Model 2 AUC 0.68 (0.68)   |   "
            "Score ≥5 → 4.7-fold higher rescue rate (43% vs 9%)",
            ha="center", va="center", fontsize=10.5, color="#222")

    out_png = V2 / "fig0_study_flow.png"
    out_tif = TIFF / "fig0_study_flow.tif"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_tif, dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Wrote {out_png.relative_to(HERE)} and {out_tif.relative_to(HERE)}")


if __name__ == "__main__":
    main()
