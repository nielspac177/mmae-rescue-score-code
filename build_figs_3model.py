"""Regenerate Figures 0-components, 2 (score-risk), 3 (calibration),
5 (parallel tables), 6 (decision threshold), and 10 (DCA) so they show
all three integer scoring models (Model 1 / Model 3 / Model 2)."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"
TIFF.mkdir(exist_ok=True, parents=True)

C = dict(blue="#374E55", gold="#DF8F44", teal="#00A1D5", red="#B24745",
         green="#79AF97", purple="#6A6599", grey="#80796B", lt="#EFE9DB")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 110,
})


def save(fig, name):
    fig.savefig(V2 / f"{name}.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(TIFF / f"{name}.tif", dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# =================================================================
# Figure 0 — score components (3 panels)
# =================================================================
def fig0_score_components():
    items_m1 = [
        ("Age <65", 0), ("Age 65–80", 1), ("Age >80", 2),
        ("SDH volume ≥100 mL", 1), ("Anticoagulation", 1),
        ("Platelets <150 ×10⁹/L", 1),
        ("Antiplatelet therapy", 1), ("Anterior + posterior embolization", 1),
    ]
    items_m3 = [
        ("Age <65", 0), ("Age 65–85", 1), ("Age >85", 2),
        ("SDH volume ≥100 mL", 1),
        ("Platelets <150 ×10⁹/L", 1),
        ("Antiplatelet therapy", 1),
    ]
    items_m2 = [
        ("Age <65", 0), ("Age 65–80", 1), ("Age >80", 2),
        ("SDH volume ≥100 mL", 1), ("Anticoagulation", 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8),
                             gridspec_kw=dict(wspace=0.04))
    cfg = [
        (axes[0], items_m1, "Model 1 — full (max 7 pts)", C["blue"]),
        (axes[1], items_m3, "Model 3 (max 5 pts)", C["purple"]),
        (axes[2], items_m2, "Model 2 — simple (max 4 pts)", C["gold"]),
    ]
    for ax, items, title, color in cfg:
        ax.axis("off")
        rows = [["Variable", "Points"]]
        for v, p in items:
            rows.append([v, str(p)])
        tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="left",
                       colWidths=[0.78, 0.18],
                       colColours=[color, color])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.55)
        for k, cell in tbl.get_celld().items():
            r, c = k
            if r == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_height(0.10)
            else:
                cell.set_facecolor("white" if r % 2 == 0 else "#F5F2EA")
                cell.set_edgecolor("#888")
                if c == 1:
                    cell.set_text_props(ha="center")
                    cell.set_text_props(weight="bold", color=C["red"])
        ax.set_title(title, fontsize=12, weight="bold", pad=14)
    save(fig, "fig0_score_components")
    print("  ✓ fig0_score_components (3 panels)")


# =================================================================
# Figure 2 — observed rescue rate by score (3 panels)
# =================================================================
def fig2_score_risk():
    m1 = pd.read_csv(V2 / "m1_risk_by_score.csv")
    m3 = pd.read_csv(V2 / "m3_risk_by_score.csv")
    m2 = pd.read_csv(V2 / "m2_risk_by_score.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw=dict(wspace=0.28))
    cfg = [
        (axes[0], m1, C["blue"],   "Model 1 — full (max 7)", 7),
        (axes[1], m3, C["purple"], "Model 3 (max 5)", 5),
        (axes[2], m2, C["gold"],   "Model 2 — simple (max 4)", 4),
    ]
    # Standardise y-axis so the three panels compare directly
    panel_ymax = 80
    for ax, tab, color, title, mx in cfg:
        x = tab["score"].values
        n = tab["n"].values
        rate = tab["rate"].values * 100
        lo = tab["ci_lo"].values * 100
        hi = tab["ci_hi"].values * 100
        ax.bar(x, rate, width=0.7, color=color, alpha=0.78,
               edgecolor="white", linewidth=1.4, zorder=2)
        ax.errorbar(x, rate, yerr=[rate - lo, hi - rate],
                    fmt="none", ecolor="#444", capsize=3, lw=1.0, zorder=3)
        for xi, ni, ri in zip(x, n, rate):
            ax.text(xi, min(ri + 1.5, panel_ymax - 3),
                    f"n={ni}", ha="center", va="bottom",
                    fontsize=8.5, color="#333")
        ax.set_xticks(np.arange(0, mx + 1))
        ax.set_xlim(-0.6, mx + 0.6)
        ax.set_ylim(0, panel_ymax)
        ax.set_xlabel("Total score")
        ax.set_ylabel("Observed rescue rate (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    save(fig, "fig2_score_risk")
    print("  ✓ fig2_score_risk (3 panels)")


# =================================================================
# Figure 3 — calibration (3 lines)
# =================================================================
def fig3_calibration():
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    y = sc["y"].values
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    cfg = [
        ("Model 1 (primary)", "pred_m1", C["blue"],   "o", "-"),
        ("Model 2 (primary)", "pred_m2", C["gold"],   "s", "-"),
        ("Model 3 (sens.)",   "pred_m3", C["purple"], "^", "--"),
    ]
    for name, col, color, marker, ls in cfg:
        if col not in sc.columns:
            continue
        p = sc[col].values
        bins = np.unique(np.quantile(p, np.linspace(0, 1, 7)))
        if len(bins) < 3:
            continue
        idx = np.digitize(p, bins[1:-1])
        means_p, means_y, ses = [], [], []
        for k in range(len(bins) - 1):
            sel = idx == k
            if sel.sum() < 4:
                continue
            means_p.append(p[sel].mean())
            means_y.append(y[sel].mean())
            ses.append(np.sqrt(y[sel].mean() * (1 - y[sel].mean()) / sel.sum()))
        ax.plot(means_p, means_y, color=color, lw=2.0, ms=7,
                marker=marker, linestyle=ls, label=name, zorder=3)
        ax.errorbar(means_p, means_y, yerr=ses, fmt="none",
                    ecolor=color, alpha=0.5, capsize=2.5, lw=0.9)
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1.1, alpha=0.7,
            label="Ideal")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(alpha=0.18)
    save(fig, "fig3_calibration")
    print("  ✓ fig3_calibration (3 lines)")


# =================================================================
# Figure 5 — parallel score tables (3 panels)
# =================================================================
def fig5_score_tables():
    m1 = pd.read_csv(V2 / "m1_risk_by_score.csv")
    m3 = pd.read_csv(V2 / "m3_risk_by_score.csv")
    m2 = pd.read_csv(V2 / "m2_risk_by_score.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             gridspec_kw=dict(wspace=0.05))
    cfg = [
        (axes[0], m1, "Model 1 — full (max 7)", C["blue"]),
        (axes[1], m3, "Model 3 (max 5)", C["purple"]),
        (axes[2], m2, "Model 2 — simple (max 4)", C["gold"]),
    ]
    for ax, tab, title, color in cfg:
        ax.axis("off")
        rows = [["Score", "n", "Failures", "Rate (%)", "95% CI"]]
        for _, r in tab.iterrows():
            rows.append([f"{int(r['score'])}", f"{int(r['n'])}",
                         f"{int(r['failures'])}",
                         f"{r['rate']*100:.1f}",
                         f"{r['ci_lo']*100:.1f}–{r['ci_hi']*100:.1f}"])
        tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center",
                       colColours=[color] * 5)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.55)
        for k, cell in tbl.get_celld().items():
            r, c = k
            if r == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_height(0.085)
            else:
                cell.set_facecolor("white" if r % 2 == 0 else "#F5F2EA")
                cell.set_edgecolor("#888")
        ax.set_title(title, fontsize=12, weight="bold", pad=14)
    fig.suptitle("Parallel score → rescue-rate tables", fontsize=13,
                 weight="bold", y=1.02)
    save(fig, "fig5_score_tables")
    print("  ✓ fig5_score_tables (3 panels)")


# =================================================================
# Figure 6 — decision-threshold dichotomization (3 panels)
# =================================================================
def fig6_decision_threshold():
    m1 = pd.read_csv(V2 / "m1_risk_by_score.csv")
    m3 = pd.read_csv(V2 / "m3_risk_by_score.csv")
    m2 = pd.read_csv(V2 / "m2_risk_by_score.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw=dict(wspace=0.28))
    cfg = [
        (axes[0], m1, "Model 1 — cutoff ≥ 4", C["blue"],   4),
        (axes[1], m3, "Model 3 — cutoff ≥ 3", C["purple"], 3),
        (axes[2], m2, "Model 2 — cutoff ≥ 3", C["gold"],   3),
    ]
    for ax, tab, title, color, cut in cfg:
        low = tab[tab["score"] < cut]
        high = tab[tab["score"] >= cut]
        n_g = [low["n"].sum(), high["n"].sum()]
        ev_g = [low["failures"].sum(), high["failures"].sum()]
        rate_g = [e / n if n else 0 for e, n in zip(ev_g, n_g)]
        bars = ax.bar(["Low score", "High score"],
                      [r * 100 for r in rate_g], width=0.55,
                      color=[color, C["red"]], alpha=0.8,
                      edgecolor="white", linewidth=1.4)
        for b, n, e, r in zip(bars, n_g, ev_g, rate_g):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{e}/{n} = {r*100:.1f}%", ha="center", fontsize=10)
        ax.set_ylabel("Rescue rate (%)")
        ax.set_ylim(0, max(60, max(rate_g) * 100 * 1.3))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    save(fig, "fig6_decision_threshold")
    print("  ✓ fig6_decision_threshold (3 panels)")


# =================================================================
# Figure 10 — Decision Curve Analysis (3 lines + treat-all/none)
# =================================================================
def fig10_dca():
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    y = sc["y"].values

    def net_benefit(prob, t):
        n = len(y)
        pred = prob >= t
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        if t >= 1: return 0
        return tp / n - (fp / n) * (t / (1 - t))

    thresholds = np.linspace(0.02, 0.55, 60)
    nb_m1   = np.array([net_benefit(sc["pred_m1"].values, t) for t in thresholds])
    nb_m3   = np.array([net_benefit(sc["pred_m3"].values, t) for t in thresholds])
    nb_m2   = np.array([net_benefit(sc["pred_m2"].values, t) for t in thresholds])
    nb_all  = np.array([net_benefit(np.ones(len(y)),       t) for t in thresholds])
    nb_none = np.zeros_like(thresholds)

    # 5-pt moving average to smooth the integer-score step jitter
    def smooth(arr, k=5):
        kern = np.ones(k) / k
        return np.convolve(arr, kern, mode="same")

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(thresholds, smooth(nb_m1), color=C["blue"], lw=2.6,
            label="Model 1 (primary)")
    ax.plot(thresholds, smooth(nb_m2), color=C["gold"], lw=2.6,
            label="Model 2 (primary)")
    ax.plot(thresholds, smooth(nb_m3), color=C["purple"], lw=2.0, ls="--",
            label="Model 3 (sensitivity)")
    ax.plot(thresholds, nb_all,  color=C["red"],  lw=1.4, ls="--",
            label="Treat all", alpha=0.85)
    ax.plot(thresholds, nb_none, color=C["grey"], lw=1.4, ls=":",
            label="Treat none", alpha=0.85)
    ax.set_xlabel("Threshold probability of rescue surgery")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.set_xlim(0.02, 0.55)
    ax.set_ylim(min(0, min(nb_m1) - 0.01), max(nb_all + nb_m1) * 1.1)
    ax.axhline(0, color="#888", lw=0.6)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right")
    save(fig, "fig10_dca")
    print("  ✓ fig10_dca (3 lines + treat-all/none)")


def main():
    print("Regenerating figures with Model 3 included…")
    fig0_score_components()
    fig2_score_risk()
    fig3_calibration()
    fig5_score_tables()
    fig6_decision_threshold()
    fig10_dca()
    print("Done.")


if __name__ == "__main__":
    main()
