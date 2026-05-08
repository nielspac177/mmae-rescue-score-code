"""Publication-quality figures for MMAE Score v2 (Model 1 + Model 2).

Each figure is rendered both as 600 DPI PNG and as 600 DPI TIFF (LZW)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"
TIFF.mkdir(exist_ok=True, parents=True)

# JAMA palette
C = dict(blue="#374E55", gold="#DF8F44", teal="#00A1D5", red="#B24745",
         green="#79AF97", purple="#6A6599", grey="#80796B", lt="#EFE9DB")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#222",
    "axes.labelcolor": "#222",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#222", "ytick.color": "#222",
    "legend.frameon": False,
    "savefig.dpi": 600,
    "figure.dpi": 110,
})


def save(fig, name: str):
    png = V2 / f"{name}.png"
    tif = TIFF / f"{name}.tif"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(tif, dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  ✓ {name}")


# ----------------------------------------------------------------------
def fig_roc(sc: pd.DataFrame):
    """ROC for Model 1 (full score) and Model 2 (simple score)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    y = sc["y"].values
    for name, col, color, ls in [
        ("Model 1 (primary, max 7)", "score_m1", C["blue"], "-"),
        ("Model 2 (primary, max 4)", "score_m2", C["gold"], "-"),
    ]:
        s = sc[col].values
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, lw=2.4, color=color, ls=ls,
                label=f"{name} — AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=C["grey"], lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC — Predicting MMA embolization rescue")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9.2)
    ax.grid(alpha=0.18)
    save(fig, "fig1_roc")


def fig_score_risk(sc: pd.DataFrame, m1: pd.DataFrame, m2: pd.DataFrame):
    """Side-by-side bar+line: score → observed failure rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw=dict(wspace=0.28))
    for ax, tab, color, title, max_score in [
        (axes[0], m1, C["blue"], "Model 1 (primary, max 7)", 8),
        (axes[1], m2, C["gold"], "Model 2 (primary, max 4)", 5),
    ]:
        x = tab["score"].values
        n = tab["n"].values
        rate = tab["rate"].values * 100
        lo = tab["ci_lo"].values * 100
        hi = tab["ci_hi"].values * 100

        bars = ax.bar(x, rate, width=0.7, color=color, alpha=0.78,
                      edgecolor="white", linewidth=1.4, zorder=2)
        # CI whiskers
        ax.errorbar(x, rate, yerr=[rate - lo, hi - rate],
                    fmt="none", ecolor="#444", capsize=3, lw=1.0, zorder=3)
        # n labels above bars
        ymax = max(hi.max() + 8, 80)
        for xi, ni, ri in zip(x, n, rate):
            ax.text(xi, ri + 1.5, f"n={ni}", ha="center", va="bottom",
                    fontsize=8.5, color="#333")

        ax.set_xticks(np.arange(0, max_score + 1))
        ax.set_xlim(-0.6, max_score + 0.6)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Total score")
        ax.set_ylabel("Observed rescue rate (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    save(fig, "fig2_score_risk")


def fig_calibration(sc: pd.DataFrame):
    """Calibration: predicted probability vs observed failure rate."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    y = sc["y"].values
    for name, col, color in [
        ("Model 1 (full)", "pred_m1", C["blue"]),
        ("Model 2 (simple)", "pred_m2", C["gold"]),
    ]:
        p = sc[col].values
        bins = np.quantile(p, np.linspace(0, 1, 7))
        bins = np.unique(bins)
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
        ax.plot(means_p, means_y, "o-", color=color, lw=2.0, ms=7,
                label=name, zorder=3)
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


def fig_forest(uni: pd.DataFrame):
    """Forest of univariate ORs for the candidate predictors."""
    keep = ["Age <65", "Age 65–80", "Age >80", "SDH volume ≥100 mL",
            "Anticoagulation", "Focal deficit at presentation",
            "Platelets <150 ×10⁹/L", "Antiplatelet therapy",
            "Anterior + posterior embolization"]
    df = uni[uni["variable"].isin(keep)].copy()
    df["variable"] = pd.Categorical(df["variable"], categories=keep, ordered=True)
    df = df.sort_values("variable", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, row in df.iterrows():
        col = C["red"] if row["OR"] > 1 else C["blue"]
        sig = row["p"] < 0.05
        ax.plot([row["OR_lo"], row["OR_hi"]], [i, i],
                color=col, lw=2.6, alpha=0.85 if sig else 0.45,
                solid_capstyle="round")
        ax.scatter(row["OR"], i, s=110 if sig else 65,
                   color=col, edgecolor="white", lw=1.4,
                   zorder=4, alpha=0.95 if sig else 0.6)
    ax.axvline(1.0, color="#888", lw=1.0, ls="--", alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["variable"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for MMA rescue (95% CI, univariate)")
    ax.set_title("Univariate associations with rescue")
    ax.grid(axis="x", alpha=0.15, which="both")

    # Annotate OR (95% CI) on the right edge
    xmax = df["OR_hi"].max() * 1.5
    for i, row in df.iterrows():
        ax.text(xmax, i, f"{row['OR']:.2f} ({row['OR_lo']:.2f}–{row['OR_hi']:.2f})  P={row['p']:.3f}",
                va="center", ha="left", fontsize=8.5, color="#333",
                family="DejaVu Sans Mono")
    ax.set_xlim(df["OR_lo"].min() * 0.5, xmax * 1.05)
    save(fig, "fig4_forest")


def fig_score_table(m1_tab: pd.DataFrame, m2_tab: pd.DataFrame):
    """Side-by-side parallel table: score | n | failures | rate (CI)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw=dict(wspace=0.05))
    for ax, tab, title, color in [
        (axes[0], m1_tab, "Model 1 (primary, max 7)", C["blue"]),
        (axes[1], m2_tab, "Model 2 (primary, max 4)", C["gold"]),
    ]:
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


def fig_score_components(summary: dict):
    """Visual depiction of the two score definitions."""
    items_m1 = [
        ("Age <65", 0), ("Age 65–80", 1), ("Age >80", 2),
        ("SDH volume ≥100 mL", 1), ("Anticoagulation", 1),
        ("Absence of focal deficit", 1), ("Platelets <150 ×10⁹/L", 1),
        ("Antiplatelet therapy", 1), ("Anterior + posterior embolization", 1),
    ]
    items_m2 = [
        ("Age <65", 0), ("Age 65–80", 1), ("Age >80", 2),
        ("SDH volume ≥100 mL", 1), ("Anticoagulation", 1),
        ("Absence of focal deficit", 1),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw=dict(wspace=0.06))
    for ax, items, title, color in [
        (axes[0], items_m1, "Model 1 — primary (max 7 pts)", C["blue"]),
        (axes[1], items_m2, "Model 2 — primary (max 4 pts)", C["gold"]),
    ]:
        ax.axis("off")
        rows = [["Variable", "Points"]]
        for v, p in items:
            rows.append([v, str(p)])
        tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="left",
                       colWidths=[0.78, 0.18],
                       colColours=[color, color])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10.5)
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


def fig_decision_thresholds(m1: pd.DataFrame, m2: pd.DataFrame):
    """Stacked bar showing the discrimination between low/high scores."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw=dict(wspace=0.28))
    for ax, tab, title, color, cut in [
        (axes[0], m1, "Model 1 — cutoff ≥5", C["blue"], 5),
        (axes[1], m2, "Model 2 — cutoff ≥4", C["gold"], 4),
    ]:
        low = tab[tab["score"] < cut]
        high = tab[tab["score"] >= cut]
        groups = ["Low score", "High score"]
        n_g = [low["n"].sum(), high["n"].sum()]
        ev_g = [low["failures"].sum(), high["failures"].sum()]
        rate_g = [e / n if n else 0 for e, n in zip(ev_g, n_g)]
        bars = ax.bar(groups, [r * 100 for r in rate_g], width=0.55,
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


# ----------------------------------------------------------------------
def main():
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    m1 = pd.read_csv(V2 / "m1_risk_by_score.csv")
    m2 = pd.read_csv(V2 / "m2_risk_by_score.csv")
    uni = pd.read_csv(V2 / "univariate_ors.csv")
    with open(V2 / "summary_v2.json") as f:
        summary = json.load(f)
    print("Generating figures…")
    fig_score_components(summary)
    fig_roc(sc)
    fig_score_risk(sc, m1, m2)
    fig_calibration(sc)
    fig_forest(uni)
    fig_score_table(m1, m2)
    fig_decision_thresholds(m1, m2)
    print("Done.")


if __name__ == "__main__":
    main()
