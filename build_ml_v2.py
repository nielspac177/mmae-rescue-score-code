"""ML comparison vs integer score — RF, GBM, ElasticNet, optional XGB.
5×10 stratified repeated CV with bootstrap AUC CIs and ROC overlay."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).parent
V2 = HERE / "v2"
TIFF = V2 / "tiff"

RNG = 42
N_SPLITS = 5
N_REPEATS = 10
N_BOOT = 1000

C = dict(blue="#374E55", gold="#DF8F44", teal="#00A1D5", red="#B24745",
         green="#79AF97", purple="#6A6599", grey="#80796B")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600,
})


def cv_proba(model_factory, X, y, n_repeats=N_REPEATS, n_splits=N_SPLITS):
    """Average out-of-fold predicted probabilities across repeats."""
    proba = np.zeros(len(y))
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=RNG + r)
        for tr, te in cv.split(X, y):
            m = model_factory()
            m.fit(X[tr], y[tr])
            proba[te] += m.predict_proba(X[te])[:, 1]
    return proba / n_repeats


def boot_auc_ci(y, proba, n_boot=N_BOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[idx], proba[idx]))
        except Exception:
            pass
    arr = np.array(aucs)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main():
    sc = pd.read_csv(V2 / "scored_cohort_v2.csv")
    feats = ["age_pts", "sdh_vol_ge100", "anticoag", "no_focal_deficit",
             "plt_lt150", "antiplatelet", "ant_post"]
    X = sc[feats].values.astype(float)
    y = sc["y"].values.astype(int)

    # Try XGBoost
    has_xgb = False
    try:
        from xgboost import XGBClassifier  # noqa
        has_xgb = True
    except Exception:
        pass

    models = {
        "Integer score (Model 1)": ("score", sc["score_m1"].values.astype(float)),
        "Logistic regression": ("ml", lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, solver="liblinear")),
        ])),
        "Elastic-net logistic": ("ml", lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                       solver="saga", max_iter=4000, C=1.0)),
        ])),
        "Random forest": ("ml", lambda: RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=4,
            random_state=RNG, n_jobs=-1)),
        "Gradient boosting": ("ml", lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=RNG)),
    }
    if has_xgb:
        from xgboost import XGBClassifier
        models["XGBoost"] = ("ml", lambda: XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RNG,
            verbosity=0, use_label_encoder=False))

    results = []
    proba_dict = {}
    for name, (kind, m) in models.items():
        if kind == "score":
            proba = m
        else:
            proba = cv_proba(m, X, y)
        apparent = roc_auc_score(y, proba)
        mean, lo, hi = boot_auc_ci(y, proba)
        results.append(dict(model=name, apparent=apparent,
                            boot_mean=mean, ci_lo=lo, ci_hi=hi))
        proba_dict[name] = proba
        print(f"{name:30s}  AUC={apparent:.3f}  ({lo:.3f}–{hi:.3f})")

    df = pd.DataFrame(results)
    df.to_csv(V2 / "ml_comparison.csv", index=False)

    # ROC overlay
    palette = [C["blue"], C["gold"], C["teal"], C["green"], C["purple"], C["red"]]
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for (name, proba), col in zip(proba_dict.items(), palette):
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, lw=2.2, color=col, alpha=0.92,
                label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1.0, alpha=0.7)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ML comparison — ROC curves")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.18)
    fig.savefig(V2 / "fig7_ml_roc.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(TIFF / "fig7_ml_roc.tif", dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Bar of AUCs with CI whiskers
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = df["model"].tolist()
    vals = df["apparent"].values
    err_lo = vals - df["ci_lo"].values
    err_hi = df["ci_hi"].values - vals
    bars = ax.barh(np.arange(len(names))[::-1], vals,
                   color=palette[:len(names)], alpha=0.85,
                   edgecolor="white", linewidth=1.0)
    ax.errorbar(vals, np.arange(len(names))[::-1],
                xerr=[err_lo, err_hi], fmt="none",
                ecolor="#444", capsize=3, lw=1.0)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, len(names) - 1 - i, f"  {v:.3f}",
                va="center", fontsize=10)
    ax.set_yticks(np.arange(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlim(0.55, 0.85)
    ax.set_xlabel("AUC for rescue prediction (apparent + 1000-bootstrap 95% CI)")
    ax.set_title("Model comparison — discrimination")
    ax.grid(axis="x", alpha=0.18)
    fig.savefig(V2 / "fig8_ml_bars.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(TIFF / "fig8_ml_bars.tif", dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Save proba for downstream use
    pd.DataFrame(proba_dict).to_csv(V2 / "ml_proba.csv", index=False)
    print(f"\nWrote v2/ml_comparison.csv, fig7_ml_roc.png, fig8_ml_bars.png")


if __name__ == "__main__":
    main()
