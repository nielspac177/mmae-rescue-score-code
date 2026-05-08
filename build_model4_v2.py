"""Model 4 — data-driven (sensitivity benchmark, with class-imbalance handling).

Pipeline:
  1. Build candidate-feature matrix (median/mode imputation, binary encodings).
  2. Pre-screen for near-zero variance and >0.95 collinearity.
  3. L1-logistic (lasso) feature selection in three flavors:
        a. natural   — no class weighting, no resampling
        b. balanced  — class_weight='balanced' inside both lasso and ML
        c. SMOTE     — synthetic minority oversampling, applied INSIDE each
                        CV fold (pipeline-style) to avoid leakage
     For each flavor, pick the strongest penalty whose CV-AUC is within 1 SE of
     the best, subject to ≤ 7 non-zero coefficients (EPV-5 cap, 36 events).
  4. Refit unpenalized logistic on the natural-flavor selected features →
     publishable coefficients (Model 4).
  5. ML benchmark (RF, GBM, XGB) on the selected features for AUC ceiling.
  6. Bootstrap 95% CI on out-of-fold predictions (1000 replicates).

Outputs: m4_logit_coefs.csv, m4_summary.json, m4_lasso_path.csv,
m4_cv_aucs.csv, m4_proba.csv, m4_imbalance_comparison.csv.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except Exception:
    HAVE_SMOTE = False

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
V2 = HERE / "v2"
RNG = 42
N_SPLITS, N_REPEATS, N_BOOT = 5, 10, 1000


# ----------------------------------------------------------------------
def yes_no(s):
    return s.astype(str).str.strip().str.lower().eq("yes").astype(int)


def build_candidate_matrix(csv: Path):
    df = pd.read_csv(csv)
    y = (df["rescue_surgery"].astype(str).str.strip() == "Yes").astype(int).values
    X = pd.DataFrame(index=df.index)

    for col, name in [("age", "age"),
                      ("plt_num", "plt"),
                      ("inrlastbeforeprocedure", "inr"),
                      ("sdhvolumebaseline", "sdh_vol"),
                      ("midlineshiftmeasureinmmbaseline", "mls_mm"),
                      ("baselinemrs", "mrs"),
                      ("gcsonpresentation", "gcs")]:
        v = pd.to_numeric(df[col], errors="coerce")
        X[name] = v.fillna(v.median())

    for col, name in [("hypertension", "hypertension"),
                      ("statins", "statins"),
                      ("antiplatelet", "antiplatelet"),
                      ("anticoagulation", "anticoag"),
                      ("headache", "headache"),
                      ("nausea", "nausea"),
                      ("focal_deficit", "focal_deficit"),
                      ("seizures", "seizures"),
                      ("gait", "gait"),
                      ("aphasia", "aphasia")]:
        X[name] = yes_no(df[col])

    X["male"] = df["gender_num"].astype(str).str.strip().str.lower().eq("male").astype(int)
    X["liquid_embolic"] = yes_no(df["useofliquidembolic"])
    X["ant_post"] = (df["branches"].astype(str).str.strip()
                      == "Anterior + posterior").astype(int)
    X["standalone"] = (pd.to_numeric(df["standalone1"], errors="coerce").fillna(0) == 1).astype(int)
    X["bilateral"] = (pd.to_numeric(df["bilateral_num2"], errors="coerce").fillna(0) == 1).astype(int)
    X["mls_ge5"] = (pd.to_numeric(df["shift5baseline"], errors="coerce").fillna(0) == 1).astype(int)
    X["sdh_vol_ge100"] = (X["sdh_vol"] >= 100).astype(int)
    X["plt_lt150"] = (X["plt"] < 150).astype(int)
    X["age_pts"] = np.where(X["age"] < 65, 0, np.where(X["age"] <= 80, 1, 2))

    dens = df["densitybaseline"].astype(str).str.strip().str.lower()
    X["mixed_density"] = dens.str.contains("mix|hyper|hetero").fillna(False).astype(int)
    struct = df["structurebaseline"].astype(str).str.strip().str.lower()
    X["separated"] = struct.str.contains("sep|membr|trabe").fillna(False).astype(int)
    membr = df["membranesbasline"].astype(str).str.strip().str.lower()
    X["membranes"] = membr.str.contains("yes").fillna(False).astype(int)
    sub = df["acute_subacutebasline"].astype(str).str.strip().str.lower()
    X["sub_acute"] = sub.str.contains("sub|acute").fillna(False).astype(int)

    keep, dropped = [], []
    for c in X.columns:
        if X[c].nunique() < 2 or X[c].std() < 1e-6:
            dropped.append(c)
        else:
            keep.append(c)
    X = X[keep]
    print(f"[BUILD] candidate matrix: {X.shape[1]} features, n={len(X)}, events={y.sum()}")
    if dropped:
        print(f"[BUILD] dropped near-zero-var: {dropped}")
    return X, y


def remove_collinear(X: pd.DataFrame, thr: float = 0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    if drop:
        print(f"[BUILD] dropped collinear (>|{thr}|): {drop}")
    return X.drop(columns=drop)


# ----------------------------------------------------------------------
def cv_proba(model_factory, X, y, smote=False):
    """Out-of-fold probabilities. SMOTE is applied INSIDE each training fold
    only — no leakage into the held-out fold."""
    proba = np.zeros(len(y))
    rng = np.random.default_rng(RNG)
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RNG + r)
        for tr, te in cv.split(X, y):
            X_tr, y_tr = X[tr], y[tr]
            if smote and HAVE_SMOTE:
                sm = SMOTE(random_state=int(rng.integers(0, 1_000_000)))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            m = model_factory()
            m.fit(X_tr, y_tr)
            proba[te] += m.predict_proba(X[te])[:, 1]
    return proba / N_REPEATS


def boot_auc_ci(y, proba, n_boot=N_BOOT):
    rng = np.random.default_rng(RNG)
    aucs, n = [], len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[idx], proba[idx]))
        except Exception:
            pass
    a = np.array(aucs)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ----------------------------------------------------------------------
def lasso_select(X, y, mode: str, max_features: int = 7):
    """L1-logistic across a path of penalties; pick best CV-AUC subject to
    ≤ max_features non-zero. mode ∈ {'natural', 'balanced', 'smote'}."""
    Cs = np.logspace(-3, 1, 30)
    Xs = StandardScaler().fit_transform(X.values)

    rows = []
    for C in Cs:
        kw = dict(penalty="l1", solver="liblinear", C=C, max_iter=5000)
        if mode == "balanced":
            kw["class_weight"] = "balanced"
        proba = cv_proba(lambda: LogisticRegression(**kw), Xs, y,
                         smote=(mode == "smote"))
        auc = roc_auc_score(y, proba)
        m = LogisticRegression(**kw).fit(Xs, y)
        nnz = int((np.abs(m.coef_) > 1e-6).sum())
        rows.append(dict(mode=mode, C=float(C), nnz=nnz, cv_auc=float(auc)))
    df = pd.DataFrame(rows)
    elig = df[df["nnz"].between(1, max_features)]
    if elig.empty:
        elig = df[df["nnz"] >= 1]
    best = elig.sort_values(["cv_auc", "nnz"], ascending=[False, True]).iloc[0]
    C_best = float(best["C"])

    kw = dict(penalty="l1", solver="liblinear", C=C_best, max_iter=5000)
    if mode == "balanced":
        kw["class_weight"] = "balanced"
    m = LogisticRegression(**kw).fit(Xs, y)
    coefs = m.coef_.flatten()
    selected = [c for c, b in zip(X.columns, coefs) if abs(b) > 1e-6]
    print(f"[LASSO {mode:8s}] best C={C_best:.4f} nnz={int(best['nnz'])} "
          f"CV-AUC={best['cv_auc']:.3f}  features={selected}")
    return selected, df


def evaluate_logit_unpenalized(X_sel: pd.DataFrame, y, mode: str):
    """Refit unpenalized logistic on selected features; return apparent +
    cross-validated AUC."""
    Xc = sm.add_constant(X_sel)
    res = sm.Logit(y, Xc).fit(disp=False, method="bfgs", maxiter=500)
    apparent = roc_auc_score(y, res.predict(Xc).values)

    kw = dict(C=1e6, max_iter=5000, solver="liblinear")
    if mode == "balanced":
        kw["class_weight"] = "balanced"
    proba = cv_proba(
        lambda: Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(**kw))]),
        X_sel.values, y, smote=(mode == "smote"))
    cv_mean, lo, hi = boot_auc_ci(y, proba)
    return res, apparent, cv_mean, lo, hi, proba


# ----------------------------------------------------------------------
def main():
    csv = HERE / "mmaecsv.csv"
    X, y = build_candidate_matrix(csv)
    X = remove_collinear(X)

    paths = []
    selections = {}
    for mode in ["natural", "balanced", "smote" if HAVE_SMOTE else None]:
        if mode is None:
            continue
        sel, dfp = lasso_select(X, y, mode=mode, max_features=7)
        selections[mode] = sel
        paths.append(dfp)
    pd.concat(paths).to_csv(V2 / "m4_lasso_path.csv", index=False)

    # Unpenalized refits → publishable coefficients (natural is the headline)
    refit_rows = []
    proba_dict = {}
    for mode, sel in selections.items():
        res, app, cv_m, lo, hi, proba = evaluate_logit_unpenalized(X[sel], y, mode)
        refit_rows.append(dict(
            mode=mode, n_features=len(sel),
            features=", ".join(sel),
            apparent_auc=app, cv_auc=cv_m, cv_lo=lo, cv_hi=hi,
            brier=float(brier_score_loss(y, proba)),
        ))
        proba_dict[f"M4 logit ({mode})"] = proba
        if mode == "natural":
            ci = np.asarray(res.conf_int())
            coefs = pd.DataFrame({
                "variable": sm.add_constant(X[sel]).columns,
                "coef": res.params.values,
                "se": res.bse.values,
                "OR": np.exp(res.params.values),
                "OR_lo": np.exp(ci[:, 0]),
                "OR_hi": np.exp(ci[:, 1]),
                "p": res.pvalues.values,
            })
            coefs.to_csv(V2 / "m4_logit_coefs.csv", index=False)
            natural_sel = sel
            natural_proba = proba
            natural_auc = (cv_m, lo, hi)

    refit_df = pd.DataFrame(refit_rows)
    refit_df.to_csv(V2 / "m4_imbalance_comparison.csv", index=False)
    print("\n[REFIT] unpenalized logit on selected features by mode:")
    print(refit_df[["mode", "n_features", "apparent_auc",
                    "cv_auc", "cv_lo", "cv_hi", "brier"]]
          .round(3).to_string(index=False))

    # ML benchmark on the natural-flavor selected features (3 flavors of weighting)
    Xsel_v = X[natural_sel].values
    candidates = {
        "RF (natural)": lambda: RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=4,
            random_state=RNG, n_jobs=-1),
        "RF (balanced)": lambda: RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=4,
            class_weight="balanced", random_state=RNG, n_jobs=-1),
        "GBM (natural)": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=RNG),
    }
    try:
        from xgboost import XGBClassifier
        # scale_pos_weight = (negatives / positives)
        spw = float((y == 0).sum() / max(1, (y == 1).sum()))
        candidates["XGB (natural)"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=RNG, verbosity=0,
            use_label_encoder=False)
        candidates["XGB (scale_pos_weight)"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=RNG, verbosity=0,
            use_label_encoder=False)
    except Exception:
        pass

    cv_rows = []
    for name, factory in candidates.items():
        proba = cv_proba(factory, Xsel_v, y)
        cv_m, lo, hi = boot_auc_ci(y, proba)
        cv_rows.append(dict(model=name,
                             cv_auc=float(roc_auc_score(y, proba)),
                             ci_lo=lo, ci_hi=hi))
        proba_dict[name] = proba
        print(f"[ML]  {name:25s} CV-AUC={roc_auc_score(y,proba):.3f} ({lo:.3f}-{hi:.3f})")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(V2 / "m4_cv_aucs.csv", index=False)
    pd.DataFrame(proba_dict).to_csv(V2 / "m4_proba.csv", index=False)

    summary = dict(
        primary_model="natural lasso, unpenalized refit",
        natural_selected_features=natural_sel,
        natural_cv_auc=float(natural_auc[0]),
        natural_cv_ci=[float(natural_auc[1]), float(natural_auc[2])],
        balanced_selected_features=selections.get("balanced", []),
        smote_selected_features=selections.get("smote", []),
        imbalance_comparison=refit_df.to_dict(orient="records"),
        ml_table=cv_df.to_dict(orient="records"),
    )
    (V2 / "m4_summary.json").write_text(json.dumps(summary, indent=2,
                                                    default=float))
    print(f"\nWrote v2/m4_*  (natural-flavor headline CV-AUC "
          f"{natural_auc[0]:.3f} [{natural_auc[1]:.3f}–{natural_auc[2]:.3f}])")


if __name__ == "__main__":
    main()
