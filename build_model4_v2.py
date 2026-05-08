"""Model 4 — data-driven derivation.

Pipeline:
  1. Build candidate-feature matrix (median/mode imputation; binary encodings).
  2. Pre-screen for near-zero variance and >0.95 collinearity.
  3. L1-logistic (lasso) with stratified 5×10 CV across a path of penalties;
     pick the lambda that gives best mean CV AUC subject to EPV-5 (≤7 features).
  4. Refit unpenalized logistic on the selected features → publishable coefficients.
  5. ML benchmark (RF, GBM, XGB) on the same selected features for AUC ceiling.
  6. Bootstrap 95% CI on the held-out CV AUC.

Outputs to v2/:
  m4_logit_coefs.csv, m4_selected_features.json, m4_cv_aucs.csv,
  m4_predictions.csv, m4_summary.json
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
V2 = HERE / "v2"
RNG = 42
N_SPLITS, N_REPEATS, N_BOOT = 5, 10, 1000


# ----------------------------------------------------------------------
def yes_no(s):
    return s.astype(str).str.strip().str.lower().eq("yes").astype(int)


def build_candidate_matrix(csv: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(csv)
    y = (df["rescue_surgery"].astype(str).str.strip() == "Yes").astype(int).values

    X = pd.DataFrame(index=df.index)

    # Continuous (median impute)
    for col, name in [("age", "age"),
                      ("plt_num", "plt"),
                      ("inrlastbeforeprocedure", "inr"),
                      ("sdhvolumebaseline", "sdh_vol"),
                      ("midlineshiftmeasureinmmbaseline", "mls_mm"),
                      ("baselinemrs", "mrs"),
                      ("gcsonpresentation", "gcs")]:
        v = pd.to_numeric(df[col], errors="coerce")
        X[name] = v.fillna(v.median())

    # Binary Yes/No → 1/0
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

    # Sex (Male=1)
    X["male"] = df["gender_num"].astype(str).str.strip().str.lower().eq("male").astype(int)

    # Procedural binaries
    X["liquid_embolic"] = yes_no(df["useofliquidembolic"])
    X["ant_post"] = (df["branches"].astype(str).str.strip()
                      == "Anterior + posterior").astype(int)
    X["standalone"] = (pd.to_numeric(df["standalone1"], errors="coerce").fillna(0) == 1).astype(int)
    X["bilateral"] = (pd.to_numeric(df["bilateral_num2"], errors="coerce").fillna(0) == 1).astype(int)

    # Imaging binaries
    X["mls_ge5"] = (pd.to_numeric(df["shift5baseline"], errors="coerce").fillna(0) == 1).astype(int)
    X["sdh_vol_ge100"] = (X["sdh_vol"] >= 100).astype(int)
    X["plt_lt150"] = (X["plt"] < 150).astype(int)
    X["age_pts"] = np.where(X["age"] < 65, 0, np.where(X["age"] <= 80, 1, 2))

    # Hematoma phenotype: density (mixed/hyperdense), structure (separated/membranes)
    dens = df["densitybaseline"].astype(str).str.strip().str.lower()
    X["mixed_density"] = dens.str.contains("mix|hyper|hetero").fillna(False).astype(int)
    struct = df["structurebaseline"].astype(str).str.strip().str.lower()
    X["separated"] = struct.str.contains("sep|membr|trabe").fillna(False).astype(int)
    membr = df["membranesbasline"].astype(str).str.strip().str.lower()
    X["membranes"] = membr.str.contains("yes").fillna(False).astype(int)
    sub = df["acute_subacutebasline"].astype(str).str.strip().str.lower()
    X["sub_acute"] = sub.str.contains("sub|acute").fillna(False).astype(int)

    # Drop near-zero-variance columns
    keep, dropped = [], []
    for c in X.columns:
        v = X[c].nunique()
        sd = X[c].std()
        if v < 2 or sd < 1e-6:
            dropped.append(c)
        else:
            keep.append(c)
    X = X[keep]

    print(f"[BUILD] candidate matrix: {X.shape[1]} features, n={len(X)}, events={y.sum()}")
    if dropped:
        print(f"[BUILD] dropped near-zero-var: {dropped}")
    return X, y, list(X.columns)


def remove_collinear(X: pd.DataFrame, thr: float = 0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    if drop:
        print(f"[BUILD] dropped collinear (>|{thr}|): {drop}")
    return X.drop(columns=drop), drop


def cv_proba(model_factory, X, y):
    proba = np.zeros(len(y))
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RNG + r)
        for tr, te in cv.split(X, y):
            m = model_factory()
            m.fit(X[tr], y[tr])
            proba[te] += m.predict_proba(X[te])[:, 1]
    return proba / N_REPEATS


def boot_auc_ci(y, proba, n_boot=N_BOOT):
    rng = np.random.default_rng(RNG)
    aucs = []
    n = len(y)
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
def lasso_feature_selection(X, y, max_features=7):
    """Run L1-logistic across a wide path of penalties; pick the strongest
    penalty (sparsest model) whose CV-AUC is within 1 SE of the best, subject
    to ≤ max_features non-zero coefficients (EPV-5 cap)."""
    Cs = np.logspace(-3, 1, 40)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    rows = []
    for C in Cs:
        proba = cv_proba(
            lambda: LogisticRegression(penalty="l1", solver="liblinear",
                                       C=C, max_iter=5000),
            Xs, y)
        auc = roc_auc_score(y, proba)
        # fit once on full data to measure non-zero count at this C
        m = LogisticRegression(penalty="l1", solver="liblinear",
                                C=C, max_iter=5000).fit(Xs, y)
        nnz = int((np.abs(m.coef_) > 1e-6).sum())
        rows.append(dict(C=float(C), nnz=nnz, cv_auc=float(auc)))
    df = pd.DataFrame(rows)

    # Best AUC subject to nnz ≤ max_features
    elig = df[df["nnz"].between(1, max_features)]
    if elig.empty:
        # fall back to fewest non-zero coef
        elig = df[df["nnz"] >= 1]
    best = elig.sort_values(["cv_auc", "nnz"], ascending=[False, True]).iloc[0]
    C_best = float(best["C"])
    print(f"[LASSO] best C={C_best:.4f}  nnz={int(best['nnz'])}  CV-AUC={best['cv_auc']:.3f}")

    m = LogisticRegression(penalty="l1", solver="liblinear",
                            C=C_best, max_iter=5000).fit(Xs, y)
    coefs = m.coef_.flatten()
    selected = [c for c, b in zip(X.columns, coefs) if abs(b) > 1e-6]
    print(f"[LASSO] selected ({len(selected)}): {selected}")
    return selected, df


# ----------------------------------------------------------------------
def main():
    csv = HERE / "mmaecsv.csv"
    X, y, _ = build_candidate_matrix(csv)
    X, _ = remove_collinear(X)

    # ----- Feature selection via lasso (EPV-5 cap = 7) -----
    selected, lasso_path = lasso_feature_selection(X, y, max_features=7)
    lasso_path.to_csv(V2 / "m4_lasso_path.csv", index=False)

    # ----- Refit unpenalized logistic on selected features -----
    Xsel = X[selected]
    Xc = sm.add_constant(Xsel)
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
    coefs.to_csv(V2 / "m4_logit_coefs.csv", index=False)

    pred_apparent = res.predict(Xc).values
    auc_apparent = roc_auc_score(y, pred_apparent)

    # ----- 5×10 CV AUC on the unpenalized logit using the selected features -----
    Xsel_v = Xsel.values
    proba_logit = cv_proba(
        lambda: Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(C=1e6, max_iter=5000,
                                                      solver="liblinear"))]),
        Xsel_v, y)
    auc_logit_cv, lo_l, hi_l = boot_auc_ci(y, proba_logit)

    # ----- ML candidates on the SAME selected feature set -----
    candidates = {
        "Logistic (selected)":  ("ml", lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1e6, max_iter=5000,
                                       solver="liblinear"))])),
        "Random forest":         ("ml", lambda: RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=4,
            random_state=RNG, n_jobs=-1)),
        "Gradient boosting":     ("ml", lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=RNG)),
    }
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = ("ml", lambda: XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=RNG,
            verbosity=0, use_label_encoder=False))
    except Exception:
        pass

    rows = []
    proba_dict = {}
    for name, (_, factory) in candidates.items():
        proba = cv_proba(factory, Xsel_v, y)
        mean, lo, hi = boot_auc_ci(y, proba)
        rows.append(dict(model=f"{name} (selected 7)",
                         cv_auc=float(roc_auc_score(y, proba)),
                         boot_mean=mean, ci_lo=lo, ci_hi=hi))
        proba_dict[f"{name} (selected 7)"] = proba
        print(f"[ML-7] {name:25s} CV-AUC={roc_auc_score(y, proba):.3f}  ({lo:.3f}–{hi:.3f})")

    # AUC ceiling: same ML on FULL candidate set (no EPV cap)
    Xfull = X.values
    for name, (_, factory) in candidates.items():
        proba = cv_proba(factory, Xfull, y)
        mean, lo, hi = boot_auc_ci(y, proba)
        rows.append(dict(model=f"{name} (full {Xfull.shape[1]})",
                         cv_auc=float(roc_auc_score(y, proba)),
                         boot_mean=mean, ci_lo=lo, ci_hi=hi))
        proba_dict[f"{name} (full {Xfull.shape[1]})"] = proba
        print(f"[ML-{Xfull.shape[1]}] {name:25s} CV-AUC={roc_auc_score(y, proba):.3f}  ({lo:.3f}–{hi:.3f})")
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(V2 / "m4_cv_aucs.csv", index=False)
    pd.DataFrame(proba_dict).to_csv(V2 / "m4_proba.csv", index=False)

    # Pick best CV-AUC ML
    best_row = cv_df.sort_values("cv_auc", ascending=False).iloc[0]
    print(f"[ML] best: {best_row['model']}  CV-AUC={best_row['cv_auc']:.3f}")

    summary = dict(
        selected_features=selected,
        n_features=len(selected),
        n=len(y), events=int(y.sum()),
        apparent_logit_auc=float(auc_apparent),
        cv_logit_auc_mean=auc_logit_cv,
        cv_logit_auc_ci=[lo_l, hi_l],
        cv_brier_logit=float(brier_score_loss(y, proba_logit)),
        best_ml=str(best_row["model"]),
        best_ml_cv_auc=float(best_row["cv_auc"]),
        best_ml_ci=[float(best_row["ci_lo"]), float(best_row["ci_hi"])],
        ml_table=cv_df.to_dict(orient="records"),
        coefficients=coefs.to_dict(orient="records"),
    )
    (V2 / "m4_summary.json").write_text(json.dumps(summary, indent=2,
                                                    default=float))
    print(f"\nWrote v2/m4_*  ({len(selected)} features, "
          f"logit CV-AUC {auc_logit_cv:.3f} [{lo_l:.3f}–{hi_l:.3f}])")


if __name__ == "__main__":
    main()
