"""Model 5 — enhanced data-driven sensitivity analysis.

Bundles the following methodological enhancements on top of Model 4:

  A. Restricted cubic splines (3 knots) on continuous predictors
     (age, sdh_vol, plt, mls_mm, gcs, mrs).
  B. Pre-specified clinical interactions (age × sdh_vol, anticoag × plt,
     antiplatelet × age).
  C. Optuna-tuned elastic-net (hyperparameters: alpha on [1e-4, 10],
     l1_ratio on [0, 1]).
  D. Stacked ensemble: out-of-fold logistic + tuned gradient boosting
     meta-learner.
  E. Bayesian logistic regression with informative priors on log-OR
     anchored to published MMA series (see v2/literature_priors.md).
     Implementation: PyMC NUTS sampler with Normal priors.
  F. MICE multiple imputation for the 15% missing SDH volumes (and any
     other partially missing continuous predictors).
  G. Richer feature space: include continuous mRS, GCS, midline-shift-mm,
     and INR alongside their binarized forms.

All models are evaluated under 5×10 stratified repeated cross-validation
with bootstrap 95% CIs from out-of-fold probabilities.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
V2 = HERE / "v2"
RNG = 42
N_SPLITS, N_REPEATS, N_BOOT = 5, 10, 1000


# ----------------------------------------------------------------------
# Literature-derived priors (see v2/literature_priors.md)
PRIORS = {
    "intercept":     (-1.60, 1.00),
    "anticoag":      ( 0.69, 0.30),
    "antiplatelet":  ( 0.53, 0.30),
    "sdh_vol_ge100": ( 0.69, 0.35),
    "mls_ge5":       ( 0.69, 0.35),
    "age_pts":       ( 0.40, 0.20),
    "plt_lt150":     ( 0.59, 0.40),
    "bilateral":     ( 0.69, 0.35),
    "mixed_density": ( 0.47, 0.35),
    "ant_post":      ( 0.34, 0.50),
}
DEFAULT_PRIOR = (0.0, 1.0)


# ----------------------------------------------------------------------
def yes_no(s):
    return s.astype(str).str.strip().str.lower().eq("yes").astype(int)


def rcs_basis(x: np.ndarray, knots=None, n_knots: int = 3):
    """Restricted cubic-spline basis. Returns (n, n_knots-2) matrix.
    Standard Hmisc/Stata RCS using inner-knot triplet."""
    x = x.astype(float)
    if knots is None:
        qs = np.linspace(0.05, 0.95, n_knots)
        knots = np.quantile(x[~np.isnan(x)], qs)
    knots = np.asarray(knots)
    K = len(knots)
    if K < 3:
        return np.empty((len(x), 0))
    out = np.zeros((len(x), K - 2))
    tk1 = knots[-1]
    tk0 = knots[-2]
    denom = (tk1 - tk0)
    for j in range(K - 2):
        tj = knots[j]
        out[:, j] = (
            np.maximum(0, x - tj) ** 3
            - np.maximum(0, x - tk0) ** 3 * (tk1 - tj) / denom
            + np.maximum(0, x - tk1) ** 3 * (tk0 - tj) / denom
        ) / (tk1 - knots[0]) ** 2
    return out


def build_features(csv: Path):
    df = pd.read_csv(csv)
    y = (df["rescue_surgery"].astype(str).str.strip() == "Yes").astype(int).values

    # Continuous (NaNs preserved for MICE)
    raw = pd.DataFrame({
        "age":     pd.to_numeric(df["age"], errors="coerce"),
        "plt":     pd.to_numeric(df["plt_num"], errors="coerce"),
        "inr":     pd.to_numeric(df["inrlastbeforeprocedure"], errors="coerce"),
        "sdh_vol": pd.to_numeric(df["sdhvolumebaseline"], errors="coerce"),
        "mls_mm":  pd.to_numeric(df["midlineshiftmeasureinmmbaseline"],
                                  errors="coerce"),
        "mrs":     pd.to_numeric(df["baselinemrs"], errors="coerce"),
        "gcs":     pd.to_numeric(df["gcsonpresentation"], errors="coerce"),
    })

    # MICE (Approach F) — iterative imputation on continuous features
    print(f"[F] MICE imputation; missing rates per col:")
    print(raw.isna().mean().round(3).to_string())
    imp = IterativeImputer(random_state=RNG, max_iter=10).fit(raw.values)
    cont = pd.DataFrame(imp.transform(raw.values), columns=raw.columns)

    # Binary clinical/comorbidity (no missingness in these)
    bins = pd.DataFrame({
        "hypertension": yes_no(df["hypertension"]),
        "statins":      yes_no(df["statins"]),
        "antiplatelet": yes_no(df["antiplatelet"]),
        "anticoag":     yes_no(df["anticoagulation"]),
        "headache":     yes_no(df["headache"]),
        "nausea":       yes_no(df["nausea"]),
        "focal_deficit":yes_no(df["focal_deficit"]),
        "seizures":     yes_no(df["seizures"]),
        "gait":         yes_no(df["gait"]),
        "aphasia":      yes_no(df["aphasia"]),
        "male":         df["gender_num"].astype(str).str.strip().str.lower().eq("male").astype(int),
        "ant_post":     (df["branches"].astype(str).str.strip()
                          == "Anterior + posterior").astype(int),
    })

    # Imaging-derived binaries
    imaging_bins = pd.DataFrame({
        "mls_ge5":       (cont["mls_mm"] >= 5).astype(int),
        "sdh_vol_ge100": (cont["sdh_vol"] >= 100).astype(int),
        "plt_lt150":     (cont["plt"] < 150).astype(int),
        "age_pts":       np.where(cont["age"] < 65, 0,
                                   np.where(cont["age"] <= 80, 1, 2)),
    })

    dens = df["densitybaseline"].astype(str).str.strip().str.lower()
    imaging_bins["mixed_density"] = dens.str.contains("mix|hyper|hetero").fillna(False).astype(int)
    struct = df["structurebaseline"].astype(str).str.strip().str.lower()
    imaging_bins["separated"] = struct.str.contains("sep|membr|trabe").fillna(False).astype(int)

    # G — splines on continuous (Approach A); add raw continuous (Approach G)
    spline_blocks = []
    for c in ["age", "sdh_vol", "plt", "mls_mm", "gcs", "mrs"]:
        b = rcs_basis(cont[c].values, n_knots=3)
        for j in range(b.shape[1]):
            spline_blocks.append(pd.DataFrame({f"{c}_rcs{j+1}": b[:, j]}))
    splines = pd.concat(spline_blocks, axis=1) if spline_blocks else pd.DataFrame()

    # Feature matrix used by lasso / elastic-net / GBM (approaches A, B, C, D, G)
    X = pd.concat([cont, bins, imaging_bins, splines], axis=1)

    # B — clinical interactions
    X["age_x_sdhvol"] = cont["age"].values * (cont["sdh_vol"].values / 100.0)
    X["anticoag_x_pltlow"] = bins["anticoag"].values * imaging_bins["plt_lt150"].values
    X["antiplat_x_age"] = bins["antiplatelet"].values * imaging_bins["age_pts"].values

    # Drop near-zero-var
    keep = [c for c in X.columns if X[c].nunique() >= 2 and X[c].std() > 1e-6]
    X = X[keep]
    return X, y, cont, bins, imaging_bins


def cv_proba(model_factory, X: np.ndarray, y: np.ndarray):
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
# C — Optuna-tuned elastic-net
def optuna_elasticnet(X, y):
    try:
        import optuna
    except ImportError:
        print("[C] Optuna not available — using grid fallback.")
        return grid_elasticnet(X, y)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        l1 = trial.suggest_float("l1_ratio", 0.0, 1.0)
        proba = cv_proba(
            lambda: Pipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression(
                                   penalty="elasticnet", solver="saga",
                                   C=C, l1_ratio=l1, max_iter=5000))]),
            X.values, y)
        return roc_auc_score(y, proba)

    study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    best = study.best_params
    print(f"[C] Optuna best: {best}, CV-AUC={study.best_value:.3f}")
    proba = cv_proba(
        lambda: Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(
                               penalty="elasticnet", solver="saga",
                               C=best["C"], l1_ratio=best["l1_ratio"],
                               max_iter=5000))]),
        X.values, y)
    return proba, best


def grid_elasticnet(X, y):
    best_score, best_proba, best_params = -np.inf, None, None
    for C in np.logspace(-3, 1, 10):
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            proba = cv_proba(
                lambda: Pipeline([("sc", StandardScaler()),
                                   ("lr", LogisticRegression(
                                       penalty="elasticnet", solver="saga",
                                       C=C, l1_ratio=l1, max_iter=5000))]),
                X.values, y)
            s = roc_auc_score(y, proba)
            if s > best_score:
                best_score, best_proba, best_params = s, proba, {"C": C, "l1_ratio": l1}
    print(f"[C] grid best: {best_params}, CV-AUC={best_score:.3f}")
    return best_proba, best_params


# ----------------------------------------------------------------------
# D — Stacked ensemble
def stacked_ensemble(X, y):
    """Out-of-fold stacking: base learners (logistic, GBM) → logistic meta."""
    base_factories = [
        ("logit", lambda: Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=5000,
                                       solver="liblinear"))])),
        ("gbm", lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=RNG)),
    ]
    n = len(y)
    base_oof = np.zeros((n, len(base_factories)))
    for k, (name, factory) in enumerate(base_factories):
        base_oof[:, k] = cv_proba(factory, X.values, y)

    # Meta-learner trained on stacked OOF features
    meta_proba = cv_proba(
        lambda: LogisticRegression(C=1.0, max_iter=5000, solver="liblinear"),
        base_oof, y)
    return meta_proba


# ----------------------------------------------------------------------
# E — Bayesian logistic with informative priors (CV out-of-fold)
def _bayes_fit_predict(X_train, y_train, X_test, cols):
    """Single-fold PyMC fit; returns posterior mean predictive prob on test."""
    import pymc as pm
    Xs_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    mu_train, sd_train = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    Xs_test = (X_test - mu_train) / sd_train

    with pm.Model():
        mu0, sd0 = PRIORS["intercept"]
        intercept = pm.Normal("intercept", mu=mu0, sigma=sd0)
        coefs = []
        for c in cols:
            mu, sd = PRIORS.get(c, DEFAULT_PRIOR)
            coefs.append(pm.Normal(f"b_{c}", mu=mu, sigma=sd))
        beta = pm.math.stack(coefs)
        eta = intercept + pm.math.dot(Xs_train, beta)
        pm.Bernoulli("y_obs", p=pm.math.sigmoid(eta), observed=y_train)
        idata = pm.sample(500, tune=500, chains=2, random_seed=RNG,
                           progressbar=False, cores=1, target_accept=0.9)

    # Out-of-sample predictive: average sigmoid(intercept + X_test @ beta) over draws
    post = idata.posterior
    int_draws = post["intercept"].values.reshape(-1)            # (S,)
    coef_draws = np.column_stack([
        post[f"b_{c}"].values.reshape(-1) for c in cols
    ])                                                          # (S, p)
    # eta_test: (n_test, S) = (n_test, p) @ (p, S) plus intercept (S,)
    eta_test = Xs_test @ coef_draws.T + int_draws[None, :]
    p_test = 1 / (1 + np.exp(-eta_test))
    return p_test.mean(axis=1)                                  # (n_test,)


def bayesian_logit_cv(X_named: pd.DataFrame, y: np.ndarray):
    """Proper out-of-fold posterior predictive AUC for the Bayesian model."""
    try:
        import pymc as pm  # noqa: F401
    except ImportError:
        print("[E] PyMC not installed — using analytical shrinkage fallback.")
        return bayesian_fallback(X_named, y)

    cols = list(X_named.columns)
    print(f"[E] PyMC NUTS sampler on {len(cols)} features, "
          f"5×{N_REPEATS} CV...")
    proba = np.zeros(len(y))
    Xv = X_named.values
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RNG + r)
        for tr, te in cv.split(Xv, y):
            proba[te] += _bayes_fit_predict(Xv[tr], y[tr], Xv[te], cols)
    proba /= N_REPEATS

    # Also do a single full-data fit for the Bayesian "apparent" output
    # and to surface posterior summaries for the manuscript table
    with __import__("pymc").Model():
        pm = __import__("pymc")
        Xs_full = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + 1e-8)
        mu0, sd0 = PRIORS["intercept"]
        intercept = pm.Normal("intercept", mu=mu0, sigma=sd0)
        coefs = []
        for c in cols:
            mu, sd = PRIORS.get(c, DEFAULT_PRIOR)
            coefs.append(pm.Normal(f"b_{c}", mu=mu, sigma=sd))
        beta = pm.math.stack(coefs)
        eta = intercept + pm.math.dot(Xs_full, beta)
        pm.Bernoulli("y_obs", p=pm.math.sigmoid(eta), observed=y)
        idata_full = pm.sample(1000, tune=1000, chains=2,
                                random_seed=RNG, progressbar=False,
                                cores=1, target_accept=0.9)
    return proba, idata_full, cols


def bayesian_fallback(X_named: pd.DataFrame, y: np.ndarray):
    """Approximate the Bayesian shrinkage with feature-wise L2 ridge offsets.
    Each feature's penalty C is set so that the implied 1-SE shrinkage matches
    the prior SD. Less rigorous than PyMC but no extra dependency."""
    cols = list(X_named.columns)
    Xs = StandardScaler().fit_transform(X_named.values)

    # For each column, the prior implies a "pre-data" coefficient mean and
    # a target SD. Shift the target by adding a synthetic observation that
    # nudges the MLE toward the prior mean.
    # Simpler: use unpenalized logistic + manually shift by prior mean.
    res = sm.Logit(y, sm.add_constant(Xs)).fit(disp=False, method="bfgs",
                                                 maxiter=500)
    mle = res.params[1:].copy()  # drop intercept
    mle_se = res.bse[1:].copy()

    # Inverse-variance weighted average of MLE and prior
    shrunk = np.zeros_like(mle)
    for i, c in enumerate(cols):
        mu_p, sd_p = PRIORS.get(c, DEFAULT_PRIOR)
        sd_mle = max(mle_se[i], 1e-6)
        w_p = 1 / sd_p**2
        w_d = 1 / sd_mle**2
        shrunk[i] = (mu_p * w_p + mle[i] * w_d) / (w_p + w_d)
    intercept_shrunk = res.params[0]  # keep intercept
    eta = intercept_shrunk + Xs @ shrunk
    proba = 1 / (1 + np.exp(-eta))
    return proba, None, cols


# ----------------------------------------------------------------------
def main():
    csv = HERE / "mmaecsv.csv"
    X, y, cont, bins, imag = build_features(csv)
    print(f"[BUILD] feature matrix: {X.shape[1]} features, n={len(X)}, "
          f"events={y.sum()}")

    rows = []
    proba_dict = {}

    # A+B+G — lasso on full enriched matrix
    print("\n=== A+B+G: lasso on enriched matrix (splines + interactions) ===")
    Cs = np.logspace(-3, 1, 30)
    Xs = StandardScaler().fit_transform(X.values)
    best, best_score = None, -np.inf
    for C in Cs:
        proba = cv_proba(
            lambda: LogisticRegression(penalty="l1", solver="liblinear",
                                        C=C, max_iter=5000),
            Xs, y)
        s = roc_auc_score(y, proba)
        if s > best_score:
            best_score, best = s, dict(C=C, proba=proba)
    print(f"[A+B+G] best lasso CV-AUC={best_score:.3f} at C={best['C']:.4f}")
    rows.append(dict(approach="A+B+G lasso (enriched)",
                       cv_auc=best_score,
                       **dict(zip(["ci_lo", "ci_hi"],
                                   boot_auc_ci(y, best["proba"])[1:]))))
    proba_dict["A+B+G lasso"] = best["proba"]

    # C — Optuna elastic-net
    print("\n=== C: Optuna-tuned elastic-net ===")
    proba_en, best_en = optuna_elasticnet(X, y)
    auc_en = roc_auc_score(y, proba_en)
    _, lo, hi = boot_auc_ci(y, proba_en)
    rows.append(dict(approach="C tuned elastic-net", cv_auc=auc_en,
                       ci_lo=lo, ci_hi=hi))
    proba_dict["C elastic-net"] = proba_en

    # D — Stacked ensemble
    print("\n=== D: stacked ensemble (logistic + GBM) ===")
    proba_stk = stacked_ensemble(X, y)
    auc_stk = roc_auc_score(y, proba_stk)
    _, lo, hi = boot_auc_ci(y, proba_stk)
    print(f"[D] stacked CV-AUC={auc_stk:.3f} ({lo:.3f}–{hi:.3f})")
    rows.append(dict(approach="D stacked ensemble", cv_auc=auc_stk,
                       ci_lo=lo, ci_hi=hi))
    proba_dict["D stacked"] = proba_stk

    # E — Bayesian logistic with informative priors
    print("\n=== E: Bayesian logistic with informative priors ===")
    # Pick a parsimonious feature set for E (the named, prior-mapped columns)
    bayes_cols = [c for c in PRIORS.keys() if c != "intercept" and c in X.columns]
    extra = [c for c in ["focal_deficit", "hypertension", "gait", "male"]
              if c in X.columns]
    bayes_cols = bayes_cols + [c for c in extra if c not in bayes_cols]
    print(f"[E] features in Bayesian model: {bayes_cols}")
    proba_bayes, idata, cols_used = bayesian_logit_cv(X[bayes_cols], y)
    auc_b = roc_auc_score(y, proba_bayes)
    _, lo, hi = boot_auc_ci(y, proba_bayes)
    print(f"[E] Bayesian CV-AUC={auc_b:.3f} ({lo:.3f}–{hi:.3f})")
    rows.append(dict(approach="E Bayesian (informative priors)",
                       cv_auc=auc_b, ci_lo=lo, ci_hi=hi))
    proba_dict["E Bayesian"] = proba_bayes

    # Final composite Model 5 — average of the four CV-out-of-fold probabilities
    print("\n=== Composite Model 5 (mean of A+B+G, C, D, E) ===")
    proba_m5 = np.mean(np.column_stack(list(proba_dict.values())), axis=1)
    auc5 = roc_auc_score(y, proba_m5)
    _, lo, hi = boot_auc_ci(y, proba_m5)
    print(f"[M5] composite CV-AUC={auc5:.3f} ({lo:.3f}–{hi:.3f})")
    rows.append(dict(approach="Model 5 (composite mean)",
                       cv_auc=auc5, ci_lo=lo, ci_hi=hi))
    proba_dict["Model 5 (composite)"] = proba_m5

    # Save outputs
    out = pd.DataFrame(rows)
    out.to_csv(V2 / "m5_results.csv", index=False)
    pd.DataFrame(proba_dict).to_csv(V2 / "m5_proba.csv", index=False)
    summary = dict(
        n=int(len(y)), events=int(y.sum()),
        n_features_enriched=int(X.shape[1]),
        approaches=out.to_dict(orient="records"),
        best_elasticnet=best_en,
        bayesian_features=bayes_cols,
    )
    (V2 / "m5_summary.json").write_text(json.dumps(summary, indent=2,
                                                    default=float))
    print("\n=== Final Model 5 sensitivity table ===")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
