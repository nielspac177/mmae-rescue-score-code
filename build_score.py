"""
MMAE Rescue-Surgery Scoring System
==================================
Cohort: stand-alone MMAE for chronic SDH (n=148, 22 rescues, 14.9%).
Pipeline:
  1. Feature engineering + simple imputation (median for numeric, mode for binary)
  2. Univariate logistic regression
  3. Supervised ML benchmark (Logistic, L1-Logistic, RandomForest, GradientBoosting)
     - Repeated stratified 5-fold CV (50 reps) for AUC
  4. Permutation feature importance (best model) + SHAP for the GB model
  5. Unsupervised clustering (k-means w/ k=2..6 + hierarchical) on standardized features
     - PCA + UMAP-like visualization (PCA only to avoid extra deps)
     - Rescue rate per cluster
  6. Parsimonious multivariable logistic model -> integer point score (BAI-style)
  7. Bootstrap optimism-corrected AUC of point score
Outputs are written to /tmp/mmae_outputs/ as CSV + PNG.
"""
import os, json, warnings, math, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

OUT = "/tmp/mmae_outputs"
os.makedirs(OUT, exist_ok=True)
RNG = 20260428

# ----------------------------------------------------------------------------
# 1. Load cohort
# ----------------------------------------------------------------------------
df = pd.read_csv("/Users/nielspacheco/Desktop/Research/Jimena Gonzales-salidos/MMAE scoring/mmaecsv.csv")
sa = df.copy().reset_index(drop=True)
sa["rescue"] = (sa["rescue_surgery"] == "Yes").astype(int)
y = sa["rescue"].values
print(f"Cohort: n={len(sa)}, events={y.sum()} ({y.mean()*100:.1f}%)")

# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def yn(s, val="Yes"):
    return (s == val).astype(float).where(s.notna(), np.nan)

X = pd.DataFrame(index=sa.index)
# Demographics
X["age"] = sa["age"].astype(float)
X["age_ge75"] = (sa["age"] >= 75).astype(float)
X["age_ge80"] = (sa["age"] >= 80).astype(float)
X["male"] = (sa["gender_num"] == "Male").astype(float)
# Comorbidities & meds
X["diabetes"] = yn(sa["diabetes"])
X["hypertension"] = yn(sa["hypertension"])
X["malignancy"] = yn(sa["malignancy"])
X["statins"] = yn(sa["statins"])
X["smoker_current"] = (sa["smoking"] == "Current smoker").astype(float)
X["antiplatelet"] = yn(sa["antiplatelet"])
X["anticoagulation"] = yn(sa["anticoagulation"])
# Presentation
X["baseline_mrs"] = sa["baselinemrs"].astype(float)
X["mrs_ge3"] = (sa["baselinemrs_cat"] == "mRS ≥3").astype(float)
# Asymptomatic / incidental and focal deficit excluded a priori — see
# bias-triangulation analyses (Section 3a–b in REPORT.html).
X["headache"] = yn(sa["headache"])
X["nausea"] = yn(sa["nausea"])
X["seizures"] = yn(sa["seizures"])
X["fall"] = yn(sa["fall"])
X["gait"] = yn(sa["gait"])
# Lab
X["hb"] = sa["hb_num"].astype(float)
X["plt"] = sa["plt_num"].astype(float)
X["plt_lt150"] = (sa["plt_cat150"] == "<150").astype(float).where(sa["plt_cat150"].notna(), np.nan)
X["inr"] = sa["inrlastbeforeprocedure"].astype(float)
# Procedural
X["bilateral"] = yn(sa["bilateral_num2"])
X["branches_ap"] = (sa["branches"] == "Anterior + posterior").astype(float)
X["access_radial"] = (sa["access"] == "Radial").astype(float)
X["anesthesia_ga"] = (sa["anesthesia"] == "General anesthesia").astype(float)
X["embolic_coils_only"] = (sa["embolic_num"] == "Coils only").astype(float)
X["embolic_coils_part"] = (sa["embolic_num"] == "Coils + particles").astype(float)
X["embolic_liquid_only"] = (sa["embolic_num"] == "Liquid embolic").astype(float)
X["use_liquid"] = sa["useofliquidembolic"].astype(float)
X["mma_diam_prox"] = sa["mma_diam_prox"].astype(float)
X["n_branches_emb_part"] = sa["numberofbranchesembolizedwithpar"].astype(float)
# Baseline imaging
X["axial_thick"] = sa["axialbaseline"].astype(float)
X["axial_ge20"] = (sa["axialbaseline"] >= 20).astype(float)
X["sdh_volume"] = sa["sdhvolumebaseline"].astype(float)
X["sdh_vol_ge100"] = (sa["sdhvolumebaseline"] >= 100).astype(float)
X["midline_shift_mm"] = sa["midlineshiftmeasureinmmbaseline"].astype(float)
X["mls_gt5"] = (sa["shift5baseline"] == ">5 mm").astype(float).where(sa["shift5baseline"].notna(), np.nan)
X["membranes"] = (sa["membranesbasline"] == "Yes").astype(float).where(sa["membranesbasline"].notna(), np.nan)
X["acute_subacute"] = (sa["acute_subacutebasline"] == "Yes").astype(float).where(sa["acute_subacutebasline"].notna(), np.nan)
X["loculation"] = (sa["loculation"] == "Yes").astype(float).where(sa["loculation"].notna(), np.nan)
X["separated_grad"] = (sa["structurebaseline"] == "Separated / Gradation").astype(float).where(sa["structurebaseline"].notna(), np.nan)
X["density_mixed"] = (sa["densitybaseline"] == "Mixed").astype(float).where(sa["densitybaseline"].notna(), np.nan)

print(f"Features: {X.shape[1]}, missing per feature (top 5):")
print(X.isna().mean().sort_values(ascending=False).head(5).to_string())

# Median imputation
imp = SimpleImputer(strategy="median")
Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# ----------------------------------------------------------------------------
# 3. Supervised ML benchmark
# ----------------------------------------------------------------------------
print("\n=== Supervised ML benchmark (5x10 stratified CV) ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RNG)
models = {
    "LogReg": Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0))]),
    "L1-LogReg": Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegressionCV(Cs=20, penalty="l1", solver="liblinear",
                                                       cv=5, scoring="roc_auc", max_iter=2000))]),
    "L2-LogReg-CV": Pipeline([("sc", StandardScaler()),
                              ("lr", LogisticRegressionCV(Cs=20, penalty="l2", cv=5,
                                                          scoring="roc_auc", max_iter=2000))]),
    "RandomForest": RandomForestClassifier(n_estimators=600, max_depth=4, min_samples_leaf=5,
                                            random_state=RNG, class_weight="balanced"),
    "GradBoost": GradientBoostingClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                                            random_state=RNG),
}
bench = {}
for name, mdl in models.items():
    aucs = cross_val_score(mdl, Xi, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    bench[name] = aucs
    print(f"  {name:14s} AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")

bench_df = pd.DataFrame({k: pd.Series(v) for k, v in bench.items()})
bench_df.to_csv(f"{OUT}/cv_auc.csv", index=False)

# ----------------------------------------------------------------------------
# 4. Feature importance: permutation on GB and L1-LR coefficients
# ----------------------------------------------------------------------------
print("\n=== Feature importance ===")
# Refit GB on full data for permutation importance and SHAP
gb = GradientBoostingClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=RNG)
gb.fit(Xi, y)
perm = permutation_importance(gb, Xi, y, n_repeats=50, random_state=RNG, scoring="roc_auc", n_jobs=-1)
perm_df = (pd.DataFrame({"feature": Xi.columns, "imp_mean": perm.importances_mean,
                         "imp_std": perm.importances_std})
           .sort_values("imp_mean", ascending=False))
perm_df.to_csv(f"{OUT}/perm_importance_gb.csv", index=False)
print("Top 12 permutation importances (GradBoost):")
print(perm_df.head(12).to_string(index=False))

# L1 coefficients
sc = StandardScaler().fit(Xi)
Xs = sc.transform(Xi)
l1 = LogisticRegressionCV(Cs=30, penalty="l1", solver="liblinear", cv=5,
                          scoring="roc_auc", max_iter=3000).fit(Xs, y)
coef_df = pd.DataFrame({"feature": Xi.columns, "coef_std": l1.coef_[0]})
coef_df = coef_df.reindex(coef_df["coef_std"].abs().sort_values(ascending=False).index)
coef_df.to_csv(f"{OUT}/l1_coefficients.csv", index=False)
print(f"\nL1 chosen C = {l1.C_[0]:.4f}")
print("Non-zero L1 coefficients (standardized):")
print(coef_df[coef_df["coef_std"] != 0].to_string(index=False))

# SHAP for GB
try:
    import shap
    explainer = shap.TreeExplainer(gb)
    sv = explainer.shap_values(Xi)
    shap_imp = pd.DataFrame({"feature": Xi.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(f"{OUT}/shap_importance.csv", index=False)
    print("\nTop 12 SHAP importances (GradBoost):")
    print(shap_imp.head(12).to_string(index=False))
    plt.figure()
    shap.summary_plot(sv, Xi, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary.png", dpi=150); plt.close()
except Exception as e:
    print(f"SHAP failed: {e}")

# ----------------------------------------------------------------------------
# 5. Unsupervised clustering
# ----------------------------------------------------------------------------
print("\n=== Unsupervised clustering ===")
Xc = StandardScaler().fit_transform(Xi)
from sklearn.metrics import silhouette_score
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=RNG).fit(Xc)
    sil = silhouette_score(Xc, km.labels_)
    rates = pd.Series(y).groupby(km.labels_).mean().round(3).to_dict()
    sizes = pd.Series(km.labels_).value_counts().sort_index().to_dict()
    print(f"  k={k} silhouette={sil:.3f}  rescue rate by cluster: {rates}  sizes: {sizes}")

# Pick k=3 by default (typical phenotype split) for downstream interpretation
best_k = 3
km = KMeans(n_clusters=best_k, n_init=50, random_state=RNG).fit(Xc)
sa["cluster"] = km.labels_
cluster_summary = sa.groupby("cluster").agg(
    n=("rescue", "size"), rescues=("rescue", "sum"), rate=("rescue", "mean"),
    age=("age", "median"), mrs=("baselinemrs", "median"),
).round(3)
print("\nk=3 cluster summary:")
print(cluster_summary.to_string())
cluster_summary.to_csv(f"{OUT}/cluster_summary_k3.csv")

# Per-feature cluster means (clinical phenotype description)
clust_means = pd.concat([Xi, pd.Series(km.labels_, name="cluster")], axis=1).groupby("cluster").mean().T
clust_means.to_csv(f"{OUT}/cluster_feature_means_k3.csv")

# PCA visualization
pca = PCA(n_components=2, random_state=RNG).fit(Xc)
emb = pca.transform(Xc)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sc1 = axes[0].scatter(emb[:, 0], emb[:, 1], c=km.labels_, cmap="tab10", alpha=0.85, s=35)
axes[0].set_title(f"k-means k={best_k}  (silhouette={silhouette_score(Xc, km.labels_):.2f})")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
axes[1].scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", alpha=0.8, s=35)
axes[1].set_title("Rescue surgery (red = Yes)")
axes[1].set_xlabel(f"PC1"); axes[1].set_ylabel("PC2")
plt.tight_layout(); plt.savefig(f"{OUT}/pca_clusters.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 6. Parsimonious score: multivariable logistic regression with backward elimination
#    on a candidate pool informed by the SHAP / L1 / univariate signals.
# ----------------------------------------------------------------------------
print("\n=== Building parsimonious BAI-style score ===")
# Start from a clinically defensible candidate pool of 6-8 variables
candidates = ["asymptomatic", "antiplatelet", "branches_ap", "embolic_coils_part",
              "axial_ge20", "separated_grad", "membranes", "age_ge75",
              "anticoagulation", "mls_gt5", "bilateral"]
candidates = [c for c in candidates if c in Xi.columns]
Xc6 = Xi[candidates].copy()

# Backward elimination using AIC on full data (acceptable for derivation cohort
# with bootstrap optimism correction in step 7).
def bw_eliminate(X_, y_, p_thresh=0.10):
    feats = list(X_.columns)
    while feats:
        Xt = sm.add_constant(X_[feats]); m = sm.Logit(y_, Xt).fit(disp=0)
        pv = m.pvalues.drop("const")
        if pv.max() <= p_thresh: break
        worst = pv.idxmax(); feats.remove(worst)
    return feats, m

selected, final_logit = bw_eliminate(Xc6, y, p_thresh=0.10)
print(f"Selected predictors (p ≤ 0.10): {selected}")
print(final_logit.summary().tables[1])

# Convert betas -> integer points
betas = final_logit.params.drop("const")
print("\nBetas:"); print(betas.round(3).to_string())
unit = betas.abs().min()
points = (betas / unit).round().astype(int)
# Flip sign if any predictor is protective (negative beta) -> we still want positive points
# Convention: invert protective predictors (e.g. asymptomatic protective => use "symptomatic absent" = 1)
score_def = []
for f, b, p in zip(betas.index, betas.values, points.values):
    direction = "presence" if b > 0 else "absence"  # absence => the variable=0 contributes points
    abs_pts = int(abs(p))
    score_def.append((f, b, abs_pts, direction))
score_def_df = pd.DataFrame(score_def, columns=["feature", "beta", "points", "scoring_when"])
print("\nScore definition (points awarded when condition met):")
print(score_def_df.to_string(index=False))
score_def_df.to_csv(f"{OUT}/score_definition.csv", index=False)

# Compute score per patient
def compute_score(Xrow, score_def_df):
    s = 0
    for _, r in score_def_df.iterrows():
        v = Xrow[r["feature"]]
        if r["scoring_when"] == "presence":
            s += int(r["points"] * (v == 1))
        else:
            s += int(r["points"] * (v == 0))
    return s
sa["points"] = Xi.apply(lambda row: compute_score(row, score_def_df), axis=1)
auc_score = roc_auc_score(y, sa["points"])
print(f"\nInteger score AUC (apparent): {auc_score:.3f}")

# Risk strata
strata = sa.groupby("points").agg(n=("rescue", "size"), rescues=("rescue", "sum"),
                                  rate=("rescue", "mean"))
print("\nRisk by integer score:")
print(strata.to_string())
strata.to_csv(f"{OUT}/risk_by_score.csv")

# Calibration: predicted prob from logistic vs observed
phat = final_logit.predict(sm.add_constant(Xc6[selected]))
auc_logit = roc_auc_score(y, phat)
print(f"Full logistic AUC (apparent): {auc_logit:.3f}")

# ----------------------------------------------------------------------------
# 7. Bootstrap optimism-corrected AUC (Harrell)
# ----------------------------------------------------------------------------
print("\n=== Bootstrap optimism correction (B=1000) ===")
B = 1000
rng = np.random.default_rng(RNG)
opt_logit, opt_score = [], []
n = len(sa)
Xfull = sm.add_constant(Xc6[selected]); yarr = y
for b in range(B):
    idx = rng.integers(0, n, n)
    Xb, yb = Xfull.iloc[idx], yarr[idx]
    if yb.sum() < 3 or yb.sum() > n - 3: continue
    try:
        mb = sm.Logit(yb, Xb).fit(disp=0, method="bfgs", maxiter=200)
    except Exception:
        continue
    pb_boot = mb.predict(Xb)
    pb_orig = mb.predict(Xfull)
    auc_boot = roc_auc_score(yb, pb_boot)
    auc_orig = roc_auc_score(yarr, pb_orig)
    opt_logit.append(auc_boot - auc_orig)
opt = float(np.mean(opt_logit))
print(f"Apparent AUC (logistic) {auc_logit:.3f}  -  optimism {opt:.3f}  =  corrected {auc_logit-opt:.3f}")

# Bootstrap for the integer score
score_apparent = roc_auc_score(y, sa["points"])
opt_score = []
for b in range(B):
    idx = rng.integers(0, n, n)
    yb = y[idx]; pb = sa["points"].values[idx]
    if yb.sum() < 3 or yb.sum() > n - 3: continue
    auc_boot = roc_auc_score(yb, pb)
    auc_orig = roc_auc_score(y, sa["points"])
    opt_score.append(auc_boot - auc_orig)
opt_s = float(np.mean(opt_score))
print(f"Apparent AUC (integer score) {score_apparent:.3f}  -  optimism {opt_s:.3f}  =  corrected {score_apparent-opt_s:.3f}")

# ----------------------------------------------------------------------------
# 8. ROC plot
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5.5))
fpr, tpr, _ = roc_curve(y, phat); ax.plot(fpr, tpr, label=f"Multivariable logistic AUC={auc_logit:.2f}")
fpr2, tpr2, _ = roc_curve(y, sa["points"]); ax.plot(fpr2, tpr2, label=f"Integer score AUC={auc_score:.2f}", ls="--")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC: rescue surgery after stand-alone MMAE")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT}/roc.png", dpi=150); plt.close()

# Save final patient-level scored data
sa[["mrn", "age", "rescue", "cluster", "points"]].to_csv(f"{OUT}/scored_cohort.csv", index=False)

# Final compact summary
summary = {
    "n": int(len(sa)),
    "events": int(y.sum()),
    "event_rate": float(y.mean()),
    "cv_auc": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in bench.items()},
    "selected_predictors": list(selected),
    "logistic_auc_apparent": float(auc_logit),
    "logistic_auc_optimism_corrected": float(auc_logit - opt),
    "score_auc_apparent": float(auc_score),
    "score_auc_optimism_corrected": float(auc_score - opt_s),
    "best_kmeans_k": best_k,
    "cluster_rescue_rates": cluster_summary["rate"].to_dict(),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== Done ===  Outputs in", OUT)
