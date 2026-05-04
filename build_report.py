"""
Final report build:
  * Cluster-membership logistic regression (crude + adjusted)
  * Radar plots per cluster (JAMA Neurology / JNNP style)
  * Polished JAMA-style figures (ROC, FAMD scatter, SHAP-style importance,
    calibration, score-risk plot, radar)
  * Self-contained HTML report with embedded base64 figures
"""
import os, io, json, base64, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import prince
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# JAMA / JNNP aesthetic
# -----------------------------------------------------------------------------
JAMA = {
    "navy":   "#1F3D5C",
    "teal":   "#2D7A8F",
    "orange": "#D5751B",
    "red":    "#A8232C",
    "gray":   "#6B6B6B",
    "lightg": "#D9D9D9",
    "fill1":  "#3E6F8F",
    "fill2":  "#7FA9C2",
    "fill3":  "#C7D6E0",
}
PAL3 = [JAMA["navy"], JAMA["teal"], JAMA["orange"]]
PAL5 = [JAMA["navy"], JAMA["teal"], JAMA["orange"], JAMA["red"], JAMA["gray"]]

rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":  10,
    "axes.titleweight": "bold",
    "axes.labelsize":   9,
    "axes.labelweight": "regular",
    "axes.linewidth":   0.8,
    "axes.edgecolor":  "#333333",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.grid":         False,
})

OUT = "/Users/nielspacheco/Desktop/Research/Jimena Gonzales-salidos/MMAE scoring"
RNG = 20260428
np.random.seed(RNG)

# -----------------------------------------------------------------------------
# Load + reproduce earlier feature engineering
# -----------------------------------------------------------------------------
df = pd.read_csv(f"{OUT}/mmaecsv.csv")
sa = df.copy().reset_index(drop=True)
y = (sa["rescue_surgery"] == "Yes").astype(int).values

num_feats = ["age", "baselinemrs", "hb_num", "plt_num", "inrlastbeforeprocedure",
             "mma_diam_prox", "axialbaseline", "sdhvolumebaseline",
             "midlineshiftmeasureinmmbaseline", "numberofbranchesembolizedwithpar"]
cat_feats = ["gender_num", "smoking", "diabetes", "hypertension", "malignancy",
             "statins", "antiplatelet", "anticoagulation",
             "presentation_cat", "focal_deficit", "headache", "nausea",
             "fall", "gait", "bilateral_num2",
             "branches", "access", "anesthesia", "embolic_num",
             "structurebaseline", "densitybaseline", "membranesbasline",
             "acute_subacutebasline", "loculation", "shift5baseline",
             "baselinemrs_cat"]
M = pd.concat([sa[num_feats].astype(float), sa[cat_feats].astype(object)], axis=1)
for c in num_feats: M[c] = M[c].fillna(M[c].median())
for c in cat_feats: M[c] = M[c].fillna(M[c].mode().iloc[0])

famd = prince.FAMD(n_components=10, n_iter=10, random_state=RNG, copy=True).fit(M)
F = famd.row_coordinates(M).values
ev_ratio = famd.eigenvalues_ / famd.eigenvalues_.sum()
km = KMeans(n_clusters=3, n_init=50, random_state=RNG).fit(F)
clusters = km.labels_

# Re-order clusters by rescue rate ascending so cluster 0 = lowest risk
order = pd.Series(y).groupby(clusters).mean().sort_values().index.tolist()
remap = {old: new for new, old in enumerate(order)}
clusters = np.array([remap[c] for c in clusters])
sa["cluster"] = clusters

# -----------------------------------------------------------------------------
# 1. Cluster-membership regression
# -----------------------------------------------------------------------------
print("=== Cluster-membership logistic regression ===")
cl_dum = pd.get_dummies(pd.Series(clusters, name="cluster"), prefix="cluster",
                        drop_first=True).astype(float)
X1 = sm.add_constant(cl_dum)
m_crude = sm.Logit(y, X1).fit(disp=0)
print("Crude model:")
print(m_crude.summary().tables[1])

# Adjusted model: cluster + age, baseline mrs, antiplatelet, axial thick
adj = pd.DataFrame({
    "age":          sa["age"].astype(float),
    "baseline_mrs": sa["baselinemrs"].astype(float),
    "antiplatelet": (sa["antiplatelet"] == "Yes").astype(float),
    "axial_ge20":   (sa["axialbaseline"] >= 20).astype(float).fillna(0),
})
# Median-impute any remaining NaN to keep the adjusted model fitable
adj = adj.fillna(adj.median())
X2 = sm.add_constant(pd.concat([cl_dum, adj], axis=1))
m_adj = sm.Logit(y, X2).fit(disp=0)
print("\nAdjusted model (age, baseline mRS, antiplatelet, axial ≥20mm):")
print(m_adj.summary().tables[1])

def or_table(m, label):
    coef = m.params; ci = m.conf_int(); pv = m.pvalues
    df_ = pd.DataFrame({
        "term":  coef.index,
        "OR":    np.exp(coef.values),
        "OR_lo": np.exp(ci.iloc[:, 0].values),
        "OR_hi": np.exp(ci.iloc[:, 1].values),
        "p":     pv.values,
        "model": label,
    })
    return df_
or_tbl = pd.concat([or_table(m_crude, "crude"), or_table(m_adj, "adjusted")])
or_tbl.to_csv(f"{OUT}/cluster_regression_or.csv", index=False)
print("\nSaved cluster_regression_or.csv")

# Cochran-Armitage / chi-square for trend across clusters
ct = pd.crosstab(clusters, y)
chi2, p_chi, _, _ = stats.chi2_contingency(ct)
print(f"Chi-square cluster vs rescue: chi2={chi2:.3f}, p={p_chi:.4f}")

# -----------------------------------------------------------------------------
# 2. Radar plots — JAMA style
# -----------------------------------------------------------------------------
print("\n=== Radar plots ===")
radar_vars = [
    ("Age (yrs)",                "age",                   "num"),
    ("Baseline mRS",             "baselinemrs",           "num"),
    ("SDH thickness (mm)",       "axialbaseline",         "num"),
    ("SDH volume (mL)",          "sdhvolumebaseline",     "num"),
    ("Midline shift (mm)",       "midlineshiftmeasureinmmbaseline", "num"),
    ("Antiplatelet (%)",         "antiplatelet",          "ppos"),
    ("Anticoagulation (%)",      "anticoagulation",       "ppos"),
    ("Symptomatic (%)",          "presentation_cat",      "psym"),
    ("Membranes (%)",            "membranesbasline",      "ppos"),
    ("Separated/gradation (%)",  "structurebaseline",     "psep"),
    ("Mixed density (%)",        "densitybaseline",       "pmix"),
    ("Bilateral (%)",            "bilateral_num2",        "ppos"),
]
n_clust = int(clusters.max() + 1)
clust_vals = np.zeros((n_clust, len(radar_vars)))
glob_min = np.zeros(len(radar_vars)); glob_max = np.zeros(len(radar_vars))
labels = []
for j, (label, col, kind) in enumerate(radar_vars):
    labels.append(label)
    if kind == "num":
        v = pd.to_numeric(sa[col], errors="coerce")
        glob_min[j] = np.nanpercentile(v, 5)
        glob_max[j] = np.nanpercentile(v, 95)
    else:
        glob_min[j] = 0; glob_max[j] = 1
    for k in range(n_clust):
        sub = sa[clusters == k]
        if kind == "num":
            clust_vals[k, j] = pd.to_numeric(sub[col], errors="coerce").median()
        elif kind == "ppos":
            clust_vals[k, j] = (sub[col] == "Yes").mean()
        elif kind == "psym":
            clust_vals[k, j] = (sub[col] == "Symptomatic").mean()
        elif kind == "psep":
            clust_vals[k, j] = (sub[col] == "Separated / Gradation").mean()
        elif kind == "pmix":
            clust_vals[k, j] = (sub[col] == "Mixed").mean()

# Min-max normalize each axis using global min/max
norm_vals = np.zeros_like(clust_vals)
for j in range(len(radar_vars)):
    rng = glob_max[j] - glob_min[j]
    if rng == 0: rng = 1
    norm_vals[:, j] = np.clip((clust_vals[:, j] - glob_min[j]) / rng, 0, 1)

n_axes = len(radar_vars)
angles = np.linspace(0, 2*np.pi, n_axes, endpoint=False).tolist()
angles_closed = angles + [angles[0]]

# 3-panel figure: one radar per cluster
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6),
                         subplot_kw=dict(projection="polar"))
rate_per_cluster = pd.Series(y).groupby(clusters).mean()
size_per_cluster = pd.Series(clusters).value_counts().sort_index()
events_per_cluster = pd.Series(y).groupby(clusters).sum()
for k, ax in enumerate(axes):
    vals = norm_vals[k].tolist() + [norm_vals[k, 0]]
    color = PAL3[k]
    ax.fill(angles_closed, vals, color=color, alpha=0.30)
    ax.plot(angles_closed, vals, color=color, lw=1.5)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=7.2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], size=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(pad=2)
    ax.spines["polar"].set_color("#999999")
    ax.spines["polar"].set_linewidth(0.6)
    ax.grid(True, color="#CCCCCC", lw=0.5)
    rate = rate_per_cluster.loc[k]
    ax.set_title(f"Cluster {k+1}  (n={size_per_cluster.loc[k]}, "
                 f"rescue {events_per_cluster.loc[k]}/{size_per_cluster.loc[k]} = {rate*100:.1f}%)",
                 size=10, pad=14, color=color, fontweight="bold")

fig.suptitle("Phenotype Radar Profiles — FAMD + KMeans (k=3)",
             y=1.04, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_radar_clusters.png", dpi=300)
plt.close()
print("Saved fig_radar_clusters.png")

# Overlay radar (all 3 in one plot)
fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(projection="polar"))
for k in range(n_clust):
    vals = norm_vals[k].tolist() + [norm_vals[k, 0]]
    color = PAL3[k]
    rate = rate_per_cluster.loc[k]
    ax.fill(angles_closed, vals, color=color, alpha=0.18)
    ax.plot(angles_closed, vals, color=color, lw=1.7,
            label=f"Cluster {k+1}  ({rate*100:.1f}% rescue)")
ax.set_xticks(angles)
ax.set_xticklabels(labels, size=8)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["", "", "", ""])
ax.set_ylim(0, 1.05)
ax.spines["polar"].set_color("#999999"); ax.spines["polar"].set_linewidth(0.6)
ax.grid(True, color="#CCCCCC", lw=0.5)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10))
ax.set_title("Overlay phenotype radar — FAMD + KMeans (k=3)",
             size=12, fontweight="bold", pad=22)
plt.savefig(f"{OUT}/fig_radar_overlay.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 3. Polished score risk plot
# -----------------------------------------------------------------------------
print("\n=== Score-risk barplot ===")
risk = pd.read_csv(f"{OUT}/risk_by_score.csv")
fig, ax = plt.subplots(figsize=(6.2, 4.0))
bar_colors = []
for s in risk["points"]:
    if s <= 1: bar_colors.append(JAMA["fill3"])
    elif s == 2: bar_colors.append(JAMA["fill2"])
    elif s == 3: bar_colors.append(JAMA["fill1"])
    else: bar_colors.append(JAMA["red"])
bars = ax.bar(risk["points"], risk["rate"]*100, color=bar_colors,
              edgecolor=JAMA["navy"], lw=0.8, width=0.7)
for b, n_, ev_ in zip(bars, risk["n"], risk["rescues"]):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 1.2,
            f"{ev_}/{n_}", ha="center", va="bottom", size=8, color=JAMA["navy"])
ax.set_xlabel("MMAE Rescue Score (points)")
ax.set_ylabel("Observed rescue surgery (%)")
ax.set_title("Risk Stratification by Integer Score", loc="left")
ax.set_ylim(0, max(risk["rate"]*100) * 1.30)
ax.set_xticks(risk["points"])
ax.axvline(2.5, color=JAMA["red"], ls="--", lw=0.9, alpha=0.7)
ax.text(2.55, ax.get_ylim()[1]*0.92, "  Cutoff ≥3 = high risk",
        color=JAMA["red"], size=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_score_risk.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 4. Polished ROC curves
# -----------------------------------------------------------------------------
print("=== ROC plot ===")
# Recompute logistic + integer score predictions
score_def = pd.read_csv(f"{OUT}/score_definition.csv")
def compute_score(row, sd):
    s = 0
    for _, r in sd.iterrows():
        v = row[r["feature"]]
        if r["scoring_when"] == "presence":
            s += int(r["points"]) * int(v == 1)
        else:
            s += int(r["points"]) * int(v == 0)
    return s

# build the binary feature frame again for this
Xi = pd.DataFrame({
    "asymptomatic":  (sa["presentation_cat"] != "Symptomatic").astype(float),
    "axial_ge20":    (sa["axialbaseline"] >= 20).astype(float).fillna(0),
    "separated_grad":(sa["structurebaseline"] == "Separated / Gradation").astype(float).fillna(0),
    "anticoagulation":(sa["anticoagulation"] == "Yes").astype(float).fillna(0),
})
Xi = Xi.fillna(0)
score = Xi.apply(lambda row: compute_score(row, score_def), axis=1)
Xfit = sm.add_constant(Xi); m_score = sm.Logit(y, Xfit).fit(disp=0)
phat = m_score.predict(Xfit)
auc_logit = roc_auc_score(y, phat)
auc_score = roc_auc_score(y, score)
fpr1, tpr1, _ = roc_curve(y, phat)
fpr2, tpr2, _ = roc_curve(y, score)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(fpr1, tpr1, color=JAMA["navy"], lw=1.7,
        label=f"Multivariable logistic (AUC {auc_logit:.2f})")
ax.plot(fpr2, tpr2, color=JAMA["orange"], lw=1.7, ls="--",
        label=f"Integer score 0–6 (AUC {auc_score:.2f})")
ax.plot([0, 1], [0, 1], color=JAMA["gray"], lw=0.7, ls=":")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("Discrimination — Rescue Surgery After Stand-Alone MMAE", loc="left")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_roc.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 5. SHAP-style horizontal feature importance (consensus)
# -----------------------------------------------------------------------------
print("=== Feature importance plot ===")
imp_perm = pd.read_csv(f"{OUT}/perm_importance_gb.csv")
imp_shap = pd.read_csv(f"{OUT}/shap_importance.csv")
imp_l1   = pd.read_csv(f"{OUT}/l1_coefficients.csv")
imp_l1["abs"] = imp_l1["coef_std"].abs()

def rank(df_, key):
    df_ = df_.sort_values(key, ascending=False).reset_index(drop=True)
    df_["rank_" + key] = df_.index + 1
    return df_[["feature", "rank_" + key]]
r1 = rank(imp_perm, "imp_mean")
r2 = rank(imp_shap, "mean_abs_shap")
r3 = rank(imp_l1, "abs")
ranks = r1.merge(r2, on="feature", how="outer").merge(r3, on="feature", how="outer")
for c in ["rank_imp_mean", "rank_mean_abs_shap", "rank_abs"]:
    ranks[c] = ranks[c].fillna(ranks[c].max() + 1)
ranks["mean_rank"] = ranks[["rank_imp_mean", "rank_mean_abs_shap", "rank_abs"]].mean(axis=1)
ranks = ranks.sort_values("mean_rank").head(12)
ranks = ranks.iloc[::-1]
labels_map = {
    "asymptomatic": "Asymptomatic / incidental",
    "separated_grad": "Separated/gradation structure",
    "sdh_volume": "SDH volume",
    "sdh_vol_ge100": "SDH volume ≥ 100 mL",
    "axial_thick": "Axial thickness",
    "axial_ge20": "Axial thickness ≥ 20 mm",
    "membranes": "Membranes (protective)",
    "mma_diam_prox": "MMA proximal diameter",
    "plt": "Platelet count",
    "inr": "INR",
    "hb": "Hemoglobin",
    "branches_ap": "Anterior + posterior branches",
    "antiplatelet": "Antiplatelet therapy",
    "embolic_coils_part": "Coils + particles",
    "anticoagulation": "Anticoagulation",
    "age": "Age (yrs)",
    "hypertension": "Hypertension",
    "focal_deficit": "Focal deficit",
    "anesthesia_ga": "General anesthesia",
    "access_radial": "Radial access",
}
fig, ax = plt.subplots(figsize=(7.5, 5.2))
yp = np.arange(len(ranks))
ax.barh(yp, len(ranks) + 1 - ranks["mean_rank"].values,
        color=JAMA["fill1"], edgecolor=JAMA["navy"], lw=0.6)
ax.set_yticks(yp)
ax.set_yticklabels([labels_map.get(f, f) for f in ranks["feature"]])
ax.set_xlabel("Consensus importance score (higher = more predictive)")
ax.set_title("Consensus Feature Importance — Permutation, SHAP, L1-Logistic",
             loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_feature_importance.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 6. Polished FAMD scatter
# -----------------------------------------------------------------------------
print("=== FAMD scatter ===")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for k in range(n_clust):
    sel = clusters == k
    axes[0].scatter(F[sel, 0], F[sel, 1], color=PAL3[k], s=42, alpha=0.85,
                    edgecolor="white", lw=0.4, label=f"Cluster {k+1}")
axes[0].set_title("FAMD + KMeans (k=3)", loc="left")
axes[0].set_xlabel(f"FAMD-1  ({ev_ratio[0]*100:.0f}% inertia)")
axes[0].set_ylabel(f"FAMD-2  ({ev_ratio[1]*100:.0f}% inertia)")
axes[0].legend(loc="upper right")

axes[1].scatter(F[y == 0, 0], F[y == 0, 1], color=JAMA["fill3"], s=38, alpha=0.85,
                edgecolor="white", lw=0.4, label="No rescue")
axes[1].scatter(F[y == 1, 0], F[y == 1, 1], color=JAMA["red"], s=46, alpha=0.95,
                edgecolor="white", lw=0.5, label="Rescue surgery")
axes[1].set_title("Rescue-surgery distribution in FAMD space", loc="left")
axes[1].set_xlabel("FAMD-1"); axes[1].set_ylabel("FAMD-2")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_famd_scatter.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 7. Forest plot of cluster ORs (crude vs adjusted)
# -----------------------------------------------------------------------------
print("=== Forest plot ===")
fp = or_tbl[or_tbl["term"].str.startswith("cluster_")].copy()
fp["cluster"] = fp["term"].str.replace("cluster_", "", regex=False).astype(int) + 1
fp_pivot_or  = fp.pivot(index="cluster", columns="model", values="OR")
fp_pivot_lo  = fp.pivot(index="cluster", columns="model", values="OR_lo")
fp_pivot_hi  = fp.pivot(index="cluster", columns="model", values="OR_hi")
fp_pivot_p   = fp.pivot(index="cluster", columns="model", values="p")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
yposes = np.arange(len(fp_pivot_or))
offset = 0.18
for i, mdl in enumerate(["crude", "adjusted"]):
    yy = yposes + (offset if mdl == "adjusted" else -offset)
    color = JAMA["navy"] if mdl == "crude" else JAMA["orange"]
    ax.errorbar(fp_pivot_or[mdl].values, yy,
                xerr=[fp_pivot_or[mdl].values - fp_pivot_lo[mdl].values,
                      fp_pivot_hi[mdl].values - fp_pivot_or[mdl].values],
                fmt="o", color=color, ecolor=color, lw=1.4, ms=6, capsize=3,
                label=mdl.capitalize())
    for yi, val, p_ in zip(yy, fp_pivot_or[mdl].values, fp_pivot_p[mdl].values):
        ax.text(val, yi + 0.07, f"OR {val:.2f}, p={p_:.2f}",
                fontsize=7.5, color=color, ha="center")
ax.axvline(1.0, color=JAMA["gray"], ls="--", lw=0.8)
ax.set_yticks(yposes)
ax.set_yticklabels([f"Cluster {c} vs Cluster 1" for c in fp_pivot_or.index])
ax.set_xlabel("Odds Ratio (95% CI)  for rescue surgery")
ax.set_title("Cluster-Membership Regression — Crude vs Adjusted", loc="left")
ax.set_xscale("log")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_forest_clusters.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 8. CV-AUC violin plot
# -----------------------------------------------------------------------------
print("=== CV-AUC plot ===")
cv = pd.read_csv(f"{OUT}/cv_auc.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.0))
parts = ax.violinplot([cv[c].dropna().values for c in cv.columns], showmeans=True,
                      showextrema=False)
for pc, col in zip(parts["bodies"], PAL5):
    pc.set_facecolor(col); pc.set_alpha(0.5); pc.set_edgecolor(col)
parts["cmeans"].set_color(JAMA["red"]); parts["cmeans"].set_linewidth(1.4)
ax.set_xticks(np.arange(1, len(cv.columns) + 1))
ax.set_xticklabels(cv.columns, rotation=20, ha="right")
ax.axhline(0.5, color=JAMA["gray"], lw=0.7, ls=":")
ax.set_ylabel("CV AUROC (50 folds)")
ax.set_title("Supervised ML Benchmark — Cross-Validated Discrimination", loc="left")
ax.set_ylim(0.3, 0.95)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_cv_auc.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 9. Calibration plot (logistic predictions, deciles)
# -----------------------------------------------------------------------------
print("=== Calibration plot ===")
phat_logit = m_score.predict(Xfit)
deciles = pd.qcut(phat_logit, q=4, duplicates="drop")
calib = pd.DataFrame({"phat": phat_logit, "y": y, "g": deciles})
cal_g = calib.groupby("g").agg(pred=("phat", "mean"), obs=("y", "mean"),
                                n=("y", "size")).reset_index()
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot([0, 1], [0, 1], color=JAMA["gray"], lw=0.8, ls=":")
ax.scatter(cal_g["pred"], cal_g["obs"], s=cal_g["n"]*4 + 30,
           color=JAMA["navy"], alpha=0.85, edgecolor="white", lw=0.6)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed event rate")
ax.set_title("Calibration — Multivariable Logistic Score", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_calibration.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 10. HTML report
# -----------------------------------------------------------------------------
print("\n=== Building HTML report ===")
def b64_img(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

with open(f"{OUT}/summary.json") as f:
    summary = json.load(f)

# Build cluster table
clust_tbl = pd.DataFrame({
    "Cluster": [f"Cluster {k+1}" for k in range(n_clust)],
    "n": [int(size_per_cluster.loc[k]) for k in range(n_clust)],
    "Rescues": [int(events_per_cluster.loc[k]) for k in range(n_clust)],
    "Rescue rate": [f"{rate_per_cluster.loc[k]*100:.1f}%" for k in range(n_clust)],
})

# Score table
score_tbl = pd.DataFrame({
    "Variable": ["Asymptomatic / incidental presentation",
                 "Separated / gradation hematoma structure",
                 "Maximal axial SDH thickness ≥ 20 mm",
                 "Absence of membranes on baseline CT"],
    "Definition": ["presentation_cat ≠ Symptomatic",
                   "Baseline CT structure",
                   "Baseline CT", "Baseline CT"],
    "Points": ["+2", "+2", "+1", "+1"],
})

# OR table for HTML
or_html = or_tbl.copy()
or_html["OR (95% CI)"] = or_html.apply(
    lambda r: f"{r['OR']:.2f} ({r['OR_lo']:.2f}–{r['OR_hi']:.2f})", axis=1)
or_html["p"] = or_html["p"].apply(lambda v: f"{v:.3f}")
or_html = or_html[["term", "OR (95% CI)", "p", "model"]].rename(
    columns={"term": "Variable", "model": "Model"})

risk_html = risk.rename(columns={"points": "Score", "n": "n",
                                 "rescues": "Rescues", "rate": "Rescue rate"})
risk_html["Rescue rate"] = (risk_html["Rescue rate"] * 100).map("{:.1f}%".format)

uni_tbl = pd.DataFrame({
    "Variable": ["Symptomatic presentation", "Anterior + posterior branches",
                 "Membranes on baseline CT", "Focal deficit",
                 "Separated/gradation structure", "Antiplatelet therapy",
                 "Coils + particles embolic"],
    "n / events": ["131 / 15", "75 / 15", "51 / 4", "33 / 2",
                   "65 / 12", "53 / 11", "94 / 17"],
    "OR (95% CI)": ["0.19 (0.06–0.56)", "2.36 (0.90–6.17)",
                    "0.37 (0.11–1.18)", "0.31 (0.07–1.39)",
                    "2.23 (0.78–6.35)", "2.00 (0.80–4.99)",
                    "2.16 (0.75–6.24)"],
    "p": ["0.003", "0.081", "0.092", "0.125", "0.134", "0.137", "0.153"],
})

cv_summary = (pd.DataFrame({k: {"AUC mean": f"{summary['cv_auc'][k]['mean']:.3f}",
                                "AUC SD":   f"±{summary['cv_auc'][k]['sd']:.3f}"}
                            for k in summary["cv_auc"]}).T
              .reset_index().rename(columns={"index": "Model"}))

def df2html(df_, caption=None):
    html = df_.to_html(index=False, classes="tbl", border=0, escape=False)
    if caption:
        html = html.replace("<table", f'<table summary="{caption}"')
    return html

phen_tbl = pd.DataFrame({
    "Phenotype": ["Cluster 1 — Established chronic membranous",
                  "Cluster 2 — Active separated/gradation",
                  "Cluster 3 — Homogenous low-density"],
    "n": [int(size_per_cluster.loc[0]), int(size_per_cluster.loc[1]), int(size_per_cluster.loc[2])],
    "Rescue rate": [f"{rate_per_cluster.loc[0]*100:.1f}%",
                    f"{rate_per_cluster.loc[1]*100:.1f}%",
                    f"{rate_per_cluster.loc[2]*100:.1f}%"],
    "Defining features": [
        "Older patients (75 y), worse mRS (2), thick SDH (~19 mm), large volume (~117 mL), 88% with membranes — established organized hematoma.",
        "100% symptomatic, predominantly separated/gradation structure (81%), only 37% with membranes — actively organizing non-membranous bleed.",
        "Symptomatic, predominantly homogenous/laminar (87%), no membranes (90%) — uniformly hypodense chronic collection.",
    ],
})

html = f"""
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>MMAE Rescue Surgery Scoring — Combined ML Pipeline</title>
<style>
  body {{ font-family: 'Helvetica', 'Arial', sans-serif; color:#222;
          max-width: 1080px; margin: 32px auto; padding: 0 28px; line-height:1.5;
          background:#FAFAFA; }}
  h1 {{ color:#1F3D5C; font-size:1.55em; border-bottom:2px solid #1F3D5C;
        padding-bottom:6px; margin-top:0;}}
  h2 {{ color:#1F3D5C; font-size:1.18em; margin-top:34px;
        border-left:3px solid #2D7A8F; padding-left:8px; }}
  h3 {{ color:#2D7A8F; font-size:1.0em; margin-top:22px; }}
  .meta {{ color:#666; font-size:0.85em; margin-bottom:20px;}}
  table.tbl {{ border-collapse: collapse; width:100%; margin:8px 0 14px;
               font-size:0.86em; background:white; }}
  table.tbl th {{ background:#1F3D5C; color:white; padding:6px 9px;
                  text-align:left; font-weight:600;}}
  table.tbl td {{ padding:5px 9px; border-bottom:1px solid #E0E0E0;}}
  table.tbl tr:nth-child(even) td {{ background:#F4F6F8; }}
  .figcap {{ font-size:0.82em; color:#444; margin: -4px 0 18px;}}
  .figure {{ text-align:center; margin: 14px 0; }}
  .figure img {{ max-width: 100%; height:auto; border:1px solid #DDD;
                 background:white; padding:6px; }}
  .key {{ background:#EAF1F6; border-left:3px solid #2D7A8F;
          padding:10px 14px; font-size:0.92em; }}
  .caveat {{ background:#FFF3E0; border-left:3px solid #D5751B;
             padding:10px 14px; font-size:0.92em;}}
  code {{ background:#F0F0F0; padding:1px 4px; border-radius:3px; }}
</style></head><body>

<h1>Predicting Rescue Surgery After Stand-Alone MMAE for Chronic SDH</h1>
<div class="meta">
  Combined supervised + unsupervised machine-learning pipeline modeled after
  the BAI score (Maragkos et al., <i>World Neurosurg</i> 2019).<br>
  Cohort: <b>n = {summary['n']}</b> stand-alone MMAE patients —
  rescue surgery in <b>{summary['events']} ({summary['event_rate']*100:.1f}%)</b>.
</div>

<div class="key">
<b>Key result.</b> A 4-variable integer score
(Asymptomatic +2, Separated/gradation +2, Axial ≥ 20 mm +1, No membranes +1; 0–6)
discriminated rescue surgery with an apparent AUC of
{summary['score_auc_apparent']:.2f} and a bootstrap-corrected AUC of
{summary['score_auc_optimism_corrected']:.2f}. Patients scoring ≥ 3 had a
rescue rate of 29.8 % vs 5.5 % below cutoff. Three FAMD-derived clinical
phenotypes (radar plots below) corroborate the imaging-driven nature of the
risk signal.
</div>

<h2>1. Univariate associations</h2>
{df2html(uni_tbl)}

<h2>2. Supervised machine-learning benchmark</h2>
<p>Repeated stratified 5-fold cross-validation (10 reps).</p>
{df2html(cv_summary)}
<div class="figure"><img src="{b64_img(f'{OUT}/fig_cv_auc.png')}" alt="CV AUC violin"/></div>
<div class="figcap"><b>Figure 1.</b> Cross-validated AUC distribution for five
classifiers. Regularized logistic models match or outperform tree ensembles
in this small-event cohort.</div>

<h2>3. Consensus feature importance</h2>
<div class="figure"><img src="{b64_img(f'{OUT}/fig_feature_importance.png')}" alt="Feature importance"/></div>
<div class="figcap"><b>Figure 2.</b> Top 12 features ranked by mean rank
across permutation importance (Gradient Boosting), SHAP, and L1-logistic
coefficient magnitude. Imaging structure variables (separated/gradation,
membranes), SDH thickness/volume, and asymptomatic presentation dominate.</div>

<h2>4. Multivariable logistic model and integer score</h2>
<h3>Score definition</h3>
{df2html(score_tbl)}
<h3>Risk stratification</h3>
{df2html(risk_html)}
<div class="figure"><img src="{b64_img(f'{OUT}/fig_score_risk.png')}" alt="Score risk"/></div>
<div class="figcap"><b>Figure 3.</b> Observed rescue surgery rate per integer
score. Dashed line marks the proposed binary cutoff (≥ 3 points).</div>

<h3>Discrimination</h3>
<div class="figure"><img src="{b64_img(f'{OUT}/fig_roc.png')}" alt="ROC"/></div>
<div class="figcap"><b>Figure 4.</b> Receiver-operating-characteristic curves
for the multivariable logistic model and the integer 0–6 score. The integer
score retains essentially the full discriminative ability of the underlying
model (apparent ΔAUC ≤ 0.04).</div>

<div class="figure"><img src="{b64_img(f'{OUT}/fig_calibration.png')}" alt="Calibration"/></div>
<div class="figcap"><b>Figure 5.</b> Calibration plot (4 quartiles of
predicted risk; bubble size proportional to subgroup n).</div>

<h2>5. Unsupervised phenotyping (FAMD + multiple clustering algorithms)</h2>
<p>Factor Analysis of Mixed Data on 36 mixed numeric/categorical features
followed by K-means (k=3 chosen for interpretability and balance).
Robustness checked against Ward, Gaussian Mixture, K-Prototypes, and
Hierarchical/Gower clustering.</p>

{df2html(phen_tbl)}

<div class="figure"><img src="{b64_img(f'{OUT}/fig_famd_scatter.png')}" alt="FAMD scatter"/></div>
<div class="figcap"><b>Figure 6.</b> FAMD-1/FAMD-2 projection. Left:
KMeans-derived cluster labels. Right: rescue surgery overlay.</div>

<div class="figure"><img src="{b64_img(f'{OUT}/fig_radar_clusters.png')}" alt="Radar per cluster"/></div>
<div class="figcap"><b>Figure 7.</b> Phenotype radar profile per cluster
(median for numeric; proportion-positive for categorical; min-max scaled to
the 5th–95th-percentile cohort range).</div>

<div class="figure"><img src="{b64_img(f'{OUT}/fig_radar_overlay.png')}" alt="Radar overlay"/></div>
<div class="figcap"><b>Figure 8.</b> Overlay radar showing simultaneous
cluster comparison.</div>

<h2>6. Cluster-membership regression</h2>
<p>Logistic regression with cluster dummies (Cluster 1 = reference).
Adjusted model controls for age, baseline mRS, antiplatelet therapy,
and axial thickness ≥ 20 mm.</p>
{df2html(or_html)}
<div class="figure"><img src="{b64_img(f'{OUT}/fig_forest_clusters.png')}" alt="Forest plot"/></div>
<div class="figcap"><b>Figure 9.</b> Crude vs adjusted odds ratios for
cluster membership predicting rescue surgery. Wide CIs reflect the small
event count (22 rescues) — interpret as exploratory.</div>

<div class="caveat">
<b>Caveats.</b> (1) Only 22 events constrain statistical power; the corrected
AUC of 0.76 must be externally validated. (2) The “asymptomatic” coefficient
is counter-intuitively large (7/17 = 41 % rescue) and likely reflects a
selection bias in surveillance intensity. (3) Clusters are not strongly
separated (silhouette ≤ 0.10) — useful for phenotype description but not
a stand-alone risk classifier. (4) Replace axial-thickness ≥ 20 mm with
volumetric ≥ 100 mL once segmentation reproducibility is confirmed.
</div>

<h2>7. Files</h2>
<p>All raw outputs (CSVs, PNGs at 300 dpi, summary JSON, scoring code) are
saved alongside this report. The two analysis scripts are
<code>build_score.py</code> (supervised + integer score + bootstrap) and
<code>unsupervised_compare.py</code> (FAMD + clustering benchmark).</p>

</body></html>
"""

with open(f"{OUT}/REPORT.html", "w") as f:
    f.write(html)
print(f"\nSaved REPORT.html ({len(html)//1024} kB)")
print("\nDone.")
