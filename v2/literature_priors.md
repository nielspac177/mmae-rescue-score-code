# Informative priors for Bayesian Model 5 — synthesis of published MMA series

Search/synthesis date: 2026-05-08. Sources: Salem 2022 (multi-center, n=509), Catapano 2021 (Barrow, n=144), Ban 2018 (Korean, n=72), Liu 2023 meta (n=2860), Joyce 2023 meta (n>2000), Pacchiarotti 2024 review, Kan 2024 EMBOLISE (n=400), Tian 2024 MAGIC-MT (n=722). Effect sizes are multivariable-adjusted log-odds ratios pooled informally across these series; SDs widened to reflect between-study heterogeneity (random-effects-style).

| Predictor | Source effect (OR) | Prior mean (log OR) | Prior SD | 95% prior CI for OR | Notes |
|---|---|---|---|---|---|
| Anticoagulation (Yes vs No) | Joyce 2023 OR 2.7; Salem 2022 OR 2.1; Liu meta OR 1.9 | **0.69** (OR 2.0) | 0.30 | 1.11 – 3.60 | Strongest cross-study consensus |
| Antiplatelet therapy | Joyce 2023 OR 1.8; Salem 2022 OR 1.5 | **0.53** (OR 1.7) | 0.30 | 0.94 – 3.06 | Slightly weaker than anticoag |
| SDH volume ≥ 100 mL | Catapano 2021 per 10 mL OR 1.1 (≥100 ≈ 2.0); Salem 2022 OR 1.8 | **0.69** (OR 2.0) | 0.35 | 1.01 – 3.97 | Continuous → binary extrapolation |
| Midline shift ≥ 5 mm | Ban 2018 OR 2.4; Catapano per mm OR 1.12 (5 mm ≈ 1.76) | **0.69** (OR 2.0) | 0.35 | 1.01 – 3.97 | Robust radiographic predictor |
| Age (per ordinal bin <65/65–80/>80) | Salem per-decade OR 1.4; Joyce 1.3 (per ~15 yr) | **0.40** (OR 1.5) | 0.20 | 1.01 – 2.21 | Per ordinal step, not per year |
| Platelets < 150 ×10⁹/L | Catapano OR 2.1; smaller series 1.5–2.0 | **0.59** (OR 1.8) | 0.40 | 0.82 – 3.95 | Wider SD — heterogeneous |
| Bilateral SDH | Salem 2022 OR 2.4; Catapano 1.6 | **0.69** (OR 2.0) | 0.35 | 1.01 – 3.97 | Indication-bias-aware |
| Mixed / separated hematoma density | Catapano OR 1.7; Salem 1.5 | **0.47** (OR 1.6) | 0.35 | 0.81 – 3.18 | Phenotype mostly Catapano |
| Anterior + posterior embolization | Internal series mixed; EMBOLISE didn't subgroup | **0.34** (OR 1.4) | 0.50 | 0.53 – 3.74 | Weakly informative |
| Hypertension, statins, gait, focal_deficit, etc. | No reliable multivariable signal | **0.0** | 1.00 | 0.14 – 7.39 | Default weakly-informative |
| Intercept | Cohort-anchored | **logit(0.168) = −1.60** | 1.00 | — | Centered on 16.8% event rate |

## Use in `build_model5_v2.py`

```python
PRIORS = {
    "intercept":       (-1.60, 1.00),
    "anticoag":        ( 0.69, 0.30),
    "antiplatelet":    ( 0.53, 0.30),
    "sdh_vol_ge100":   ( 0.69, 0.35),
    "mls_ge5":         ( 0.69, 0.35),
    "age_pts":         ( 0.40, 0.20),
    "plt_lt150":       ( 0.59, 0.40),
    "bilateral":       ( 0.69, 0.35),
    "mixed_density":   ( 0.47, 0.35),
    "ant_post":        ( 0.34, 0.50),
    # default weakly informative for unmapped features
    "_default":        ( 0.00, 1.00),
}
```

## Sensitivity

- Wider priors (SD doubled) → posterior closer to MLE → check robustness
- Tighter priors (SD halved) → over-shrinkage risk if local cohort genuinely differs
- Run both as sensitivity-of-sensitivity in build_model5
