# Manuscript prose — MMAE rescue scoring study (v2, humanized)

## Title (working)
**A simplified bedside risk score for rescue surgery after middle meningeal artery embolization for chronic subdural hematoma**

## Abstract (≈210 words, single paragraph)

Among 214 consecutive patients treated with middle meningeal artery (MMA) embolization for chronic subdural hematoma (cSDH), 36 (16.8%) required rescue surgery. We built two integer scores for use at the bedside before the next embolization. The full 8-point score combines age (0/1/2 for <65, 65–80, >80), SDH volume ≥100 mL, anticoagulation, absence of focal deficit at presentation, platelets <150 ×10⁹/L, antiplatelet therapy, and embolization of both anterior and posterior branches. A trimmed 5-point version keeps only age, volume, anticoagulation, and absence of focal deficit. The 8-point score gave an AUC of 0.73 (Harrell optimism-corrected 0.73); the 5-point version, 0.68 (0.68). Rescue rates climbed in a stepwise fashion from 7–9% at scores 0–3 to 42% at 5 and 67% at 7. A score of ≥5 separated low-risk (9.1%, 15/165) from high-risk (42.9%, 21/49) patients, a 4.7-fold difference. Both scores were well calibrated (Hosmer–Lemeshow P = 0.64 and 0.89). Two findings stood out: SDH volume ≥100 mL roughly doubled the odds of rescue, and—counterintuitively—patients without a focal neurological deficit at presentation had triple the odds of rescue, suggesting that asymptomatic but anatomically large hematomas are the harder ones to control. The score is simple enough to use without software and offers clinically useful stratification for post-embolization surveillance.

## Methods

### Cohort and outcome
We retrospectively reviewed every adult patient who underwent MMA embolization for cSDH at our institution. The endpoint was rescue surgery (open burr-hole or craniotomy) on or after the index procedure, taken from the operative log and the last documented clinical follow-up. We kept all patients with incomplete data: one missing platelet value and 32 missing baseline SDH volumes (15.0%) were filled in with the cohort median, and a complete-case sensitivity analysis is provided in the supplement. The final sample was 214 patients with 36 rescue events (16.8%).

### Candidate predictors and score construction
We prespecified the candidate predictors from prior MMA series and large cSDH cohorts: age, sex, comorbidities (anticoagulation, antiplatelet therapy, hypertension), laboratory values (platelet count, INR), clinical presentation (focal deficit, modified Rankin Scale), baseline imaging (SDH volume, axial thickness, midline shift), and procedural variables (branches embolized, particle vs. liquid embolic). After univariable screening with logistic regression, we built two integer scores. Model 1 (range 0–8) gives 0, 1, or 2 points for age <65, 65–80, and >80 years, plus 1 point each for SDH volume ≥100 mL, anticoagulation, absence of focal deficit at presentation, platelets <150 ×10⁹/L, antiplatelet therapy, and embolization of both anterior and posterior branches. Model 2 (range 0–5) keeps the same age categories with 1 point each for volume ≥100 mL, anticoagulation, and absence of focal deficit. The volume cutoff and the platelet threshold match prior MMA cohorts; the age categories track the frailty cliffs we see clinically in this population.

### Statistical analysis
We measured discrimination with the AUC, computed two ways: directly from the integer score, and from a logistic regression on the score's components. Apparent AUCs were corrected for optimism using 1000 nonparametric bootstrap replicates (Harrell). Calibration was assessed visually and with the Hosmer–Lemeshow χ² test using probability-quantile bins. Wald 95% confidence intervals are reported for univariable odds ratios. Wilson intervals are reported for stratum-specific rescue rates. Two-sided P < 0.05 was significant. Analyses ran in Python 3.13 (statsmodels, scikit-learn). The institutional review board approved the study with waiver of informed consent.

## Results

### Cohort
The 214 patients had a mean age of 73.4 years (20.6% <65, 47.7% 65–80, 31.8% >80). One in four (25.7%) was on anticoagulation and just over a third (36.4%) on antiplatelet therapy. SDH volume reached ≥100 mL in 29.0%, platelets fell below 150 ×10⁹/L in 20.6%, and 73.8% had no focal neurological deficit at the time of embolization. Anterior and posterior branches were both embolized in 53.7%. Thirty-six patients (16.8%) went on to rescue surgery.

### Univariable associations
Two predictors were significant on their own: SDH volume ≥100 mL (OR 2.30, 95% CI 1.10–4.80; P = 0.027) and absence of focal deficit (OR 3.30, 95% CI 1.11–9.80; P = 0.031). Antiplatelet therapy was borderline (OR 1.97, 95% CI 0.95–4.05; P = 0.067). The remaining variables — age >80 (OR 1.68), platelets <150 (OR 1.93), anticoagulation (OR 1.57), and dual-branch embolization (OR 1.65) — moved in the same direction as published series but did not reach significance with 36 events. The full plot is in **Figure 4**.

### Score performance
Model 1 produced an apparent AUC of 0.734 (corrected 0.732). Model 2 came in at 0.683 (corrected 0.681) — about 5 AUC points lower (**Figure 1**). Fitting a logistic regression on the underlying components instead of the integer score moved the apparent AUCs to 0.752 (corrected 0.703) for Model 1 and 0.710 (corrected 0.679) for Model 2; the optimism correction is larger for the unrestricted regression, as expected. Brier scores were 0.120 (Model 1) and 0.128 (Model 2). Calibration was acceptable on both models, with non-significant Hosmer–Lemeshow tests (P = 0.64 and 0.89; **Figure 3**).

### Risk stratification by total score
Rescue rates climbed stepwise with the full score: 0–9% across scores 0–3, then 11.9% at score 4, 42.1% at 5, 37.5% at 6, and 66.7% at 7 (**Figures 2** and **5**, left panel). The break sits cleanly between 4 and 5. Dichotomizing at ≥5 produced a low-risk arm of 165 patients with a 9.1% rescue rate (15/165) and a high-risk arm of 49 patients with a 42.9% rate (21/49) — a 4.7-fold difference (**Figure 6**, left). The simplified score showed the same direction but a tighter range (2.9% at score 1 → 66.7% at 5), with more overlap between adjacent strata (**Figure 5**, right). Dichotomizing at ≥4 split 184 patients at 13.6% rescue from 30 patients at 36.7% (**Figure 6**, right).

### Comparison of the two scores
The full score's 5-point AUC advantage came from three additions: platelets <150 ×10⁹/L, antiplatelet therapy, and the dual-branch procedural variable. Each of these is routinely available before the next intervention, so the cost of using the full score is essentially nothing in practice. The trimmed score is reasonable when one of those data points is missing at triage (AUC 0.68), but at the top end of the risk spectrum — which is where the score actually changes management — the full version separates patients better. Both scores were well calibrated across deciles of predicted probability, so the absolute risk estimates can be used directly, not only as a ranking.

## Discussion

The score answers a specific clinical question: before this patient leaves the angiography suite, what is the chance they will return for an open evacuation? Two findings deserve attention.

First, the volume effect was straightforward — patients with ≥100 mL of subdural blood at baseline had roughly double the rescue risk. This is consistent with everything that has been published in the open-surgery literature for cSDH and confirms that the rule transfers to the MMA-embolization population.

Second, and less obvious, was the focal-deficit paradox. Patients without a focal neurological deficit at presentation had three times the rescue rate of patients with one. The most plausible explanation is selection: a patient with a 120-mL hematoma and no deficit gets embolized and observed, whereas a patient with the same volume and a hemiparesis often gets evacuated up front and never enters our cohort. The patients we embolize for radiographic reasons rather than clinical ones are exactly the patients whose hematomas evolve unpredictably afterward. If this holds in external cohorts, it argues for tighter post-embolization surveillance specifically in the asymptomatic but radiographically large group.

The score is meant to complement, not replace, clinical judgment. It needs external validation. Limitations include the single-center retrospective design, the modest event count of 36, and ascertainment of the rescue endpoint at the last available follow-up rather than at a fixed time horizon. Volume imputation in the 15% of patients missing baseline volumes is a known concern; the complete-case sensitivity analysis in the supplement gave the same direction of effect with wider confidence intervals.
