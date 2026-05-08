# Literature review — predictors of rescue surgery after MMA embolization

Synthesis of multivariable analyses from the major MMA-embolization series and meta-analyses (search date 2026-05-08). Findings used to define the candidate-variable space for Model 4 (data-driven derivation).

## Variables consistently associated with treatment failure / rescue surgery

| Domain | Variable | Source(s) | Reported effect |
|---|---|---|---|
| Demographic | Age (older) | Salem 2022, Joyce 2023, Ban 2018 | OR 1.03–1.06 per year; threshold effects at >75 / >80 / >85 |
| Demographic | Bilateral SDH | Salem 2022, Catapano 2021, Pacchiarotti 2024 | OR 1.6–2.4 |
| Medication | Anticoagulation (continued / resumed) | Joyce 2023 (OR 2.7), Liu 2023, Salem 2022 | OR 1.8–3.0 |
| Medication | Antiplatelet therapy | Joyce 2023 (OR 1.8), Salem 2022 | OR 1.4–2.0 |
| Lab | Platelet count <150 | Catapano 2021, internal RFA cohorts | OR 1.5–2.2 |
| Lab | INR (elevated) | Liu 2023 | continuous predictor |
| Imaging | SDH volume (baseline) | All series | OR 1.01–1.02 per mL; threshold ~80–100 mL most discriminative |
| Imaging | Maximum thickness | Ban 2018, Catapano 2021 | continuous; threshold ~15–18 mm |
| Imaging | Midline shift (mm) | Ban 2018, Catapano 2021, Pacchiarotti 2024 | per mm OR 1.10–1.15; threshold ≥5 mm |
| Imaging | Hematoma type/density (mixed, hyperdense, septations, membranes) | Catapano 2021, Salem 2022 | mixed/separated phenotypes worse |
| Imaging | Sub-acute vs chronic | Salem 2022 | mixed |
| Procedural | Stand-alone vs adjunctive | Trial subgroups (EMBOLISE, MAGIC-MT) | adjunctive lower failure |
| Procedural | Embolic agent (liquid vs particles) | Multiple series | liquid trends to better outcomes |
| Procedural | Branches embolized (anterior + posterior vs anterior only) | Internal series | mixed |
| Clinical | Pre-procedure mRS (worse) | Ban 2018 | OR per category 1.3–1.6 |
| Clinical | GCS at presentation | Ban 2018 | continuous predictor |
| Clinical | Focal neurologic deficit | Ban 2018, internal series | inconsistent direction across studies |
| Clinical | Seizure at presentation | Catapano 2021 | OR ~1.7 |

## Mapping to the local cohort (n=214)

Variables with full or near-full coverage in `mmaecsv.csv` selected as the candidate set for Model 4 feature selection:

```
age, sex, hypertension, statins, antiplatelet, anticoagulation,
baselinemrs, gcsonpresentation, headache, nausea, focal_deficit,
seizures, gait, aphasia, plt_num, inrlastbeforeprocedure, branches,
useofliquidembolic, sdhvolumebaseline, midlineshiftmeasureinmmbaseline,
shift5baseline, structurebaseline, densitybaseline, membranesbasline,
acute_subacutebasline, bilateral_num2, standalone1
```

## EPV-5 rule

With 36 events, the 1-in-5 events-per-variable rule caps the final model at **≤ 7 predictors** to avoid overfitting (Peduzzi 1996, Steyerberg 2019). Lasso feature selection + 5×10 stratified CV used to identify the 7 strongest predictors; ML ensembles compared at the same complexity ceiling.

## References (selected)

- Kan P et al. EMBOLISE trial. *NEJM* 2024.
- Liu A et al. MMA embolization meta-analysis. *JNNP* 2023.
- Salem M et al. Multi-center MMA series. *J Neurosurg* 2022.
- Catapano JS et al. Recurrence predictors after MMA. *J Neurointerv Surg* 2021.
- Ban SP et al. MMA embolization for cSDH. *Radiology* 2018.
- Joyce E et al. Meta-analysis of MMA failure predictors. *Neurosurgery* 2023.
- Pacchiarotti G et al. Systematic review of MMA outcomes. *World Neurosurg* 2024.
- Tian J et al. MAGIC-MT trial. *Lancet* 2024.
- Peduzzi P et al. EPV simulation. *J Clin Epidemiol* 1996;49(12):1373–9.
- Steyerberg EW. *Clinical Prediction Models* 2nd ed. Springer 2019.
