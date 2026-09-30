# UtA mechanistic modelling protocol

## Primary question

Can a gestation-dependent, radial-dilation-dominated model reproduce the population distribution and within-woman evolution of uterine-artery Doppler indices more plausibly than models driven by distal resistance or arterio-venous shunt development?

## Scope and interpretation

The available measurements are Doppler summary indices rather than vessel dimensions or raw waveforms. Model parameters inferred from these data will therefore be described as **model-equivalent latent vascular parameters**, not anatomical measurements. RI and S/D are algebraically redundant, so they must not be treated as independent outcomes.

## Phase 1: reproducibility and data audit

1. Preserve the downloaded Clark et al. code unchanged in `external/`.
2. Reproduce its baseline output through a compatibility adapter.
3. Audit missingness, duplicate keys, algebraic consistency, gestational coverage, and consistency between examination-date intervals and GA intervals.
4. Exclude the 17 composite rows that concatenate multiple examination dates and reports; their Doppler values cannot be assigned reliably to a single GA. Treat the remaining four observations before 14 weeks as insufficient for first-trimester trajectory calibration.

## Phase 2: sensitivity and identifiability

1. Build solver response surfaces over radial radius, A-V shunt radius, terminal resistance and compliance, arterial stiffness, uterine/arcuate radius, and heart rate.
2. Compute local Jacobians, parameter correlations, profile likelihoods and synthetic-recovery experiments.
3. Select only parameter combinations that can be identified from PI plus one non-redundant waveform summary.
4. Constrain non-identifiable parameters using literature-informed ranges; do not report falsely precise individual anatomy.

## Phase 3: hierarchical calibration

Fit competing nonlinear hierarchical models:

- **M0 statistical benchmark:** spline of log(PI) over GA with woman-level random effects.
- **M1 radial adaptation:** a monotone gestational trajectory in model-equivalent radial radius; other parameters fixed or tightly constrained.
- **M2 distal adaptation:** gestational trajectory in terminal resistance.
- **M3 shunt adaptation:** gestational trajectory in effective A-V shunt calibre.
- **M4 parsimonious combined model:** only if phase-2 recovery demonstrates identifiability.

Account for repeated examinations using woman-level random effects. Use robust observation error and pre-specified exclusion or sensitivity rules derived from the audit.

## Validation

- Split by woman, never by examination.
- Use repeated nested cross-validation or a locked 70/30 development-validation split plus bootstrap uncertainty.
- Compare held-out PI error, calibration, predictive interval coverage and trajectory prediction with M0.
- Test whether an early examination improves prediction of a later examination within the subset with repeated measurements.
- Report solver numerical failures and parameter-boundary estimates.

## Secondary fetal-growth analysis

Model gestational-age-adjusted log(EFW), not raw EFW. Separate between-woman and within-woman associations where possible. Treat this analysis as exploratory unless later delivery outcomes become available.

## Required additions for stronger clinical claims

Seek maternal blood pressure, heart rate, BMI, parity, placental laterality, notch status, left/right UtA measurements, delivery GA, birthweight, pre-eclampsia/FGR diagnoses and placental pathology. Raw velocity waveforms from even a smaller nested cohort would materially improve identifiability.

## Reporting safeguards

- Follow TRIPOD-style reporting for prediction components and report mechanistic assumptions explicitly.
- Publish solver version, parameter bounds, seeds, woman-level split IDs and complete calibration diagnostics.
- Distinguish mechanistic compatibility from proof that radial arteries caused the observed clinical pattern.
