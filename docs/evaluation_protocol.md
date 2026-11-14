# Evaluation Protocol (v1, frozen before any modelling)

## 1. Data used
- Modelling set: 274 subjects with both cranial and facial data
  (AFR, ASN, EUR). Hispanic subjects have no cranial data and are excluded
  from cranial-to-facial modelling.
- Inputs: 16 cranial measurements, plus sex and ancestry (AFR, ASN, EUR). OBBL is
  excluded because it is missing for all 47 ASNM subjects, and imputing it cannot be
  validated for that group. OBBL is studied later as a sensitivity analysis on the
  other groups. In practice sex and ancestry are estimated by the anthropologist, so
  evaluation with the recorded labels is optimistic (risk R14).
- Targets: 66 facial distances measured on the CT-derived face (the `_k` version).

## 2. Split
- File: `splits/split_v1.csv`, created by `src/make_split.py` (seed 42).
- Test set: 55 subjects (20%), stratified by ancestry x sex. Locked.
  It is used once, for the final selected model only.
- Development set: 219 subjects. All model selection and tuning happen here,
  using repeated stratified 5-fold cross-validation (10 repeats, seeds 0-9).
- Splits are by subject. Each subject's `_k` and `_r` faces always stay together.
- Preprocessing (scaling, imputation, feature selection) is fitted on the
  training folds only, inside the cross-validation loop.

## 3. Metrics
- Primary: mean absolute error (MAE, mm) per subject over the 66 distances.
- Secondary: RMSE, per-distance MAE, and all of these per ancestry and per sex.
- Geometry metrics (3D landmark error) are defined in Phase 4.
- Uncertainty: coverage of 90% prediction intervals, checked in
  cross-validation and on the test set.

## 4. Baselines
1. Mean predictor: the training-fold mean of each distance.
2. ReFace: the `_r` distances of the same subjects.
3. Ridge regression on the 16 cranial measurements.
4. Bias-corrected ReFace: the ReFace distance minus its mean error per distance,
   estimated on the training folds only.
5. Ridge regression on the 16 cranial measurements plus sex and ancestry.
## 5. Comparison rule
- Models are compared with ReFace and with each other by the per-subject
  difference in MAE, with a 95% bootstrap confidence interval
  (10,000 resamples, subjects as the unit).
- A model counts as an improvement only if the interval excludes zero.
- The minimum size of a meaningful improvement is 0.10 mm (about 5% of ReFace's
  2.26 mm MAE), measured against the best baseline, with the confidence interval
  excluding zero. Set after measuring the baselines and before training any model.
## 6. Known limitations
- The test set is small (55 subjects), so test-set intervals will be wide.
  Cross-validation on the development set is the main evidence; the test set
  is the confirmation.
- It is not known whether ReFace's own training data included these subjects.
  The ReFace comparison is therefore indicative, not a strict like-for-like test.

## 7. Change control
- Any change to this protocol is logged below with the date and the reason.
- No model is trained or tuned using the test set.

### Change log
- 2026-11-14: v1 created.
- 2026-11-14: Sex and ancestry added as allowed inputs; ridge + sex + ancestry added as
  baseline 5; meaningful-improvement threshold set to 0.10 mm. Decided from
  development-set results only; no test-set information used.