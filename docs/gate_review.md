# Gate Review: End of Part 1 (2026-11-14)

## Decision
GO, with revised success criteria.

## Evidence (development set, 219 subjects, repeated 5-fold CV, test set untouched)
| Model | MAE (mm) | Difference vs ReFace (mm), 95% CI |
|-------|----------|-----------------------------------|
| Mean predictor | 3.47 | +1.21 [+1.03, +1.40] |
| ReFace | 2.26 | n/a |
| ReFace bias-corrected | 2.14 | -0.12 [-0.17, -0.07] |
| Ridge, 16 cranial measures | 2.52 | +0.26 [+0.15, +0.38] |
| Ridge + sex + ancestry | 2.32 | +0.06 [-0.03, +0.16] |

Exploratory probe (3 repeats, development data only, not part of the formal protocol):
- Ridge + sex + ancestry: 2.33 mm, +0.07 [-0.02, +0.16] vs ReFace.
- PLS, low-rank targets and a small neural network on the 16 measures: all about 2.53 mm.
- Ridge + each subject's true tissue depth (not available for an unknown skull): 2.37 mm.
- ReFace + cranial correction: 2.13 mm, -0.14 [-0.19, -0.08] vs ReFace.

## What we learned
1. Cranial measurements carry real signal (about 27% lower error than the mean).
2. Model complexity does not help on these 16 inputs; the limit is information.
3. Sex and ancestry as inputs close most of the gap to ReFace.
4. Tissue depth cannot be an input for an unknown skull; it can only be a training
   target or a population prior.
5. Subgroup error is highest for AFR for every model.

## Revised success criteria
- S1a: the final model beats the mean predictor, 95% CI excluding zero.
- S1b: non-inferiority to ReFace: the upper end of the 95% CI of (model minus ReFace)
  is below +0.10 mm. A claim of being better than ReFace needs the upper end below 0.
- S4 to S6 unchanged.

## Conditions and open items
- Ask the supervisor: can ReFace be run on new skulls? (decides whether a hybrid
  model is possible.)
- Resolve the data questions in the dataset documentation (HWF/HWM subjects,
  `gn - z`, suspicious tissue-depth values).
- Roadmap Phase 1 still needs the literature review and pretrained-model survey.
- All results above are cross-validated on 219 development subjects; none come
  from the locked test set.

## Phase 4 gate: geometry (distances -> 3D landmarks)
- The 66 true distances are embeddable in 3D: reconstruction error at most 0.0065 mm.
- Passing predicted distances through 3D loses nothing (MAE 2.336 -> 2.335 mm).
- Mean 3D landmark error 2.59 mm (90% below 4.40 mm); best stomion 1.75, worst exocanthions 3.1-3.2 mm.
- Mirror ambiguity removed by convention (nose tip in front); tested against rotation, translation, mirroring.
- Display: predicted faces are placed on the training mean face; 90% landmark radii average 4.66 mm,
  honest for all 12 landmarks and all 6 ancestry x sex groups (90-93%).
- Rejected: stress refinement after MDS (no gain); eye-based display axes (radii 6.55 mm vs 4.66 mm).

## Phase 5 gate: tissue depth integration
- Tissue depth cannot be a model input: it is unknown for an unknown skull.
- Classic use (depth vectors placed at skull landmarks) is not feasible: the cranial data has
  17 distances, no 3D skull landmark coordinates.
- Tested use as an auxiliary training target (E4): +0.020 mm improvement over the same network
  without it (CI -0.033 to -0.007), far below the 0.10 mm threshold.
- Personalised tissue-depth estimates for the user were considered and deferred: they would not
  improve the face prediction (main objective). Can be added later without changing the pipeline.

## Phase 6 gate: complete head (GNM Head v3, Apache 2.0)
- Landmark correspondence calibrated on    development subjects: every landmark within 1.4 mm of the average CT face (iBUG start was up to 7.5 mm off for eye corners and alar points).
- Fixed a mirror (handedness) bug in orient_face; earlier error metrics unaffected (verified).
- GNM fitting does NOT improve landmark accuracy: 2.575 -> 2.667 mm (+0.092, CI +0.075 to +0.109);
  the penalty always went to the smallest value. Ridge already encodes the population prior of our data.
- Decision: GNM is for display only (penalty 1, head coefficients stay within ~2 SD). The predicted
  landmarks and their 90% spheres remain the scientific output.
- Limit: the head is validated only at the 12 landmarks; the rest of the surface is GNM's plausible guess.