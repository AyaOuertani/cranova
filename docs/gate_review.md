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
