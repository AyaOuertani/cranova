# Requirements: Forensic Facial Approximation from Cranial Measurements

## 1. Purpose
Estimate facial geometry from cranial measurements, as an *approximation tool*
for forensic research. It does not identify individuals.

## 2. Scope
- Input: 17 cranial inter-landmark distances (mm) (16 used as primary features), plus sex and ancestry.
- Output: 66 facial inter-landmark distances between 12 landmarks (mm),
  converted to 3D landmark positions.
- Population: AFR, ASN, EUR subjects (matched subjects only, ~274).
- Hispanic (HIS) subjects have no cranial data: not used for supervised
  cranial-to-facial training.

## 3. Out of scope (unless later phases prove it is supported by data)
- Facial regions with no ground truth (forehead, chin contour, jaw, ears,
  scalp, back of head): these come from templates/priors and are labelled so.
- Identification or matching of individuals.
- Any claim about ancestries or ages not present in the data.

## 4. Success criteria
| ID | Criterion | How it is tested |
|----|-----------|------------------|
| S1a | Model MAE on held-out subjects is lower than the mean-predictor baseline | Paired comparison, 95% CI excludes zero |
| S1b | Non-inferiority to ReFace (2.26 mm MAE) | Upper end of the 95% CI of (model minus ReFace) is below +0.10 mm |
| S2 | Model beats a predict-the-mean baseline | Same test set |
| S3 | Predicted distances are geometrically consistent (3D landmark reconstruction error reported) | MDS reconstruction residual |
| S4 | Every prediction has an uncertainty estimate, and its coverage is verified on held-out subjects | Empirical coverage vs nominal level |
| S5 | Performance reported per ancestry and per sex, not only overall | Subgroup tables |
| S6 | Every reported number is reproducible from a clean clone | Fixed seed, locked environment, data checksums |

The size of the improvement that counts as "meaningful" in S1 is fixed in
Step 5, after the baselines are measured, and before any model is trained.

## 5. Constraints
- ~274 matched subjects; small sample, so simple models come first.
- Distances only; no 3D coordinates are available.
- Datasets are CC BY 4.0; cite Data in Brief 2022 and the Mendeley datasets.
## 6. Ethics and limitations (to be expanded in the thesis)
- Outputs are approximations with stated uncertainty, not evidence of identity.
- Interface must distinguish measured, predicted and template-derived regions.
- No fabricated metrics or confidence scores anywhere in the application.