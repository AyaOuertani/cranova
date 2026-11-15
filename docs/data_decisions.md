# Data cleaning decisions (Phase 2)

Rule applied to every decision: it serves an objective, is feasible with the data, is testable, and is logged.
Raw files are never modified; every change is made by `src/build_clean_data.py` and recorded.

| ID | Decision | Reason | Evidence / test |
|----|----------|--------|-----------------|
| D1 | Model only the 274 subjects present in cranial AND facial data (271 also have tissue depth). The 13 cranial-only subjects (cau096 and the HWF/HWM group) and 114 facial-only subjects stay in the subject registry but are not modelled. | A supervised model needs both input and target. | `subject_registry.csv`, `test_matching_counts` |
| D2 | OBBL (orbital breadth) is not a primary input. | Missing for all 47 ASN males. | `test_obbl_missing_only_for_asnm` |
| D3 | Five tissue-depth columns (me, sg, az L, az R, gn - z) are excluded. | Not defined in the workbook's landmark reference; cannot be interpreted. | `data_dictionary.csv`, `test_tissue_depth_cleaning` |
| D4 | One tissue-depth value (his021, mmb R, 74.63 mm, robust z = 21.1) is set to missing. | Physically implausible for a tissue depth: likely an entry error. Logged. Values that are extreme but plausible (asn029, asn036 sc) are kept. | `cleaning_log.csv` |
| D5 | Tissue depth is an auxiliary training target only, never an input. | It is unknown for an unknown skull. | `data_dictionary.csv` |
| D6 | Cranial outliers are kept and flagged, not removed. | No evidence of errors; removal would shrink real variation. | `cranial_outlier_flags_dev.csv` |
| D7 | Missing values stay missing. No imputation in the cleaned files. | Imputation, if any, belongs inside cross-validation folds. | — |
| D8 | ID typo ans012 corrected to asn012. | Documented in dataset documentation. | `test_id_typo_is_fixed` |
| D9 | ReFace values are never model inputs, only a baseline. | Not available for a new skull without running ReFace. | `data_dictionary.csv` |