# Risk Register (v1, 2026-11-14)

| ID | Risk | L | I | Mitigation | Evidence / trigger |
|----|------|---|---|-----------|--------------------|
| R1 | Small sample: 274 modelling subjects, 55 in the test set, so estimates and intervals are noisy | H | H | Repeated CV on the development set; report confidence intervals; simple models first | Protocol section 6 |
| R2 | ReFace is a strong baseline and we may not beat it | H | H | Add a bias-corrected ReFace baseline; define fallback contribution (calibrated uncertainty, open reproducible pipeline) if S1 fails | ReFace MAE 2.26 mm vs 3.46 mm for the mean predictor; better on all 66 distances (EDA Fig 6) |
| R3 | No Hispanic cranial data, so the model does not cover that population | H | M | Out-of-range input warning (U2); state the limitation in the report and the dashboard | 0 HIS subjects in the cranial file |
| R4 | Distances only, 12 landmarks: no coordinates, mirror ambiguity, no forehead, jaw, chin or ear ground truth | H | M | Reconstruct up to reflection; resolve with a template; label template-derived regions | Phase 1 geometry check |
| R5 | Likely data-entry errors in tissue depth | M | M | Review the flagged values against the source; use robust methods; never delete silently | afr100 az about 110 mm; his021 mmb R 74.6 mm |
| R6 | Missing values: OBBL (all ASNM subjects) and az L/R (about 50%) | H | L | OBBL dropped from primary features; az excluded or treated as a sensitivity analysis | Dataset documentation |
| R7 | Data leakage: test set used during development, or one subject's `_k` and `_r` faces split across train and test | M | H | Frozen split; subject-level folds; preprocessing inside CV; EDA on development subjects only | Protocol sections 2 and 7 |
| R8 | cans were collected as ReFace's reference database (Data in Brief 2022) | M | M | overlap with ReFace's build data unverified | Protocol section 6 |
| R9 | Undocumented tissue-depth columns (`gn - z`, `me`, `sg`, `az`) | M | M | Clarify with the supervisor or source paper before Phase 5; exclude if unresolved | Dataset documentation |
| R10 | Uncertainty intervals unreliable at this sample size | M | H | Conformal or ensemble methods; verify coverage in CV and on the test set; report if coverage is poor | Requirement S4 |
| R11 | Head-model licence or technical limits (Phase 6) block the planned complete-head representation | M | M | Check licence and compatibility before building; keep a simple template as fallback | Roadmap Phase 6 |
| R12 | Data licence unknown, limiting what can be shared (code, thesis appendices) | M | M | Keep raw data out of Git; verify the licence before publishing anything derived | `.gitignore` |
| R13 | Outputs read as identification evidence | L | H | Wording in the app and thesis; always show uncertainty; label predicted vs template regions | Requirements section 6 |
| R14 | Sex and ancestry are estimated in practice and may be wrong; evaluation with recorded labels is optimistic | M | M | Test robustness to wrong labels; show the assumed sex and ancestry prominently in the dashboard | Probe: ridge + sex + ancestry |
## Review
- Re-read at the end of each phase. Add new risks as they appear; never delete a risk, mark it closed with the reason.