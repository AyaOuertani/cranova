# Phase 3: model experiments on the development set (219 subjects)

Protocol: repeated stratified 5-fold CV x 10 seeds, same folds for every model.
Metric: per-subject MAE over the 66 facial distances (mm).
Selection rule (fixed before running): a candidate replaces the best baseline only if it
improves MAE by at least 0.10 mm AND the 95% bootstrap CI of the paired difference excludes zero.
Best baseline: ridge + sex + ancestry (2.321 mm).

| Experiment | Model | MAE (mm) | Diff vs best baseline (mm) | 95% CI | Selected |
|---|---|---|---|---|---|
| E1 | Ridge | 2.321 | 0.000 | [0.000, 0.000] | No |
| E1 | Elastic net | 2.321 | -0.001 | [-0.005, 0.004] | No |
| E1 | PLS | 2.360 | +0.039 | [0.020, 0.058] | No |
| E2 | MLP (tuned, ensemble of 5) | 2.333 | +0.012 | [-0.023, 0.046] | No |
| E3 | Residual MLP (ridge + correction) | 2.323 | +0.002 | [0.000, 0.003] | No |
| E4 | Multi-task MLP (+ tissue depth targets) | 2.313 | -0.008 | [-0.041, 0.024] | No |
| E5 | Transfer learning | not applicable | - | - | - |

## E5: transfer learning, not applicable
No pretrained model exists that maps cranial measurements to facial distances (literature review,
pretrained-model survey). Published skull-to-face models use their own, mostly private, data and
different inputs. Fine-tuning a model from another domain would add parameters without adding
information. Recorded as not applicable, as the protocol allows.
Pretrained 3D head models are a different question (Phase 6: shape prior for the complete head).

## Notes for the report
- E2 tuning: a first run with weight decay up to 10 always selected the edge of the grid
  (MAE 2.248, -0.074 vs baseline, below the 0.10 mm rule). The grid was widened to 0.1-1000
  (15 configurations, within the budget of 30); the final result is 2.333. The narrower
  run's gain was not stable and is not used.
- E3: the tuning chose the strongest weight decay in every fold (average correction 0.08 mm):
  ridge residuals contain nothing the network could learn from 175 training subjects.
- E4 improves on E2 by 0.020 mm (CI [-0.033, -0.007]): real but far below the 0.10 mm threshold.

## Conclusion
No candidate meets the selection rule. Increasing model complexity (linear variants, tuned MLP,
residual MLP, multi-task MLP) did not improve held-out accuracy; the limiting factor is the
information in the cranial measurements, not the model class. The model carried forward is
ridge + sex + ancestry.