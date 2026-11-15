import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import (N_FOLDS, N_REPEATS, TAB, bootstrap_ci, encode_sex_ancestry,
                       load_development_data, per_subject_mae)

CANDIDATES = ["Ridge", "Elastic net", "PLS"]
BEST_BASELINE = "Ridge + sex + ancestry"
MEANINGFUL_GAIN_MM = 0.10

#Each candidate is a pipeline: standardise inputs, then a model that tunes on the training fold
def make_candidate(name):
    if name == "Ridge":
        model = RidgeCV(alphas=np.logspace(-2, 3, 30))
    elif name == "Elastic net":
        model = MultiTaskElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 15),
                                      cv=3, max_iter=5000)
    elif name == "PLS":
        model = GridSearchCV(PLSRegression(scale=False), {"n_components": list(range(1, 11))},
                             cv=3, scoring="neg_mean_absolute_error")
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), model)

def cross_validated_predictions(x_with_demographics, y, groups):
    xv, yv = x_with_demographics, y.to_numpy()
    preds = {name: np.zeros((N_REPEATS, len(xv), yv.shape[1])) for name in CANDIDATES}
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)  # same folds as the baselines
        for train, val in folds.split(xv, groups):
            for name in CANDIDATES:
                preds[name][rep, val] = make_candidate(name).fit(xv[train], yv[train]).predict(xv[val])
    return preds, yv


#Paired difference to the best baseline, and whether it meets the selection rule
def compare_with_best_baseline(candidate_mae, names=None):
    reference = pd.read_csv(TAB / "baseline_per_subject_mae_cv.csv", index_col=0)[BEST_BASELINE]
    rows = []
    for name in (names or CANDIDATES):
        diff = (candidate_mae[name].to_numpy() - reference.to_numpy())
        lo, hi = bootstrap_ci(diff)
        rows.append({"model": name, "MAE_mm": candidate_mae[name].mean(), "diff_vs_best_baseline_mm": diff.mean(),
                     "ci_low": lo, "ci_high": hi,
                     "selected": bool(hi < 0 and diff.mean() <= -MEANINGFUL_GAIN_MM)})
    return pd.DataFrame(rows).set_index("model")

if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    xd = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)])
    preds, yv = cross_validated_predictions(xd, y, groups)
    mae = per_subject_mae(preds, yv)
    mae.round(4).to_csv(TAB / "e1_per_subject_mae_cv.csv")
    result = compare_with_best_baseline(mae)
    result.round(3).to_csv(TAB / "e1_classical_results_cv.csv")
    print(f"Best baseline: {BEST_BASELINE}")
    print(result.round(3).to_string())