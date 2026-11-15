import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict

from baselines import (N_FOLDS, N_REPEATS, TAB, encode_sex_ancestry, load_development_data, make_ridge,
                       per_subject_mae)
from experiments_classical import compare_with_best_baseline
from experiments_mlp import GRID, N_ENSEMBLE, make_mlp

#Ridge base + MLP correction learned from out-of-fold ridge residuals of the training fold.
def fit_and_predict(x_train, y_train, x_val):
    base = make_ridge().fit(x_train, y_train)
    oof = cross_val_predict(make_ridge(), x_train, y_train, cv=KFold(5, shuffle=True, random_state=0))
    residual = y_train - oof
    search = GridSearchCV(make_mlp(seed=0), {"regressor__" + k: v for k, v in GRID.items()}, cv=3,
                          scoring="neg_mean_absolute_error", n_jobs=2)
    search.fit(x_train, residual)
    best = {k.replace("regressor__mlpregressor__", ""): v for k, v in search.best_params_.items()}
    members = [make_mlp(seed, best["hidden_layer_sizes"], best["alpha"]).fit(x_train, residual)
               for seed in range(N_ENSEMBLE)]
    correction = np.mean([m.predict(x_val) for m in members], axis=0)
    return base.predict(x_val) + correction, best, np.abs(correction).mean()


if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    xv = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)])
    yv = y.to_numpy()
    preds = np.zeros((N_REPEATS, len(xv), yv.shape[1]))
    chosen, size_of_correction = [], []
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)  # same folds as the baselines
        for train, val in folds.split(xv, groups):
            preds[rep, val], best, size = fit_and_predict(xv[train], yv[train], xv[val])
            chosen.append(best)
            size_of_correction.append(size)
        print(f"repeat {rep + 1}/{N_REPEATS} done", flush=True)
    mae = per_subject_mae({"Residual MLP": preds}, yv)
    mae.round(4).to_csv(TAB / "e3_per_subject_mae_cv.csv")
    result = compare_with_best_baseline(mae, names=["Residual MLP"])
    result.round(3).to_csv(TAB / "e3_residual_mlp_results_cv.csv")
    print(result.round(3).to_string())
    print(f"\nAverage size of the MLP correction: {np.mean(size_of_correction):.3f} mm")
    print("Chosen configurations (count over all folds):")
    print(pd.Series([str(c) for c in chosen]).value_counts().to_string())