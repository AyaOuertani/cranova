import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from baselines import (N_FOLDS, N_REPEATS, ROOT, TAB, bootstrap_ci, encode_sex_ancestry,
                       load_development_data, per_subject_mae)
from experiments_classical import compare_with_best_baseline
from experiments_mlp import GRID, N_ENSEMBLE, make_mlp

N_FACIAL = 66

#Error on the facial distances only (the first 66 outputs)
def facial_mae(y_true, y_pred):
    return np.abs(y_true[:, :N_FACIAL] - y_pred[:, :N_FACIAL]).mean()

#Cleaned tissue depths (Phase 2 output) for the given subjects
def load_tissue_depth_for(subject_ids):
    return pd.read_csv(ROOT / "data" / "processed" / "tissue_depth_clean.csv", index_col="id").reindex(subject_ids)

#Train on facial + tissue-depth targets; return facial predictions only.
def fit_and_predict(x_train, y_train, td_train, x_val):
    has_td = ~np.isnan(td_train).all(axis=1)                 # subjects with no tissue depth at all: not used for training
    td = np.where(np.isnan(td_train), np.nanmedian(td_train, axis=0), td_train)[has_td]  # rare single gaps: training median
    x_fit, targets = x_train[has_td], np.hstack([y_train[has_td], td])
    scorer = make_scorer(facial_mae, greater_is_better=False)
    search = GridSearchCV(make_mlp(seed=0), {"regressor__" + k: v for k, v in GRID.items()}, cv=3,
                          scoring=scorer, n_jobs=2)
    search.fit(x_fit, targets)
    best = {k.replace("regressor__mlpregressor__", ""): v for k, v in search.best_params_.items()}
    members = [make_mlp(seed, best["hidden_layer_sizes"], best["alpha"]).fit(x_fit, targets)
               for seed in range(N_ENSEMBLE)]
    return np.mean([m.predict(x_val)[:, :N_FACIAL] for m in members], axis=0), best

if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    xv = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)])
    yv = y.to_numpy()
    tdv = load_tissue_depth_for(x.index).to_numpy()
    print(f"Dev subjects without any tissue depth: {int(np.isnan(tdv).all(axis=1).sum())}")
    preds = np.zeros((N_REPEATS, len(xv), yv.shape[1]))
    chosen = []
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)  # same folds as the baselines
        for train, val in folds.split(xv, groups):
            preds[rep, val], best = fit_and_predict(xv[train], yv[train], tdv[train], xv[val])
            chosen.append(best)
        print(f"repeat {rep + 1}/{N_REPEATS} done", flush=True)
    mae = per_subject_mae({"Multi-task MLP": preds}, yv)
    mae.round(4).to_csv(TAB / "e4_per_subject_mae_cv.csv")
    result = compare_with_best_baseline(mae, names=["Multi-task MLP"])
    single_task = pd.read_csv(TAB / "e2_per_subject_mae_cv.csv", index_col=0)["MLP"]
    diff = mae["Multi-task MLP"].to_numpy() - single_task.to_numpy()
    low, high = bootstrap_ci(diff)
    result["diff_vs_E2_single_task_mm"], result["E2_ci_low"], result["E2_ci_high"] = diff.mean(), low, high
    result.round(3).to_csv(TAB / "e4_multitask_mlp_results_cv.csv")
    print(result.round(3).to_string())
    print("\nChosen configurations (count over all folds):")
    print(pd.Series([str(c) for c in chosen]).value_counts().to_string())
    
    