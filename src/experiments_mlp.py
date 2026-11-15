import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import N_FOLDS, N_REPEATS, TAB, encode_sex_ancestry, load_development_data, per_subject_mae
from experiments_classical import compare_with_best_baseline

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_ENSEMBLE = 5
GRID = {"mlpregressor__hidden_layer_sizes": [(16,), (32,), (64, 32)],
        "mlpregressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}  # alpha = weight decay: 3 x 5 = 15 configurations

#Standardised inputs and targets
def make_mlp(seed, hidden=(32,), alpha=1.0):
    network = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True,
                           validation_fraction=0.15, n_iter_no_change=20, max_iter=500, random_state=seed)
    return TransformedTargetRegressor(make_pipeline(StandardScaler(), network), transformer=StandardScaler())

#Tune on the training fold, then avarage the predictions of an ensemble of diffrently seeded networks
def fit_and_predict(x_train, y_train, x_val):
    search = GridSearchCV(make_mlp(seed=0), {"regressor__" + k: v for k, v in GRID.items()}, cv=3,
                          scoring="neg_mean_absolute_error", n_jobs=2)
    search.fit(x_train, y_train)
    best = {k.replace("regressor__mlpregressor__", ""): v for k, v in search.best_params_.items()}
    members = [make_mlp(seed, best["hidden_layer_sizes"], best["alpha"]).fit(x_train, y_train)
               for seed in range(N_ENSEMBLE)]
    return np.mean([m.predict(x_val) for m in members], axis=0), best

if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    xv = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)])
    yv = y.to_numpy()
    preds = np.zeros((N_REPEATS, len(xv), yv.shape[1]))
    chosen = []
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)  # same folds as the baselines
        for train, val in folds.split(xv, groups):
            preds[rep, val], best = fit_and_predict(xv[train], yv[train], xv[val])
            chosen.append(best)
        print(f"repeat {rep + 1}/{N_REPEATS} done", flush=True)
    mae = per_subject_mae({"MLP": preds}, yv)
    mae.round(4).to_csv(TAB / "e2_per_subject_mae_cv.csv")
    result = compare_with_best_baseline(mae, names=["MLP"])
    reface = pd.read_csv(TAB / "baseline_per_subject_mae_cv.csv", index_col=0)["ReFace"]
    result["diff_vs_ReFace_mm"] = (mae["MLP"] - reface).mean()
    result.round(3).to_csv(TAB / "e2_mlp_results_cv.csv")
    print(result.round(3).to_string())
    print("\nChosen configurations (count over all folds):")
    print(pd.Series([str(c) for c in chosen]).value_counts().to_string())