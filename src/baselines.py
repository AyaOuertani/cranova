from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_data import CRANIAL_MEASURES, load_cranial, load_facial

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

N_REPEATS, N_FOLDS, N_BOOTSTRAP = 10, 5, 10_000
PRIMARY = [m for m in CRANIAL_MEASURES if m != "OBBL"]
MODELS = ["Mean predictor", "ReFace", "ReFace bias-corrected", "Ridge regression", "Ridge + sex + ancestry"]

#Return cranial inputs X, CT facial distances Y, ReFace distances R and group labels.
def load_development_data():
    split = pd.read_csv(ROOT / "splits" / "split_v1.csv", index_col="id")
    dev = split.index[split["role"] == "dev"]
    x = load_cranial().loc[dev, PRIMARY]
    y = load_facial("k").loc[dev].drop(columns="sex")
    r = load_facial("r").loc[dev].drop(columns="sex")
    assert list(y.columns) == list(r.columns), "facial versions must share the same distances"
    groups = split.loc[dev, "ancestry"] + "_" + split.loc[dev, "sex"]
    return x, y, r, groups, split.loc[dev]

#Numeric sex (1 = male) and one-hot ancestry columns, as extra model inputs
def encode_sex_ancestry(meta):
    sex = (meta["sex"] == "M").astype(float).to_numpy()[:, None]
    ancestry = pd.get_dummies(meta["ancestry"]).to_numpy(float)
    return np.hstack([sex, ancestry])

#Standardise the inputs, then ridge regression; the penalty is chosen inside the training fold.
def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))

#Out-of-fold predictions of every model, for every repeat: dict name -> (repeats, subjects, 66)
def cross_validated_predictions(x, y, r, groups, demographics):
    n = len(x)
    preds = {name: np.zeros((N_REPEATS, n, y.shape[1])) for name in MODELS}
    xv, yv, rv = x.to_numpy(), y.to_numpy(), r.to_numpy()
    xd = np.hstack([xv, demographics])  # cranial measures + sex + ancestry
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)
        for train, val in folds.split(xv, groups):
            preds["Mean predictor"][rep, val] = yv[train].mean(axis=0)
            preds["ReFace"][rep, val] = rv[val]
            bias = (rv[train] - yv[train]).mean(axis=0)  # learned on the training folds only
            preds["ReFace bias-corrected"][rep, val] = rv[val] - bias
            preds["Ridge regression"][rep, val] = make_ridge().fit(xv[train], yv[train]).predict(xv[val])
            preds["Ridge + sex + ancestry"][rep, val] = (
                make_ridge().fit(xd[train], yv[train]).predict(xd[val]))
    return preds, yv

#MAE over the 66 distances for each subject, averaged over the repeats
def per_subject_mae(preds, yv):
    return pd.DataFrame({name: np.abs(p - yv).mean(axis=2).mean(axis=0) for name, p in preds.items()})

#95% CI of the mean, resampling subjects.
def bootstrap_ci(values, seed=0):
    rng=np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

#Overall table (MAE, RMSE, paired difference to ReFace) and subgroup MAE table
def summarise(preds, yv, subject_mae, meta):
    rows = []
    for name in MODELS:
        rmse = np.mean([np.sqrt(((preds[name][k] - yv) ** 2).mean()) for k in range(N_REPEATS)])
        lo, hi = bootstrap_ci(subject_mae[name].to_numpy())
        diff = (subject_mae[name] - subject_mae["ReFace"]).to_numpy()
        d_lo, d_hi = bootstrap_ci(diff)
        rows.append({"model": name, "MAE_mm": subject_mae[name].mean(),
                     "MAE_ci_low": lo, "MAE_ci_high": hi, "RMSE_mm": rmse,
                     "diff_vs_ReFace_mm": diff.mean(), "diff_ci_low": d_lo, "diff_ci_high": d_hi})
    overall = pd.DataFrame(rows).set_index("model")
    sub = pd.concat({"ancestry": subject_mae.groupby(meta["ancestry"].to_numpy()).mean(),
                     "sex": subject_mae.groupby(meta["sex"].to_numpy()).mean()})
    return overall, sub

def plot_baseline_comparison(overall):
    colour = "#52514e"  # single neutral colour: the models are labelled on the axis
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    panels = [
        (left, overall, "MAE_mm", "MAE_ci_low", "MAE_ci_high",
         "MAE over 66 facial distances (mm), 95% CI", "Cross-validated error (development set)"),
        (right, overall.drop(index="ReFace"), "diff_vs_ReFace_mm", "diff_ci_low", "diff_ci_high",
         "Difference in MAE vs ReFace (mm): negative = better than ReFace",
         "Paired difference per subject, 95% CI"),
    ]
    for ax, data, centre, low, high, xlabel, title in panels:
        pos = np.arange(len(data))[::-1]  # first model at the top
        ax.errorbar(data[centre], pos, xerr=[data[centre] - data[low], data[high] - data[centre]],
                    fmt="o", color=colour, ecolor=colour, capsize=3, markersize=6)
        ax.set_yticks(pos, data.index, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
    right.axvline(0, color="#0b0b0b", linewidth=0.9)  # zero = no difference to ReFace
    fig.tight_layout()
    fig.savefig(FIG / "fig09_baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    print(f"Development subjects: {len(x)} | inputs: {x.shape[1]} | targets: {y.shape[1]}")
    preds, yv = cross_validated_predictions(x, y, r, groups, encode_sex_ancestry(meta))
    subject_mae = per_subject_mae(preds, yv)
    overall, sub = summarise(preds, yv, subject_mae, meta)
    overall.round(3).to_csv(TAB / "baseline_results_cv.csv")
    sub.round(3).to_csv(TAB / "baseline_results_by_group_cv.csv")
    subject_mae.round(4).to_csv(TAB / "baseline_per_subject_mae_cv.csv")
    plot_baseline_comparison(overall)
    print(overall.round(3).to_string())
    print()
    print(sub.round(2).to_string())