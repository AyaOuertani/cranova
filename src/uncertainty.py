import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from baselines import FIG, N_FOLDS, N_REPEATS, TAB, encode_sex_ancestry, load_development_data, make_ridge

LEVELS = [0.5, 0.8, 0.9]
METHODS = ["Per distance", "Per distance + group"]
SERIES_COLOR = "#52514e"

#Quantile of absolute residuals with the finite-sample correction ceil((n+1)*level)/n
def conformal_quantile(abs_residuals, level):
    n = abs_residuals.shape[0]
    k = min(n, int(np.ceil((n + 1) * level)))
    return np.sort(abs_residuals, axis=0)[k-1]

#Half-width q for each validation subject and each distance, for both methods
def interval_half_widths(abs_residuals, train_groups, val_groups, level):
    per_distance = np.tile(conformal_quantile(abs_residuals, level), (len(val_groups), 1))
    per_group = np.zeros_like(per_distance)
    for group in np.unique(val_groups):
        in_group = train_groups == group
        per_group[val_groups == group] = conformal_quantile(abs_residuals[in_group], level)
    return {"Per distance": per_distance, "Per distance + group": per_group}

def run_cross_validation(xv, yv, groups):
    """Return a long table: one row per repeat, method, level, subject with coverage and mean width."""
    rows = []
    for rep in range(N_REPEATS):
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)  # same folds as everywhere else
        for train, val in folds.split(xv, groups):
            model = make_ridge().fit(xv[train], yv[train])
            oof = cross_val_predict(make_ridge(), xv[train], yv[train], cv=KFold(5, shuffle=True, random_state=rep))
            abs_residuals = np.abs(yv[train] - oof)
            prediction = model.predict(xv[val])
            for level in LEVELS:
                half = interval_half_widths(abs_residuals, groups[train], groups[val], level)
                for method in METHODS:
                    inside = np.abs(yv[val] - prediction) <= half[method]
                    for k, subject in enumerate(val):
                        rows.append((rep, method, level, subject, inside[k].mean(), 2 * half[method][k].mean()))
    return pd.DataFrame(rows, columns=["repeat", "method", "level", "subject", "coverage", "width_mm"])



def plot_honesty_check(overall, by_group):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {"Per distance": ("o", "-"), "Per distance + group": ("s", "--")}
    for method, (marker, line) in styles.items():
        d = overall[overall["method"] == method]
        left.plot(d["level"], d["coverage"], marker=marker, linestyle=line, color=SERIES_COLOR, label=method)
    left.plot([0.4, 1], [0.4, 1], color="#0b0b0b", linewidth=0.8)
    left.text(0.52, 0.47, "perfect honesty", fontsize=8, rotation=34)
    left.set_xlabel("Promised coverage")
    left.set_ylabel("Achieved coverage (held-out subjects)")
    left.set_title("Are the intervals honest?", fontsize=9)
    left.legend(fontsize=8, frameon=False)
    d = by_group[by_group["level"] == 0.9]
    groups = sorted(d["group"].unique())
    pos = np.arange(len(groups))[::-1]
    for method, (marker, _) in styles.items():
        values = d[d["method"] == method].set_index("group").loc[groups, "coverage"]
        offset = 0.12 if method == "Per distance" else -0.12
        right.plot(values, pos + offset, marker, color=SERIES_COLOR, label=method, markersize=6)
    right.axvline(0.9, color="#0b0b0b", linewidth=0.9)
    right.set_yticks(pos, groups, fontsize=8)
    right.set_xlabel("Achieved coverage of the 90% intervals")
    right.set_title("By ancestry and sex (promise = 0.90)", fontsize=9)
    right.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_uncertainty_honesty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    xv = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)])
    yv, group_labels = y.to_numpy(), groups.to_numpy()
    table = run_cross_validation(xv, yv, group_labels)
    table["group"] = group_labels[table["subject"]]
    overall = table.groupby(["method", "level"])[["coverage", "width_mm"]].mean().reset_index()
    by_group = table.groupby(["method", "level", "group"])[["coverage", "width_mm"]].mean().reset_index()
    overall.round(3).to_csv(TAB / "uncertainty_coverage_overall.csv", index=False)
    by_group.round(3).to_csv(TAB / "uncertainty_coverage_by_group.csv", index=False)
    plot_honesty_check(overall, by_group)
    print(overall.round(3).to_string(index=False))
    print("\n90% intervals by group:")
    print(by_group[by_group["level"] == 0.9].pivot(index="group", columns="method", values=["coverage", "width_mm"]).round(3).to_string())