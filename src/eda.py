from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib

from load_data import (CRANIAL_MEASURES, CRANIAL_NAMES, ancestry_from_id, load_cranial, load_facial, load_tissue_depth)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

split = pd.read_csv(ROOT / "splits" / "split_v1.csv", index_col="id")
DEV = split.index[split["role"] == "dev"]

ANC_COLOR = {"AFR": "#D0957E", "ASN": "#E21D28", "EUR": "#003399", "HIS": "#A23F7E"}
SEX_COLOR = {"F": "#BBA1D9", "M": "#2F6F8F"}  # sex: used only in figures without ancestry colours
INK, INK2 = "#0b0b0b", "#52514e"
SERIES_COLOR = "#9314a1"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})

PRIMARY = [m for m in CRANIAL_MEASURES if m != "OBBL"] 

#Subjects per ancestry and sex in each dataset
def plot_sampel_composition():
    datasets = {"Cranial": load_cranial(), "Facial": load_facial("k"), "Tissue depth": load_tissue_depth()}
    order = ["AFR", "ASN", "EUR", "HIS", "HWF/HWM"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (name, df) in zip(axes, datasets.items()):
        anc = pd.Series([ancestry_from_id(i) for i in df.index], index=df.index)
        counts = pd.crosstab(anc, df["sex"]).reindex(order).fillna(0).astype(int)
        x = np.arange(len(order))
        for k, sex in enumerate(["F", "M"]):
            bars = ax.bar(x + (k - 0.5) * 0.4, counts[sex], 0.38, color=SEX_COLOR[sex],
                          label="Female" if sex == "F" else "Male", linewidth=0)
            for b, v in zip(bars, counts[sex]):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=7)
        ax.set_xticks(x, order, fontsize=8)
        ax.set_title(f"{name} (n = {len(df)})", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Subjects")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig01_sample_composition.png")
    plt.close(fig)


#Distributuion of the 16 primary cranial measures by ancestry
def plot_cranial_distributions_by_ancestry():
    c = load_cranial().loc[DEV]
    anc = ["AFR", "ASN", "EUR"]
    fig, axes = plt.subplots(4, 4, figsize=(10, 8.5))
    for ax, m in zip(axes.ravel(), PRIMARY):
        data = [c.loc[c["ancestry"] == a, m] for a in anc]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, tick_labels=anc,
                        flierprops=dict(marker="o", markersize=2.5, markeredgewidth=0.5),
                        medianprops=dict(color=INK, linewidth=1.2),
                        whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
        for patch, a in zip(bp["boxes"], anc):
            patch.set(facecolor=ANC_COLOR[a], edgecolor=INK2, linewidth=0.6, alpha=0.9)
        ax.set_title(f"{m}: {CRANIAL_NAMES[m]}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", visible=False)
    fig.supylabel("mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_cranial_distributions.png")
    plt.close(fig)
    summary = c[PRIMARY].agg(["count", "mean", "std", "min", "max"]).T.round(2)
    summary.insert(0, "name", [CRANIAL_NAMES[m] for m in summary.index])
    summary.to_csv(TAB / "cranial_summary_dev.csv")

#Scale/variability of the 66 facial distances and how many directions they really contain
def plot_facial_distance_variability():
    d = load_facial("k").loc[DEV].drop(columns="sex")
    mean, sd = d.mean(), d.std()
    pca = PCA().fit(d)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95, n99 = int(np.argmax(cum >= 0.95)) + 1, int(np.argmax(cum >= 0.99)) + 1

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.scatter(mean, sd, s=18, color=SERIES_COLOR, edgecolor="white", linewidth=0.6)
    for name in sd.sort_values(ascending=False).index[:4]:
        a.annotate(name, (mean[name], sd[name]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    a.set_xlabel("Mean distance (mm)")
    a.set_ylabel("Standard deviation across subjects (mm)")
    a.set_title("66 facial distances: size vs variability", fontsize=9)

    k = np.arange(1, 16)
    b.plot(k, cum[:15] * 100, color=ANC_COLOR["AFR"], linewidth=2, marker="o", markersize=5,
           markeredgecolor="white")
    b.axhline(95, color=INK2, linewidth=0.8, linestyle="--")
    b.annotate(f"{n95} components reach 95%", (n95, cum[n95 - 1] * 100), xytext=(n95 + 1.5, 88),
               fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    b.set_xlabel("Number of principal components")
    b.set_ylabel("Cumulative variance explained (%)")
    b.set_title("How many independent directions do the 66 distances contain?", fontsize=9)
    b.set_ylim(40, 101)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_facial_variability.png")
    plt.close(fig)

    out = pd.DataFrame({"mean_mm": mean, "sd_mm": sd, "min_mm": d.min(), "max_mm": d.max()}).round(2)
    out.to_csv(TAB / "facial_summary_dev.csv")
    print(f"Facial PCA: {n95} components reach 95% variance, {n99} reach 99%")
    print("Largest SD distances:", ", ".join(f"{n} ({sd[n]:.1f})" for n in sd.sort_values(ascending=False).index[:4]))


DIVRGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "diverging", ["#2a78d6", "#f2f2f0", "#eb6834"])

#Correlation between the 16 cranial measures: which ones carry overlapping information.
def plot_cranial_correlation_matrice():
    corr = load_cranial().loc[DEV, PRIMARY].corr()
    fig, ax =plt.subplots(figsize=(8, 7))
    im =ax.imshow(corr, cmap=DIVRGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(PRIMARY)), PRIMARY, rotation=90, fontsize=8)
    ax.set_yticks(range(len(PRIMARY)), PRIMARY, fontsize=8)
    ax.grid(False)
    for i in range(len(PRIMARY)):
        for j in range(len(PRIMARY)):
            ax.text(j,i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    
    fig.colorbar(im, ax=ax, shrink=0.8, label="Person correlation")
    ax.set_title("Correlation btween cranial measurments", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_cranial_correlation_matrix.png")
    plt.close(fig)
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = pairs[pairs.abs() > 0.7].sort_values(ascending=False)
    print(f"Cranial pairs with |r| > 0.7: {len(strong)}",
          "; ".join(f"{a}-{b} ({v:.2f})" for (a, b), v in strong.items()))

#Correlation of each cranial measue with each facial distance
def plot_cranial_to_facial_correlations():
    x = load_cranial().loc[DEV, PRIMARY]
    y = load_facial("k").loc[DEV].drop(columns="sex")
    r = y.apply(lambda col: x.corrwith(col))  # rows: cranial measures, columns: facial distances
    r = r[r.abs().max().sort_values(ascending=False).index]  # strongest facial distances first
    fig, ax = plt.subplots(figsize=(14, 4.8))
    im = ax.imshow(r, cmap=DIVRGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(r.shape[1]), r.columns, rotation=90, fontsize=5.5)
    ax.set_yticks(range(r.shape[0]), r.index, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Cranial measure (rows) vs facial distance (columns), sorted by strongest correlation",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_cranial_to_facial_correlations.png")
    plt.close(fig)
    r.round(3).to_csv(TAB / "cranial_facial_correlation_dev.csv")
    best = r.abs().max()
    print(f"Best single cranial predictor per facial distance: median |r| = {best.median():.2f}; "
          f"{(best > 0.5).sum()} of {len(best)} distances have |r| > 0.5")
    top = r.abs().stack().sort_values(ascending=False).head(5)
    print("Strongest pairs:", "; ".join(f"{c}~{f} ({r.loc[c, f]:.2f})" for (c, f), _ in top.items()))

#Where ReFace errs, compared with simply predicting each distance's mean
def plot_reface_error_profile():
    known = load_facial("k").loc[DEV].drop(columns="sex")
    reface = load_facial("r").loc[DEV].drop(columns="sex")
    error = reface - known
    reface_mae = error.abs().mean()
    mean_pred_mae = (known - known.mean()).abs().mean()  # in-sample: a descriptive reference only
    bias = error.mean()

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.scatter(mean_pred_mae, reface_mae, s=18, color=SERIES_COLOR, edgecolor="white", linewidth=0.6)
    top = max(mean_pred_mae.max(), reface_mae.max()) * 1.05
    a.plot([0, top], [0, top], color=INK2, linewidth=0.8, linestyle="--")
    a.text(top * 0.55, top * 0.62, "ReFace worse than\nthe mean", fontsize=8, color=INK2)
    closest = (reface_mae - mean_pred_mae).sort_values(ascending=False).index[:2]
    for name, dy in zip(closest, (6, -11)):  # the two distances where ReFace is closest to the mean
        a.annotate(name, (mean_pred_mae[name], reface_mae[name]), xytext=(6, dy),
                   textcoords="offset points", fontsize=7)
    a.set_xlabel("MAE of predicting the mean (mm)")
    a.set_ylabel("MAE of ReFace (mm)")
    a.set_title("Per distance: ReFace vs mean predictor", fontsize=9)
    b.hist(bias, bins=15, color=SERIES_COLOR, edgecolor="white", linewidth=0.6)
    b.axvline(0, color=INK, linewidth=0.8)
    b.set_xlabel("Mean signed error of ReFace (mm): negative = ReFace too small")
    b.set_ylabel("Number of distances")
    b.set_title("ReFace bias across the 66 distances", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_reface_error_profile.png")
    plt.close(fig)
    pd.DataFrame({"reface_mae_mm": reface_mae, "mean_predictor_mae_mm": mean_pred_mae,
                  "reface_bias_mm": bias}).round(3).to_csv(TAB / "reface_error_by_distance_dev.csv")
    worse = int((reface_mae > mean_pred_mae).sum())
    print(f"ReFace MAE {reface_mae.mean():.2f} mm vs mean predictor {mean_pred_mae.mean():.2f} mm "
          f"(dev subjects, in-sample mean); ReFace worse on {worse} of {len(reface_mae)} distances")


#Missing tissue-depth values per landmark.
def plot_missing_values():
    td = load_tissue_depth()
    td = td.loc[td.index.intersection(DEV)].drop(columns="sex")
    pct = (td.isna().mean() * 100).sort_values(ascending=False)
    shown = pct.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(shown.index, shown.values, color=SERIES_COLOR, height=0.6)
    for yv, v in enumerate(shown.values):
        ax.text(v + 0.8, yv, f"{v:.0f}%", va="center", fontsize=7)
    ax.set_xlabel("Subjects with a missing value (%)")
    ax.set_title(f"Tissue depth: 12 columns with most missing values (n = {len(td)})", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_missing_values_tissue_depth.png")
    plt.close(fig)
    c = load_cranial().loc[DEV, PRIMARY + ["OBBL"]]
    table = pd.concat({"cranial_missing_pct": c.isna().mean() * 100,
                       "tissue_depth_missing_pct": td.isna().mean() * 100}).round(1)
    table.to_csv(TAB / "missing_values_dev.csv")
    obbl = c.loc[c["OBBL"].isna()].index
    print(f"OBBL missing for {len(obbl)} dev subjects; tissue-depth columns with >5% missing: "
          f"{', '.join(pct[pct > 5].index)}")


#Range of soft-tissue depth at each landmark (development subjects with tissue-depth data)
def plot_tissue_depth_overview():
    td = load_tissue_depth()
    td = td.loc[td.index.intersection(DEV)].drop(columns="sex")
    order = td.median().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 10))
    bp = ax.boxplot([td[c].dropna() for c in order], orientation="horizontal", patch_artist=True,
                    tick_labels=order,
                    flierprops=dict(marker="o", markersize=2, markeredgewidth=0.4),
                    medianprops=dict(color=INK, linewidth=1.1), whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7))
    for patch in bp["boxes"]:
        patch.set(facecolor="#c9c8c2", edgecolor=INK2, linewidth=0.6)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Soft-tissue depth (mm)")
    ax.set_title(f"Tissue depth by landmark, sorted by median (n = {len(td)})", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_tissue_depth_overview.png")
    plt.close(fig)
    td.agg(["count", "mean", "std", "min", "max"]).T.round(2).to_csv(TAB / "tissue_depth_summary_dev.csv")


def flag_cranial_outliers(z_limit: float = 3.0):
    c = load_cranial().loc[DEV, ["ancestry", "sex"] + PRIMARY]
    groups = c.groupby(["ancestry", "sex"])
    z = (c[PRIMARY] - groups[PRIMARY].transform("mean")) / groups[PRIMARY].transform("std")
    flags = z.stack()
    flags = flags[flags.abs() > z_limit].rename("z_score").reset_index()
    flags.columns = ["id", "measure", "z_score"]
    flags["value_mm"] = [c.loc[i, m] for i, m in zip(flags["id"], flags["measure"])]
    flags.round(2).to_csv(TAB / "cranial_outlier_flags_dev.csv", index=False)
    print(f"Cranial values with |z| > {z_limit}: {len(flags)} of {z.size} "
          f"({len(flags) / z.size * 100:.2f}%)")


#List tissue-depth values that are extreme for their landmark (possible entry errors).
def flag_tissue_depth_outliers(robust_z_limit: float = 8.0):
    td = load_tissue_depth().drop(columns=["sex", "gn - z"])  # 'gn - z' is undocumented
    median = td.median()
    mad = (td - median).abs().median()
    robust_z = 0.6745 * (td - median) / mad
    flags = robust_z.stack()
    flags = flags[flags.abs() > robust_z_limit].rename("robust_z").reset_index()
    flags.columns = ["id", "landmark", "robust_z"]
    flags["value_mm"] = [td.loc[i, m] for i, m in zip(flags["id"], flags["landmark"])]
    flags["landmark_median_mm"] = [median[m] for m in flags["landmark"]]
    flags.round(2).to_csv(TAB / "tissue_depth_outlier_flags.csv", index=False)
    print(f"Tissue-depth values with robust |z| > {robust_z_limit}: {len(flags)}")
    print(flags.round(1).to_string(index=False))


if __name__ == "__main__":
    plot_sampel_composition()
    plot_cranial_distributions_by_ancestry()
    plot_facial_distance_variability()
    plot_cranial_correlation_matrice()
    plot_cranial_to_facial_correlations()
    plot_reface_error_profile()
    plot_missing_values()
    plot_tissue_depth_overview()
    flag_cranial_outliers()
    flag_tissue_depth_outliers()
    print("Figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("Tables:", sorted(p.name for p in TAB.glob("*.csv")))