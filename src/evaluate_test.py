import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from baselines import ROOT, TAB, bootstrap_ci, make_ridge
from geometry import LANDMARKS, mean_face, reconstruct, rigid_align
from landmark_uncertainty import landmark_errors_on_template, radii_per_group
from load_data import CRANIAL_MEASURES, load_cranial, load_facial
from uncertainty import interval_half_widths

PRIMARY = [m for m in CRANIAL_MEASURES if m != "OBBL"]
ANCESTRIES = ["AFR", "ASN", "EUR"]
LEVEL = 0.9
MARKER = TAB / "TEST_SET_USED.txt"


#Inputs (16 measures + sex + ancestry), CT distances, ReFace distances and group labels.
def load_subjects(ids, split):
    meta = split.loc[ids]
    demographics = np.column_stack([(meta["sex"] == "M").astype(float)] +
                                   [(meta["ancestry"] == a).astype(float) for a in ANCESTRIES])
    x = np.hstack([load_cranial().loc[ids, PRIMARY].to_numpy(), demographics])
    y, r = load_facial("k").loc[ids].drop(columns="sex"), load_facial("r").loc[ids].drop(columns="sex")
    assert list(y.columns) == list(r.columns)
    return x, y.to_numpy(), r.to_numpy(), (meta["ancestry"] + "_" + meta["sex"]).to_numpy()


#Fit everything on `train`, evaluate once on `test`. Returns the result tables
def evaluate(train, test):
    (x_tr, y_tr, r_tr, g_tr), (x_te, y_te, r_te, g_te) = train, test
    model = make_ridge().fit(x_tr, y_tr)
    predictions = {"Mean predictor": np.tile(y_tr.mean(axis=0), (len(y_te), 1)),
                   "ReFace": r_te,
                   "ReFace bias-corrected": r_te - (r_tr - y_tr).mean(axis=0),
                   "Ridge (cranial only)": make_ridge().fit(x_tr[:, :len(PRIMARY)], y_tr).predict(x_te[:, :len(PRIMARY)]),
                   "Ridge + sex + ancestry (final model)": model.predict(x_te)}
    mae = pd.DataFrame({name: np.abs(p - y_te).mean(axis=1) for name, p in predictions.items()})
    final = mae["Ridge + sex + ancestry (final model)"].to_numpy()
    rows = []
    for name in mae:
        lo, hi = bootstrap_ci(mae[name].to_numpy())
        d = final - mae[name].to_numpy()
        d_lo, d_hi = bootstrap_ci(d)
        rows.append({"model": name, "MAE_mm": mae[name].mean(), "ci_low": lo, "ci_high": hi,
                     "final_minus_this_mm": d.mean(), "diff_ci_low": d_lo, "diff_ci_high": d_hi})
    accuracy = pd.DataFrame(rows).set_index("model")

    # 90% intervals for distances and spheres for landmarks, calibrated on the training subjects only
    oof = cross_val_predict(make_ridge(), x_tr, y_tr, cv=KFold(5, shuffle=True, random_state=0))
    half = interval_half_widths(np.abs(y_tr - oof), g_tr, g_te, LEVEL)["Per distance + group"]
    inside_interval = (np.abs(y_te - predictions["Ridge + sex + ancestry (final model)"]) <= half).mean(axis=1)
    template = mean_face(y_tr)
    radii = radii_per_group(landmark_errors_on_template(oof, y_tr, template), g_tr, g_te)
    landmark_err = landmark_errors_on_template(predictions["Ridge + sex + ancestry (final model)"], y_te, template)
    aligned_err = np.array([np.linalg.norm(rigid_align(reconstruct(p), reconstruct(t)) - reconstruct(t), axis=1)
                            for p, t in zip(predictions["Ridge + sex + ancestry (final model)"], y_te)])
    per_subject = pd.DataFrame({"group": g_te, "MAE_mm": final, "interval_coverage": inside_interval,
                                "interval_width_mm": 2 * half.mean(axis=1),
                                "landmark_error_mm": aligned_err.mean(axis=1),
                                "sphere_coverage": (landmark_err <= radii).mean(axis=1),
                                "sphere_radius_mm": radii.mean(axis=1)})
    by_group = per_subject.groupby("group").mean()
    by_group.loc["all"] = per_subject.drop(columns="group").mean()
    by_landmark = pd.DataFrame({"landmark_error_mm": aligned_err.mean(axis=0),
                                "sphere_coverage": (landmark_err <= radii).mean(axis=0)}, index=LANDMARKS)
    return accuracy, by_group, by_landmark


if __name__ == "__main__":
    final_run = "--final" in sys.argv
    split = pd.read_csv(ROOT / "splits" / "split_v1.csv", index_col="id")
    dev = split.index[split["role"] == "dev"]
    if final_run:
        if MARKER.exists():
            sys.exit(f"The test set was already used ({MARKER.read_text().strip()}). It is evaluated only once.")
        train_ids, test_ids, prefix = dev, split.index[split["role"] == "test"], "test"
    else:
        strata = split.loc[dev, "ancestry"] + "_" + split.loc[dev, "sex"]
        train_ids, test_ids = train_test_split(dev, test_size=0.2, stratify=strata, random_state=123)
        prefix = "dryrun"
    accuracy, by_group, by_landmark = evaluate(load_subjects(train_ids, split), load_subjects(test_ids, split))
    accuracy.round(3).to_csv(TAB / f"{prefix}_accuracy.csv")
    by_group.round(3).to_csv(TAB / f"{prefix}_by_group.csv")
    by_landmark.round(3).to_csv(TAB / f"{prefix}_by_landmark.csv")
    if final_run:
        MARKER.write_text(f"used on {datetime.now():%Y-%m-%d %H:%M}; train {len(train_ids)}, test {len(test_ids)}\n")
    print(f"{'TEST SET' if final_run else 'DRY RUN (development subjects only)'}: "
          f"trained on {len(train_ids)}, evaluated on {len(test_ids)}\n")
    print(accuracy.round(3).to_string())
    print("\nBy group:")
    print(by_group.round(3).to_string())
    print("\nBy landmark:")
    print(by_landmark.round(3).to_string())