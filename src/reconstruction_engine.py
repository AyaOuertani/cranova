from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from baselines import ROOT, make_ridge
from evaluate_test import ANCESTRIES, LEVEL, PRIMARY, load_subjects
from geometry import DISTANCE_NAMES, LANDMARKS, mean_face, place_on_template
from landmark_uncertainty import landmark_errors_on_template
from load_data import CRANIAL_NAMES
from uncertainty import conformal_quantile

BUNDLE = ROOT / "models" / "reconstruction_bundle.joblib"
SEXES = ["F", "M"]

def build_bundle(path=BUNDLE):
    split = pd.read_csv(ROOT / "splits"/ "split_v1.csv", index_col="id")
    dev = split.index[split["role"] == "dev"]
    x, y, _, groups = load_subjects(dev, split)
    oof = cross_val_predict(make_ridge(), x, y, cv=KFold(5, shuffle=True, random_state=0))
    template = mean_face(y)
    landmark_errors = landmark_errors_on_template(oof, y, template)
    bundle = {
        "model": make_ridge().fit(x, y),
        "half_width": {g: conformal_quantile(np.abs(y - oof)[groups == g], LEVEL) for g in np.unique(groups)},
        "radius": {g: conformal_quantile(landmark_errors[groups == g], LEVEL) for g in np.unique(groups)},
        "template": template,
        "ranges": pd.DataFrame(x[:, :len(PRIMARY)], columns=PRIMARY).agg(["min", "max"]),
        "n_training_subjects": len(dev)
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle

def load_bundle(path= BUNDLE):
    return joblib.load(path)

def input_warnings(bundle, measurements):
    ranges, warnings = bundle["ranges"], []
    for code in PRIMARY:
        value, low, high = measurements[code], ranges.loc["min", code], ranges.loc["max", code]
        if not low<= value <= high:
            warnings.append(f"{code} ({CRANIAL_NAMES[code]}) - {value:g} mm is outside the training range "
                            f"{low:.0f} - {high:.0f} mm")
    return warnings

def predict_case(bundle, measurments, sex, ancestry):
    if sex not in SEXES or ancestry not in ANCESTRIES:
        raise ValueError(f"sex must be one of {SEXES} and ancestry one of {ANCESTRIES}")
    inputs = [measurments[code] for code in PRIMARY] + [float(sex == "M")] + [float(ancestry == a) for a in ANCESTRIES]
    distances = bundle["model"].predict(np.array([inputs]))[0]
    group = f"{ancestry}_{sex}"
    half = bundle["half_width"][group]
    table = pd.DataFrame({"predicted_mm": distances, "low_90_mm": distances - half, "high_90_mm": distances + half},index=pd.Index(DISTANCE_NAMES, name="distance"))
    landmarks = place_on_template(distances, bundle["template"])
    return{"distances": table, "landmarks": pd.DataFrame(landmarks, index=LANDMARKS, columns=["x", "y", "z"]),
           "radius_90_mm": pd.Series(bundle["radius"][group], index=LANDMARKS),
           "warnings": input_warnings(bundle, measurments)}

if __name__ == "__main__":
    bundle = build_bundle()
    print(f"Bundle saved: {BUNDLE.relative_to(ROOT)} | trained on {bundle['n_training_subjects']} subjects")
    print("Groups:", sorted(bundle["half_width"]))