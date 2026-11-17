import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from baselines import TAB, encode_sex_ancestry, load_development_data, make_ridge
from geometry import (DISTANCE_NAMES, LANDMARKS, coordinates_to_distances, reconstruct, rigid_align)

#out of fold prediction, same folds as repeat 0 of all other experiments
def cross_validated_distance_predictions(xv, yv, groups):
    predictions = np.zeros_like(yv)
    for train, val in StratifiedKFold(5, shuffle=True, random_state=0).split(xv, groups):
        predictions[val] = make_ridge().fit(xv[train], yv[train]).predict(xv[val])
    return predictions

#Per subject and landmark: 3D position error(mm) after rigiD allignment to the true landmarks
def landmark_errors(predicted_distances, true_distances):
    errors = np.zeros((len(true_distances), len(LANDMARKS)))
    for k, (predictied, true) in enumerate(zip(predicted_distances, true_distances)):
        truth = reconstruct(true)
        aligned = rigid_align(reconstruct(predictied), truth)
        errors[k] = np.linalg.norm(aligned - truth, axis=1)
    return errors

if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    assert list(y.columns) == DISTANCE_NAMES, "distance columns must follow the landmark pair order"
    xv, yv = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)]), y.to_numpy()
    
    exactness = max(np.abs(coordinates_to_distances(reconstruct(d)) - d).max() for d in yv)
    print(f"Check on true distances: largest distance error after reconstruction = {exactness:.4f} mm")
    
    predicted = cross_validated_distance_predictions(xv, yv, groups.to_numpy())
    re_derived = np.array([coordinates_to_distances(reconstruct(p)) for p in predicted])
    print(f"Predicted distances: MAE {np.abs(predicted - yv).mean():.3f} mm | "
          f"after passing through 3D landmarks: {np.abs(re_derived - yv).mean():.3f} mm")


    errors = landmark_errors(predicted, yv)
    table = pd.DataFrame({"mean_error_mm": errors.mean(axis=0), "p90_error_mm": np.percentile(errors, 90, axis=0)},
                         index=LANDMARKS)
    table.loc["all landmarks"] = [errors.mean(), np.percentile(errors, 90)]
    table.round(3).to_csv(TAB / "landmark_error_by_landmark_cv.csv")
    print(table.round(2).to_string())
    by_group = pd.Series(errors.mean(axis=1)).groupby(groups.to_numpy()).mean().round(2)
    print("\nMean landmark error by group (mm):")
    print(by_group.to_string())