import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from baselines import N_FOLDS, N_REPEATS, TAB, encode_sex_ancestry, load_development_data, make_ridge
from geometry import LANDMARKS, mean_face, place_on_template
from uncertainty import conformal_quantile


LEVEL = 0.9

#Per subject and landmark: distance(mm) btw predicted and true landmark, both placed on the template
def landmark_errors_on_template(predicted_distances, true_distances, template):
    return np.array([np.linalg.norm(place_on_template(p, template) - place_on_template(t, template), axis=1) for p, t in zip(predicted_distances, true_distances)])

#Radius for each validation subject and landmark, from the training errors of the subject's group
def radii_per_group(train_errors, train_groups, val_groups):
    radii = np.zeros((len(val_groups), train_errors.shape[1]))
    for group in np.unique(val_groups):
        radii[val_groups == group] = conformal_quantile(train_errors[train_groups == group], LEVEL)
    return radii

if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    xv, yv, labels = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)]), y.to_numpy(), groups.to_numpy()
    inside, radius = [], []
    for rep in range(N_REPEATS):
        for train, val in StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep).split(xv, labels):
            template = mean_face(yv[train])                      # built from training subjects only
            oof = cross_val_predict(make_ridge(), xv[train], yv[train], cv=KFold(5, shuffle=True, random_state=rep))
            train_errors = landmark_errors_on_template(oof, yv[train], template)
            predicted = make_ridge().fit(xv[train], yv[train]).predict(xv[val])
            val_errors = landmark_errors_on_template(predicted, yv[val], template)
            radii = radii_per_group(train_errors, labels[train], labels[val])
            inside.append(pd.DataFrame(val_errors <= radii, columns=LANDMARKS).assign(group=labels[val]))
            radius.append(pd.DataFrame(radii, columns=LANDMARKS).assign(group=labels[val]))
    inside, radius = pd.concat(inside), pd.concat(radius)
    table = pd.DataFrame({"coverage": inside[LANDMARKS].mean(), "radius_mm": radius[LANDMARKS].mean()})
    table.loc["all landmarks"] = [inside[LANDMARKS].to_numpy().mean(), radius[LANDMARKS].to_numpy().mean()]
    by_group = pd.DataFrame({"coverage": inside.groupby("group")[LANDMARKS].mean().mean(axis=1),
                             "radius_mm": radius.groupby("group")[LANDMARKS].mean().mean(axis=1)})
    table.round(3).to_csv(TAB / "landmark_uncertainty_cv.csv")
    by_group.round(3).to_csv(TAB / "landmark_uncertainty_by_group_cv.csv")
    print(f"Promised coverage: {LEVEL:.0%}")
    print(table.round(3).to_string())
    print("\nBy group (averaged over landmarks):")
    print(by_group.round(3).to_string())