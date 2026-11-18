import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from baselines import N_FOLDS, N_REPEATS, TAB, bootstrap_ci, encode_sex_ancestry, load_development_data, make_ridge
from geometry import reconstruct, rigid_align
from gnm_correspondence import METRES_TO_MM, load_gnm_head

PENALTIES = [1, 3, 10, 30, 100, 300, 1000]
N_ALTERNATIONS = 10

#Average-head landmarks (12 x 3, mm) and how each head coefficient moves them
def landmark_model(gnm, vertex_ids):
    head = np.array([name.startswith("head") for name in gnm.identity_names])
    average = np.asarray(gnm.template_vertex_positions)[vertex_ids] * METRES_TO_MM
    basis = np.asarray(gnm.vertex_identity_basis)[head][:, vertex_ids] * METRES_TO_MM
    return average, basis

#Alternative : rifid placment of the head on the targets, then the best coefficients(ridge solve)
def fit_head(targets, average, basis, penalty):
    a = basis.reshape(len(basis), -1).T                 # 36 x 170
    gram = a.T @ a + penalty * np.eye(a.shape[1])
    beta = np.zeros(len(basis))
    for _ in range(N_ALTERNATIONS):
        model = average + np.tensordot(beta, basis, axes=1)
        placed = rigid_align(targets, model)            # targets moved into the head's frame
        beta = np.linalg.solve(gram, a.T @ (placed - average).ravel())
    fitted = average + np.tensordot(beta, basis, axes=1)
    return rigid_align(fitted, targets), beta

#Mean 3D landmark error (mm) after rigid alignment to the true landmarks
def landmark_error(points, true_distances):
    truth = reconstruct(true_distances)
    return np.linalg.norm(rigid_align(points, truth) - truth, axis=1).mean()

def errors_for_penalty(predicted, true, average, basis, penalty):
    return np.array([landmark_error(fit_head(reconstruct(p), average, basis, penalty)[0], t) for p, t in zip(predicted, true)])


if __name__ == "__main__":
    gnm = load_gnm_head()
    vertex_ids = pd.read_csv(TAB / "gnm_landmark_vertices.csv", index_col="landmark")["gnm_vertex"].to_numpy()
    average, basis = landmark_model(gnm, vertex_ids)
    x, y, r, groups, meta = load_development_data()
    xv, yv, labels = np.hstack([x.to_numpy(), encode_sex_ancestry(meta)]), y.to_numpy(), groups.to_numpy()
    raw, fitted, chosen = np.zeros((N_REPEATS, len(yv))), np.zeros((N_REPEATS, len(yv))), []
    for rep in range(N_REPEATS):
        for train, val in StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep).split(xv, labels):
            oof = cross_val_predict(make_ridge(), xv[train], yv[train], cv=KFold(5, shuffle=True, random_state=rep))
            penalty = min(PENALTIES, key=lambda lam: errors_for_penalty(oof, yv[train], average, basis, lam).mean())
            chosen.append(penalty)
            predicted = make_ridge().fit(xv[train], yv[train]).predict(xv[val])
            raw[rep, val] = [landmark_error(reconstruct(p), t) for p, t in zip(predicted, yv[val])]
            fitted[rep, val] = errors_for_penalty(predicted, yv[val], average, basis, penalty)
        print(f"repeat {rep + 1}/{N_REPEATS} done", flush=True)
    raw, fitted = raw.mean(axis=0), fitted.mean(axis=0)
    diff = fitted - raw
    low, high = bootstrap_ci(diff)
    result = pd.DataFrame({"landmark_error_mm": [raw.mean(), fitted.mean()]},
                          index=["Predicted landmarks (Phase 4)", "After GNM fit"])
    result.round(3).to_csv(TAB / "gnm_fit_results_cv.csv")
    print(result.round(3).to_string())
    print(f"\nDifference (fit - predicted): {diff.mean():+.3f} mm, 95% CI [{low:+.3f}, {high:+.3f}]")
    print("Chosen penalties:", pd.Series(chosen).value_counts().sort_index().to_dict())