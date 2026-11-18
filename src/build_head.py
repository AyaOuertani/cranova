import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import FIG, ROOT, TAB, encode_sex_ancestry, load_development_data, make_ridge
from fit_gnm import fit_head, landmark_model
from geometry import LANDMARKS, reconstruct, rigid_align
from gnm_correspondence import METRES_TO_MM, load_gnm_head

DISPLAY_PENALTY = 1
OUT = ROOT / "outputs" / "heads"
INK, SERIES_COLOR = "#0b0b0b", "#52514e"


#Train on every development subject except this one, then predict its 66 distances.
def predict_distances_left_out(subject, x, y, meta):
    xv = pd.DataFrame(np.hstack([x.to_numpy(), encode_sex_ancestry(meta)]), index=x.index)
    others = x.index != subject
    model = make_ridge().fit(xv[others], y[others])
    return model.predict(xv.loc[[subject]])[0]


#All vertices (mm) of the GNM head for the head-identity coefficients beta.
def full_head_mesh(gnm, beta):
    head = np.array([name.startswith("head") for name in gnm.identity_names])
    basis = np.asarray(gnm.vertex_identity_basis)[head]
    return (np.asarray(gnm.template_vertex_positions) + np.tensordot(beta, basis, axes=1)) * METRES_TO_MM


def write_obj(path, vertices, triangles):
    with open(path, "w") as f:
        f.write("# GNM head fitted to predicted landmarks (units: mm)\n")
        f.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in vertices)
        f.writelines(f"f {i + 1} {j + 1} {k + 1}\n" for i, j, k in triangles)   # OBJ counts from 1


#Triangles of the outer skin only (no eyeballs, teeth or mouth interior), for the figure.
def skin_triangles(gnm):
    on_skin = np.zeros(gnm.num_vertices, dtype=bool)
    on_skin[np.asarray(gnm.vertex_group_indices("skin_exterior"))] = True
    triangles = np.asarray(gnm.triangles)
    return triangles[on_skin[triangles].all(axis=1)]


#Front and side view: shaded skin, predicted landmarks (filled) and true landmarks (open).
def plot_head(vertices, triangles, predicted, true, subject):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.4))
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    #           axes,    horizontal, toward the viewer, title,        horizontal label
    views = [(axes[0], 0, np.array([0, 0, 1.0]), "Front view", "x (mm, subject's left)"),
             (axes[1], 2, np.array([-1.0, 0, 0]), "Side view", "z (mm, toward the nose)")]
    for ax, h, viewer, title, xlabel in views:
        facing = normals @ viewer > 0                       # hide triangles that face away (back-face culling)
        depth = corners.mean(axis=1) @ viewer
        order = np.argsort(depth[facing])                   # far triangles first, near ones on top
        shade = (normals @ viewer)[facing][order]           # simple light from the viewer
        ax.tripcolor(vertices[:, h], vertices[:, 1], triangles[facing][order], facecolors=shade,
                     cmap="Greys_r", vmin=-0.6, vmax=1.0, shading="flat", edgecolors="face",
                     linewidth=0, rasterized=True)
        ax.scatter(true[:, h], true[:, 1], s=40, facecolors="none", edgecolors=SERIES_COLOR, linewidths=1.2,
                   label="true (CT)", zorder=3)
        ax.scatter(predicted[:, h], predicted[:, 1], s=14, color=INK, label="predicted", zorder=4)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("y (mm, up)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"Subject {subject}: GNM head fitted to the predicted landmarks", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"fig12_head_{subject}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    x, y, r, groups, meta = load_development_data()
    subject = sys.argv[1] if len(sys.argv) > 1 else sorted(x.index)[0]
    assert subject in x.index, f"{subject} is not a development subject"
    gnm = load_gnm_head()
    vertex_ids = pd.read_csv(TAB / "gnm_landmark_vertices.csv", index_col="landmark")["gnm_vertex"].to_numpy()
    average, basis = landmark_model(gnm, vertex_ids)

    predicted_landmarks = reconstruct(predict_distances_left_out(subject, x, y, meta))
    _, beta = fit_head(predicted_landmarks, average, basis, DISPLAY_PENALTY)
    vertices = full_head_mesh(gnm, beta)
    head_landmarks = vertices[vertex_ids]
    predicted_on_head = rigid_align(predicted_landmarks, head_landmarks)     # same frame as the mesh
    true_on_head = rigid_align(reconstruct(y.loc[subject].to_numpy()), predicted_on_head)

    OUT.mkdir(parents=True, exist_ok=True)
    triangles = np.asarray(gnm.triangles)
    write_obj(OUT / f"{subject}_head.obj", vertices, triangles)
    error = np.linalg.norm(predicted_on_head - true_on_head, axis=1)
    pd.DataFrame(np.hstack([predicted_on_head, true_on_head, error[:, None]]), index=pd.Index(LANDMARKS, name="landmark"),
                 columns=["pred_x", "pred_y", "pred_z", "true_x", "true_y", "true_z", "error_mm"]
                 ).round(2).to_csv(OUT / f"{subject}_landmarks.csv")
    plot_head(vertices, skin_triangles(gnm), predicted_on_head, true_on_head, subject)
    print(f"Subject {subject} ({meta.loc[subject, 'ancestry']}, {meta.loc[subject, 'sex']})")
    print(f"Head mesh: {len(vertices)} vertices, {len(triangles)} triangles -> outputs/heads/{subject}_head.obj")
    print(f"Largest head coefficient: {np.abs(beta).max():.2f} standard deviations")
    print(f"Landmark error, predicted vs true: mean {error.mean():.2f} mm, max {error.max():.2f} mm "
          f"({LANDMARKS[error.argmax()]})")