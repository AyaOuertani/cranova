import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gnm.shape import gnm_landmarks, gnm_numpy

from baselines import FIG, TAB, load_development_data
from geometry import LANDMARKS, mean_face, rigid_align

METRES_TO_MM = 1000.0
IBUG_START = {"se": [27], "exL": [45], "exR": [36], "prn": [30], "acL": [35], "acR": [31], "sn": [33],
              "ls": [51], "li": [57], "sto": [62, 66], "chL": [54], "chR": [48]}
PAIRS = [("exL", "exR"), ("acL", "acR"), ("chL", "chR")]
SERIES_COLOR, INK = "#52514e", "#0b0b0b"

def load_gnm_head():
    return gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)

#GNM average head (all parameters zero), in millimeters
def average_head_mm(gnm):
    zeros = (np.zeros(gnm.identity_dim),np.zeros(gnm.expression_dim), np.zeros((gnm.num_joints, 3)), np.zeros(3))
    return np.asarray(gnm(*zeros)) * METRES_TO_MM

#Our 12 landmarks located from GNM's 68 iBUG landmarks (barycentric points on the mesh)
def ibug_start_positions(vertices_mm):
    ibug = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    points = (vertices_mm[ibug.indices] * ibug.weights[..., None]).sum(axis=1)
    return np.array([points[IBUG_START[name]].mean(axis=0) for name in LANDMARKS])

#Nearest skin vertex per landmark; for left/right pairs, the best exact mirror pair
def nearest_vertices(targets, vertices_mm, skin, mirror):
    distance = np.linalg.norm(vertices_mm[skin][None] - targets[:, None], axis=2)
    chosen = skin[distance.argmin(axis=1)]
    for left, right in PAIRS:
        i, j = LANDMARKS.index(left), LANDMARKS.index(right)
        cost = (np.linalg.norm(vertices_mm[skin] - targets[i], axis=1)
                + np.linalg.norm(vertices_mm[mirror[skin]] - targets[j], axis=1))
        best = skin[cost.argmin()]
        chosen[i], chosen[j] = best, mirror[best]
    return chosen

def calibrate_correspondence(vertices_mm, skin, mirror, data_face, max_iterations=10):
    positions = ibug_start_positions(vertices_mm)
    for _ in range(max_iterations):
        targets = rigid_align(data_face, positions)       # average data face placed on the GNM landmarks
        chosen = nearest_vertices(targets, vertices_mm, skin, mirror)
        if np.allclose(vertices_mm[chosen], positions):
            break
        positions = vertices_mm[chosen]
    offsets = np.linalg.norm(rigid_align(data_face, positions) - positions, axis=1)
    return chosen, offsets

#Front and side view of the GNM average head with the 12 corresponding vertices
def plot_landmarks_on_head(vertices_mm, skin, chosen):
    face_side = vertices_mm[skin]
    face_side = face_side[face_side[:, 2] > 60]           # front part of the head only, for readability
    fig, (front, side) = plt.subplots(1, 2, figsize=(9, 5.2))
    for ax, (h, label) in zip((front, side), ((0, "x (mm, subject's left)"), (2, "z (mm, toward the nose)"))):
        ax.scatter(face_side[:, h], face_side[:, 1], s=0.3, color="#c9c8c3", rasterized=True)
        ax.scatter(vertices_mm[chosen, h], vertices_mm[chosen, 1], s=22, color=INK, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    for name, k in zip(LANDMARKS, chosen):
        front.annotate(name, vertices_mm[k, [0, 1]], fontsize=7, color=SERIES_COLOR,
                       xytext=(4, 2), textcoords="offset points")
    front.set_ylabel("y (mm, up)")
    front.set_title("Front view", fontsize=9)
    side.set_title("Side view", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_gnm_landmarks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    gnm = load_gnm_head()
    vertices_mm = average_head_mm(gnm)
    skin = np.asarray(gnm.vertex_group_indices("skin_exterior"))
    mirror = np.asarray(gnm.mirror_indices)
    x, y, r, groups, meta = load_development_data()
    data_face = mean_face(y.to_numpy())
    start = ibug_start_positions(vertices_mm)
    start_offsets = np.linalg.norm(rigid_align(data_face, start) - start, axis=1)
    chosen, offsets = calibrate_correspondence(vertices_mm, skin, mirror, data_face)
    table = pd.DataFrame({"gnm_vertex": chosen, "offset_before_mm": start_offsets, "offset_after_mm": offsets},
                         index=pd.Index(LANDMARKS, name="landmark"))
    table.round(2).to_csv(TAB / "gnm_landmark_vertices.csv")
    plot_landmarks_on_head(vertices_mm, skin, chosen)
    print("Distance between the average data face and the GNM average head, per landmark (mm):")
    print(table.round(1).to_string())