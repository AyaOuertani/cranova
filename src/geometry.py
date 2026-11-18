from itertools import combinations
import numpy as np
from load_data import FACIAL_LANDMARK_NAMES

LANDMARKS = list(FACIAL_LANDMARK_NAMES)
PAIRS = list(combinations(range(len(LANDMARKS)), 2))
DISTANCE_NAMES = [f"{LANDMARKS[i]}-{LANDMARKS[j]}" for i, j in PAIRS]

#66 distances (column order = DISTANCE_NAMES) -> symetric 12 x 12 matrix
def distances_to_matrix(distances):
    matrix = np.zeros((len(LANDMARKS), len(LANDMARKS)))
    for value, (i, j) in zip(distances, PAIRS):
        matrix[i, j] = matrix[j, i] = value
    return matrix

#12 x 3 coordinates -> the 66 distances
def coordinates_to_distances(coordinates):
    return np.array([np.linalg.norm(coordinates[i] - coordinates[j]) for  i, j in PAIRS])

#Classical multidimensional scaling: the 3D points whose distances best match the matrix
def classical_mds(distance_matrix, dimensions=3):
    n = len(distance_matrix)
    centering = np.eye(n) - np.ones((n,n)) / n
    gram = -0.5 * centering @ (distance_matrix ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:dimensions]
    return vectors[:, top] * np.sqrt(np.clip(values[top], 0, None))

#Fixed axes: origin btw the eyes, x left->right eye, y up, z toward the nose tip (mirror fixed)
def orient_face(coordinates):
    index = {name: k for k, name in enumerate(LANDMARKS)}
    origin = (coordinates[index["exL"]] + coordinates[index["exR"]]) / 2
    x = coordinates[index["exL"]] - coordinates[index["exR"]]   # toward the subject's left, as in GNM
    x = x / np.linalg.norm(x)
    down = (coordinates[index["chL"]] + coordinates[index["chR"]]) / 2 - origin
    y = -(down - (down @ x) * x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    local = (coordinates - origin) @ np.column_stack([x, y, z])
    if local[index["prn"], 2] < 0:
        local[:, 2] *= -1
    return local

#66 distances -> oriented 12 x 3 coordinates
def reconstruct(distances):
    return orient_face(classical_mds(distances_to_matrix(distances)))

#Rotate and translate `moving` onto `fixed` (no scaling, no mirroring)
def rigid_align(moving, fixed):
    a, b = moving - moving.mean(axis=0), fixed - fixed.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, sign]) @ vt
    return a @ rotation + fixed.mean(axis=0 )

#Average oriented face of a set of subjects: the template that predicted faces are shown on
def mean_face(distance_rows):
    return np.mean([reconstruct(d) for d in distance_rows], axis = 0)

#66 distances -> 3D landmarks, rotated and moved to fit the template as closely as possible
def place_on_template(distances, template):
    return rigid_align(reconstruct(distances), template)
