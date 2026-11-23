import numpy as np
import pandas as pd 
from scipy.optimize import minimize
from scipy.spatial import KDTree

from baselines import ROOT

SKULL_FILE = ROOT / "data" / "external" / "skull" / "skull_LP_UVs.obj"
MIDLINE_TOLERANCE = 0.03    

#reads the 3D file into points and triangles
def read_obj(path):
    vertices, triangles = [],[]
    for line in open(path):
        if line.startswith("v "):
            vertices.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            triangles += [[ids[0], ids[k], ids[k+1]] for k in range(1, len(ids)-1)]
    return np.array(vertices), np.array(triangles)

#finds four bone points on the midline (back of the head, forehead, nasion, upper jaw) plus the skull's width
def skull_landmarks(vertices):
    midline = np.where(np.abs(vertices[:, 0]) < MIDLINE_TOLERANCE)[0]
    y, z = vertices[midline, 1], vertices[midline, 2]

    def first(mask, values):
        return midline[mask][values[mask].argmax()]
    opisthocranion = first(y > -0.2, -z)                            # most posterior point of the vault
    glabella = first((y > 0.15) & (y < 0.6) & (z > 0.8), z)         # most anterior point of the forehead
    nasion = first((y > 0) & (y < vertices[glabella, 1]) & (z > 0.88), -z)   # deepest point below glabella
    prosthion = first((y > -0.52) & (y < -0.3) & (z > 0.9), z)      # most anterior point of the upper jaw
    vault = vertices[vertices[:, 1] > 0.2]                          # above the zygomatic arches
    return {"opisthocranion": opisthocranion, "glabella": glabella, "nasion": nasion, "prosthion": prosthion, "breadth": np.ptp(vault[:, 0])}

#stretches each axis to the entered GOL, XCB and UFHT.
def scale_to_measurements(vertices, marks, gol, xcb, ufht):
    length = vertices[marks["glabella"], 2] - vertices[marks["opisthocranion"], 2]
    height = vertices[marks["nasion"], 1] - vertices[marks["prosthion"], 1]
    return vertices * np.array([xcb / marks["breadth"], ufht / height, gol / length])

def mean_tissue_depths():
    return pd.read_csv(ROOT / "data" / "processed" / "tissue_depth_clean.csv", index_col="id").mean()

DEPTH_AT = {"se": "n", "sn": "sn", "ls": "ls", "li": "li"}

def tilt_and_move(skull_mm, pivot, params):
    angle, dy, dz = params
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (skull_mm - pivot) @ rotation.T + pivot + np.array([0.0, dy, dz])

#tilts and shifts the skull until the skin depths match your data's averages.
def place_under_face(skull_mm, marks, face_points, depths):
    start = face_points["se"] - np.array([0.0, 0.0, depths["se"]])
    skull_mm = skull_mm + (start - skull_mm[marks["nasion"]])
    skull_mm[:, 0] += face_points["se"][0] - start[0]
    pivot = skull_mm[marks["nasion"]].copy()
    skin = np.array([face_points[k] for k in DEPTH_AT])
    wanted = np.array([depths[k] for k in DEPTH_AT])

    def mismatch(params):
        bone = KDTree(tilt_and_move(skull_mm, pivot, params))
        return ((bone.query(skin)[0] - wanted) ** 2).sum()
    best = minimize(mismatch, x0=np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3})
    return tilt_and_move(skull_mm, pivot, best.x)

def build_reference_skull(gol, xcb, ufht, face_points):
    vertices, triangles = read_obj(SKULL_FILE)
    marks = skull_landmarks(vertices)
    scaled = scale_to_measurements(vertices, marks, gol, xcb, ufht)
    average = mean_tissue_depths()
    depths = {k: average[column] for k, column in DEPTH_AT.items()}
    return place_under_face(scaled, marks, face_points, depths), triangles