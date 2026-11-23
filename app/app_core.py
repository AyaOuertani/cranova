import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  
import pandas as pd 
import plotly.graph_objects as go 
import streamlit as st  

from baselines import TAB  
from build_head import skin_triangles 
from evaluate_test import PRIMARY, load_subjects 
from fit_gnm import fit_head, landmark_model 
from geometry import LANDMARKS, reconstruct, rigid_align 
from gnm_correspondence import METRES_TO_MM, load_gnm_head 
from reconstruction_engine import load_bundle, predict_case 
from skull_placement import SKULL_FILE, build_reference_skull

DISPLAY_PENALTY = 1
from brand import BONE, FONT_UI, OBSIDIAN, STONE, TERRACOTTA  

ACCENT, TRUTH, SKIN = TERRACOTTA, BONE, STONE  

@st.cache_resource(show_spinner="Loading the model...")
def get_bundle():
    return load_bundle()

@st.cache_resource(show_spinner="Loadin the GNM head model...")
def get_head_model():
    gnm = load_gnm_head()
    ids = pd.read_csv(TAB / "gnm_landmark_vertices.csv", index_col="landmark")["gnm_vertex"].to_numpy()
    average, basis = landmark_model(gnm, ids)
    head = np.array([name.startswith("head") for name in gnm.identity_names])
    return {"ids": ids, "average": average, "basis": basis, "skin": skin_triangles(gnm),
            "triangles": np.asarray(gnm.triangles),
            "template": np.asarray(gnm.template_vertex_positions) * METRES_TO_MM,
            "full_basis": np.asarray(gnm.vertex_identity_basis)[head] * METRES_TO_MM}

@st.cache_data(show_spinner=False)
def demo_subjects():
    split = pd.read_csv(ROOT / "splits" / "split_v1.csv", index_col="id")
    ids = split.index[split["role"] == "test"]
    x, y, _, _ = load_subjects(ids, split)
    measurements = pd.DataFrame(x[:, :len(PRIMARY)], index=ids, columns=PRIMARY)
    return measurements, pd.DataFrame(y, index=ids), split.loc[ids, ["sex", "ancestry"]]

@st.cache_data(show_spinner="Reconstructing...")
def reconstruct_case(measurements: tuple, sex: str, ancestry: str):
    values = dict(zip(PRIMARY, measurements))
    result = predict_case(get_bundle(), values, sex, ancestry)
    head = get_head_model()
    _, beta = fit_head(result["landmarks"].to_numpy(), head["average"], head["basis"], DISPLAY_PENALTY)
    vertices = head["template"] + np.tensordot(beta, head["full_basis"], axes=1)
    landmarks = rigid_align(result["landmarks"].to_numpy(), vertices[head["ids"]])
    skull = None
    if SKULL_FILE.exists():
        face_points = dict(zip(LANDMARKS, vertices[head["ids"]]))
        skull = build_reference_skull(values["GOL"], values["XCB"], values["UFHT"], face_points)
    return {"result": result, "vertices": vertices, "landmarks": landmarks, "skull": skull}

def true_landmarks_on(predicted_landmarks, true_distances):
    return rigid_align(reconstruct(np.asarray(true_distances)), predicted_landmarks)

def sphere_mesh(centres, radii, resolution=10):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, resolution), np.linspace(0, np.pi, resolution))
    unit = np.column_stack([(np.cos(u) * np.sin(v)).ravel(), (np.sin(u) * np.sin(v)).ravel(), np.cos(v).ravel()])
    faces = []
    for a in range(resolution - 1):
        for b in range(resolution - 1):
            p = a * resolution + b
            faces += [[p, p + 1, p + resolution], [p + 1, p + resolution + 1, p + resolution]]
    points, triangles = [], []
    for k, (centre, radius) in enumerate(zip(centres, radii)):
        points.append(centre + radius * unit)
        triangles.append(np.array(faces) + k * len(unit))
    return np.vstack(points), np.vstack(triangles)

def mesh_trace(vertices, triangles, colour, opacity, name):
    return go.Mesh3d(x=vertices[:, 0], y=-vertices[:, 2], z=vertices[:, 1],   # plotly z = up; -z keeps it unmirrored
                     i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], color=colour, opacity=opacity,
                     name=name, flatshading=False, hoverinfo="skip", showlegend=True,
                     lighting=dict(ambient=0.45, diffuse=0.8, specular=0.15, roughness=0.8),
                     lightposition=dict(x=0, y=1000, z=300))


def points_trace(points, colour, name, symbol="circle", size=5, labels=None):
    return go.Scatter3d(x=points[:, 0], y=-points[:, 2], z=points[:, 1], mode="markers", name=name,
                        marker=dict(size=size, color=colour, symbol=symbol, line=dict(width=1, color=OBSIDIAN)),
                        text=labels, hovertemplate="%{text}<extra>" + name + "</extra>")


def scene_figure(case, view, face_opacity, show_spheres, truth=None):
    head = get_head_model()
    traces = []
    if view != "Face" and case["skull"] is not None:
        traces.append(mesh_trace(case["skull"][0], case["skull"][1], BONE, 1.0, "Reference skull"))
    if view != "Skull":
        traces.append(mesh_trace(case["vertices"], head["skin"], SKIN, face_opacity, "Predicted face"))
        if show_spheres:
            points, triangles = sphere_mesh(case["landmarks"], case["result"]["radius_90_mm"].to_numpy())
            traces.append(mesh_trace(points, triangles, ACCENT, 0.18, "90% landmark range"))
        traces.append(points_trace(case["landmarks"], ACCENT, "Predicted landmarks", labels=LANDMARKS))
        if truth is not None:
            traces.append(points_trace(truth, TRUTH, "True landmarks (CT)", symbol="circle-open", size=6,
                                       labels=LANDMARKS))
    fig = go.Figure(traces)
    hidden = dict(visible=False, showbackground=False)
    fig.update_layout(scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="data",
                                 camera=dict(eye=dict(x=0.6, y=-1.5, z=0.15), up=dict(x=0, y=0, z=1))),
                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=600,
                      legend=dict(orientation="h", y=0.02, x=0.02, font=dict(color="#9C9187", size=11, family=FONT_UI)),
                      uirevision="keep")       # keep the user's rotation when inputs change
    return fig


def obj_text(vertices, triangles, comment):
    lines = [f"# {comment} (units: mm)"]
    lines += [f"v {a:.3f} {b:.3f} {c:.3f}" for a, b, c in vertices]
    lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in triangles]
    return "\n".join(lines)
