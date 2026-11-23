import numpy as np
import pandas as pd
import streamlit as st

from app_core import (PRIMARY, demo_subjects, get_bundle, get_head_model, obj_text, reconstruct_case,
                      scene_figure, true_landmarks_on)
from brand import label, note, stats
from load_data import CRANIAL_NAMES

VAULT = ["GOL", "XCB", "BBH", "BNL", "AUB", "WFB", "FRC", "PAC", "OCC"]
FACE = ["ZYB", "BPL", "UFHT", "MAB", "NLB", "NLHL", "OBHL"]
NEW_CASE = "New case"

bundle = get_bundle()
measurements, true_distance, demo_meta = demo_subjects()
typical = bundle["ranges"].mean().round(1)

case_file, stage = st.columns([1, 2.4], gap="large")
with case_file:
    label("Case file")
    demo = st.selectbox("Start from", [NEW_CASE] + list(measurements.index), label_visibility="collapsed",
                        help="Demo subjects come from the locked test set: never used for training.")
    start = typical if demo == NEW_CASE else measurements.loc[demo]
    start_sex = "F" if demo == NEW_CASE else demo_meta.loc[demo, "sex"]
    start_ancestry = "EUR" if demo == NEW_CASE else demo_meta.loc[demo, "ancestry"]
    with st.form("case", border=True):
        sex = st.segmented_control("Sex", ["F","M"], default=start_sex, key=f"sex_{demo}")
        ancestry = st.segmented_control("Ancestry", ["AFR", "ASN", "EUR"], default=start_ancestry, key=f"anc_{demo}")

        values = {}
        for title, codes in (("Cranial vault · mm", VAULT), ("Face · mm", FACE)):
            label(title)
            left, right = st.columns(2)
            for n, code in enumerate(codes):
                with (left if n % 2 == 0 else right):
                    values[code] = st.number_input(code, min_value=0.0, max_value=400.0, step=0.5, format="%.1f",
                                                   value=float(start[code]), help=CRANIAL_NAMES[code],
                                                   key=f"{code}_{demo}")
        st.form_submit_button("Reconstruct", type="primary", use_container_width=True)

if sex is None or ancestry is None:
    with stage:
        note("Choose sex and ancestry: the model and its ranges depend on them.")
    st.stop()

case = reconstruct_case(tuple(values[c] for c in PRIMARY), sex, ancestry)
result = case["result"]
truth = None if demo == NEW_CASE else true_landmarks_on(case["landmarks"], true_distance.loc[demo])

with stage:
    head_row = st.columns([1.6, 1.2, 1], vertical_alignment="bottom")
    with head_row[0]:
        label("Case " + ("· new" if demo == NEW_CASE else f"· {demo} · test subject"))
        view = st.segmented_control("View", ["Bone", "Bone + face", "Face"], default="Bone + face",
                                    label_visibility="collapsed") or "Bone + face"
    opacity = head_row[1].slider("Face opacity", 0.1, 1.0, 0.5, 0.05, disabled=view != "Bone + face")
    spheres = head_row[2].toggle("90% ranges", value=True)
    for warning in result["warnings"]:
        st.warning(warning + ": the face is an extraplation, trust it less.", icon=":material/warning:")
    mode = {"Bone": "Skull", "Bone + face": "Face over skull", "Face": "Face"}[view]
    st.plotly_chart(scene_figure(case, mode, opacity if mode == "Face over skull" else 1.0, spheres, truth),
                    use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]})
    half = (result["distances"]["high_90_mm"] - result["distances"]["predicted_mm"]).mean()
    cards = [("Distance range · 90%", f"± {half:.1f} mm", "mean over 66 facial distances", False),
            ("Landmark radius · 90%", f"{result['radius_90_mm'].mean():.1f} mm", "mean over 12 landmarks", False),
            ("Calibrated for", f"{ancestry} · {sex}", "ranges are specific to this group", False)]
    if truth is not None:
        error = np.linalg.norm(case["landmarks"] - truth, axis=1)
        inside = (error <= result["radius_90_mm"].to_numpy()).mean()
        cards.append(("Error vs CT · demo", f"{error.mean():.1f} mm", f"{inside:.0%} of landmarks inside range", True))
    stats(cards)
    
st.write("")
tab_distances, tab_landmarks, tab_export = st.tabs(["Facial distances", "Landmarks", "Export"])
with tab_distances:
    table = result["distances"].copy()
    if truth is not None:
        table["true_mm"] = true_distance.loc[demo].to_numpy()
    st.dataframe(table.round(1), use_container_width=True, height=320)
with tab_landmarks:
    landmarks = pd.DataFrame(case["landmarks"], index=result["landmarks"].index, columns=["x", "y", "z"])
    landmarks["radius_90_mm"] = result["radius_90_mm"]
    st.dataframe(landmarks.round(1), use_container_width=True)
with tab_export:
    case_id = "case" if demo == NEW_CASE else demo
    columns = st.columns(3)
    columns[0].download_button("Head · .obj", obj_text(case["vertices"], get_head_model()["triangles"], "Predicted head"),
                               f"{case_id}_head.obj", use_container_width=True)
    columns[1].download_button("Distances · .csv", result["distances"].round(2).to_csv(),
                               f"{case_id}_distances.csv", use_container_width=True)
    columns[2].download_button("Landmarks · .csv", landmarks.round(2).to_csv(), f"{case_id}_landmarks.csv",
                               use_container_width=True)
note("An approximation to guide an investigation, never to confirm an identity. The head is validated only at the "
     "12 landmarks; the skull is a reference model scaled to the measurements, not the case's skull.")