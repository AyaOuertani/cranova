import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_core import TAB
from brand import BONE_DIM, FONT_UI, HAIRLINE, MOSS, STONE, TERRACOTTA, display, label, note, stats

FINAL = "Ridge + sex + ancestry (final model)"
SHORT = {FINAL: "CRANOVA model", "Ridge (cranial only)": "Ridge, cranial only", "Mean predictor": "Population mean",
         "ReFace": "ReFace", "ReFace bias-corrected": "ReFace, bias-corrected"}

display("How far to trust it.", "Every number below comes from 55 subjects locked away before any model was built, "
        "and evaluated once.")

if not(TAB / "test_accuracy.csv").exists():
    note("No test results yet: run python src/evaluate_test.py --final once.")
    st.stop()

accuracy = pd.read_csv(TAB / "test_accuracy.csv", index_col="model")
groups = pd.read_csv(TAB / "test_by_group.csv", index_col="group")
landmarks = pd.read_csv(TAB / "test_by_landmark.csv", index_col=0)
final = accuracy.loc[FINAL]
stats([("Facial distance error", f"{final['MAE_mm']:.2f} mm", f"95% CI {final['ci_low']:.2f} - {final['ci_high']:.2f}", True),
       ("Gain vs population mean", f"{-accuracy.loc['Mean predictor', 'final_minus_this_mm']:.2f} mm",
        "the skull carries facial information", False),
       ("90% distance ranges", f"{groups.loc['all', 'interval_coverage']:.0%}", "of true values inside", False),
       ("90% landmark ranges", f"{groups.loc['all', 'sphere_coverage']:.0%}", "of true landmarks inside", False)])

def style(fig, height, title):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                       margin=dict(l=10, r=10, t=36, b=10), font=dict(color=BONE_DIM, size=12, family=FONT_UI),
                       title=dict(text=title.upper(), font=dict(size=11, color=BONE_DIM), x=0.01))
    fig.update_xaxes(gridcolor=HAIRLINE, zeroline=False, ticksuffix=" mm")
    fig.update_yaxes(gridcolor=HAIRLINE)
    return fig

st.write("")
left, right = st.columns([1.3, 1], gap="large")
with left:
    order = accuracy.sort_values("MAE_mm", ascending=False)
    names = [SHORT.get(n, n) for n in order.index]
    colours = [TERRACOTTA if n == FINAL else STONE for n in order.index]
    fig = go.Figure(go.Scatter(
        x=order["MAE_mm"], y=names, mode="markers", marker=dict(size=12, color=colours, line=dict(width=0)),
        error_x=dict(type="data", symmetric=False, array=order["ci_high"] - order["MAE_mm"],
                     arrayminus=order["MAE_mm"] - order["ci_low"], color=MOSS, thickness=2, width=0),
        hovertemplate="%{y} : %{x:.2f} mm<extra></extra>"
    ))
    st.plotly_chart(style(fig, 330, "Error per method - 95% CI"), use_container_width=True,
                    config={"displayModeBar": False})
    note("Reface needs the full skull scan, and it's referance data come from the same CT subjects."
         "CRANOVA needs 16 measurements and gives calibrated ranges.")
    
with right:
    by_landmark = landmarks.sort_values("landmark_error_mm")
    fig = go.Figure(go.Bar(x=by_landmark["landmark_error_mm"], y=by_landmark.index, orientation="h",
                           marker=dict(color=STONE), hovertemplate="%{y}: %{x:.2f} mm<extra></extra>"))
    st.plotly_chart(style(fig, 360, "3D error per landmark"), use_container_width=True,
                    config={"displayModeBar": False})

label("By ancestry and sex")
table = groups.rename(columns={"MAE_mm": "Distance error (mm)", "interval_coverage": "Distance ranges hit",
                               "interval_width_mm": "Range width (mm)", "landmark_error_mm": "Landmark error (mm)",
                               "sphere_coverage": "Landmark ranges hit", "sphere_radius_mm": "Radius (mm)"})

st.dataframe(table.style.format({c: "{:.0%}" for c in ["Distance ranges hit", "Landmark ranges hit"]} | {c: "{:.2f}" for c in table.columns if "hit" not in c}),
             use_container_width=True)
note("ReFace needs the full skull scan, and its reference data come from the same CT subjects. "
     "CRANOVA needs 16 measurements and gives calibrated ranges.")
