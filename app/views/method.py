"""Method: the path from bone to face, the data, the limits, the credits."""
import streamlit as st

from brand import BONE, BONE_DIM, FONT_DISPLAY, HAIRLINE, SURFACE, TERRACOTTA, display, label

display("From bone to face.", "Five steps, each one tested before the next was built.")
steps = [("Measure", "16 cranial measurements, sex and ancestry"),
         ("Predict", "Ridge regression trained on 219 CT subjects"),
         ("Bound", "66 facial distances, each with a 90% range"),
         ("Locate", "12 landmarks in 3D, each with a 90% radius"),
         ("Display", "Head model fitted over a scaled reference skull")]
nodes = "".join(f"""<div style="flex:1; position:relative; padding:0 10px;">
  <div style="width:34px; height:34px; border:1px solid {TERRACOTTA}; border-radius:50%; background:{SURFACE};
       color:{BONE}; display:flex; align-items:center; justify-content:center; font-family:{FONT_DISPLAY};
       font-size:1.1rem; position:relative; z-index:1;">{n}</div>
  <div style="margin-top:12px; font-size:0.7rem; letter-spacing:0.26em; text-transform:uppercase; color:{BONE};">{t}</div>
  <div style="font-size:0.85rem; color:{BONE_DIM}; margin-top:4px;">{d}</div></div>"""
                for n, (t, d) in enumerate(steps, start=1))
st.markdown(f"""<div style="position:relative; display:flex; margin:10px 0 34px;">
  <div style="position:absolute; top:17px; left:27px; right:10%; height:1px;
       background:linear-gradient(90deg, {BONE} 0%, {TERRACOTTA} 100%); opacity:0.6;"></div>{nodes}</div>""",
            unsafe_allow_html=True)

data, limits = st.columns(2, gap="large")
with data:
    label("Data")
    st.markdown("""
- CT-derived cranial and facial measurements of **274 adults** (AFR, ASN, EUR; F and M), CC BY 4.0
- **219** development subjects for training and tuning · **55** locked test subjects, evaluated once
- Tissue-depth data place the reference skull under the face
- Every alternative tested (elastic net, PLS, neural networks, head-model prior) reached the same accuracy:
  the limit is the information in the measurements, not the model
""")
with limits:
    label("Limits")
    st.markdown("""
- An **approximation** to guide an investigation, never to confirm an identity
- The head surface is validated **only at the 12 landmarks**
- The skull shown is a **reference model** scaled to the measurements, not the case's skull
- Larger errors for **AFR** subjects; only AFR, ASN and EUR are supported
- All 16 measurements are required: no damaged or incomplete skulls yet
""")
st.markdown(f'<div style="border-top:1px solid {HAIRLINE}; margin-top:18px; padding-top:12px; font-size:0.78rem; '
            f'color:{BONE_DIM};">Head model: GNM Head v3, Google (Apache 2.0) · Reference skull: Sketchfab model, '
            'author and licence to be confirmed · Fonts: Jost and Cormorant Garamond (SIL Open Font License) · '
            'Data citations: see the thesis.</div>', unsafe_allow_html=True)
