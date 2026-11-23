import streamlit as st

from brand import apply_brand, brand_bar

st.set_page_config(page_title="CRANOVA · Facial approximation", page_icon="app/static/cranova_mark.png",
                   layout="wide")
apply_brand()
pages = [st.Page("views/reconstruct.py", title="Reconstruct", default=True),
         st.Page("views/evaluation.py", title="Evaluation"),
         st.Page("views/method.py", title="Method")]
current = st.navigation(pages, position="hidden")
brand_bar(pages, current)
current.run()
