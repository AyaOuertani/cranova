import numpy as np
import streamlit as st

OBSIDIAN, SURFACE, SURFACE_2, HAIRLINE = "#0A0A09", "#131311", "#1A1A17", "#2C2D28"
BONE, BONE_DIM, TERRACOTTA, TERRACOTTA_LIGHT = "#DFCDC0", "#9C9187", "#B56543", "#D0855F"
MOSS, STONE = "#373933", "#7E8079"
FONT_UI, FONT_DISPLAY = "Jost, 'Segoe UI', sans-serif", "'Cormorant Garamond', Georgia, serif"


def contour_background():
    """Thin flowing contour lines (the brand's topographic motif) as an inline SVG."""
    x = np.linspace(0, 900, 90)
    paths = []
    for k in range(26):
        y = 520 - k * 13 + 60 * np.sin(x / 140 + k * 0.18) + 25 * np.sin(x / 55 - k * 0.1)
        points = " ".join(f"{a:.0f},{b:.0f}" for a, b in zip(x, y))
        paths.append(f'<polyline points="{points}" fill="none" stroke="%23{BONE[1:]}" '
                     f'stroke-opacity="{0.05 + 0.004 * k:.3f}" stroke-width="0.8"/>')
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 600'>{''.join(paths)}</svg>"
    return svg.replace('"', "'").replace("<", "%3C").replace(">", "%3E")


def apply_brand():
    st.markdown(f"""<style>
@font-face {{font-family: 'Jost'; src: url('app/static/jost.woff2') format('woff2'); font-weight: 100 900;}}
@font-face {{font-family: 'Cormorant Garamond'; src: url('app/static/cormorant.woff2') format('woff2');
             font-weight: 300 700;}}
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  font-family: {FONT_UI} !important; letter-spacing: 0.01em;}}
.stApp {{background: {OBSIDIAN} url("data:image/svg+xml,{contour_background()}") no-repeat right -120px bottom -160px / 900px;}}
header[data-testid="stHeader"] {{background: transparent; height: 0;}}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{display: none;}}
.block-container {{padding-top: 1.2rem; max-width: 1440px;}}
h1, h2, h3, h4, h5 {{font-family: {FONT_UI} !important; font-weight: 300 !important; color: {BONE};}}

/* brand bar */
.cr-brand {{display: flex; align-items: center; gap: 14px;}}
.cr-brand img {{height: 46px;}}
.cr-word {{font-weight: 300; font-size: 1.45rem; letter-spacing: 0.42em; color: {BONE};}}
.cr-tag {{font-size: 0.62rem; letter-spacing: 0.32em; color: {TERRACOTTA_LIGHT}; text-transform: uppercase; margin-top: 2px;}}
.cr-rule {{height: 1px; background: linear-gradient(90deg, {TERRACOTTA} 0, {HAIRLINE} 120px, {HAIRLINE} 100%);
           margin: 10px 0 22px;}}
[data-testid="stPageLink-NavLink"] {{background: transparent !important; padding: 6px 0;}}
[data-testid="stPageLink-NavLink"] p, .cr-nav-active {{font-size: 0.72rem !important; letter-spacing: 0.26em;
  text-transform: uppercase; color: {BONE_DIM};}}
[data-testid="stPageLink-NavLink"]:hover p {{color: {BONE};}}
.cr-nav-active {{color: {BONE}; padding: 6px 0 5px; border-bottom: 1px solid {TERRACOTTA}; display: inline-block;
                 margin-top: -4px;}}

/* typography helpers */
.cr-label {{font-size: 0.66rem; letter-spacing: 0.3em; text-transform: uppercase; color: {TERRACOTTA_LIGHT};
            margin: 4px 0 10px;}}
.cr-display {{font-family: {FONT_DISPLAY}; font-weight: 300; font-size: 2.6rem; line-height: 1.05; color: {BONE};
              margin: 0 0 4px;}}
.cr-lead {{color: {BONE_DIM}; font-size: 0.95rem; max-width: 640px; margin-bottom: 18px;}}

/* stat cards */
.cr-stats {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1px;
            background: {HAIRLINE}; border: 1px solid {HAIRLINE};}}
.cr-stat {{background: {SURFACE}; padding: 14px 18px 16px;}}
.cr-stat .l {{font-size: 0.62rem; letter-spacing: 0.26em; text-transform: uppercase; color: {BONE_DIM};}}
.cr-stat .v {{font-family: {FONT_DISPLAY}; font-size: 2.1rem; font-weight: 400; color: {BONE}; line-height: 1.15;}}
.cr-stat .v.accent {{color: {TERRACOTTA_LIGHT};}}
.cr-stat .n {{font-size: 0.75rem; color: {BONE_DIM};}}

/* case file panel and stage */
[data-testid="stForm"] {{background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 2px;
                         border-top: 2px solid {TERRACOTTA};}}
[data-baseweb="input"], [data-baseweb="select"] > div {{background: {SURFACE_2} !important; border-radius: 2px !important;
                                                        border-color: {HAIRLINE} !important;}}
[data-baseweb="input"] input {{font-variant-numeric: tabular-nums;}}
.stButton button[kind="primaryFormSubmit"], .stFormSubmitButton button {{border-radius: 2px; letter-spacing: 0.22em;
  text-transform: uppercase; font-size: 0.75rem;}}
[data-testid="stPlotlyChart"] {{background: radial-gradient(ellipse at 50% 45%, #1C1B17 0%, {OBSIDIAN} 70%);
                                border: 1px solid {HAIRLINE};}}
.stTabs [data-baseweb="tab"] p {{font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;}}
.stTabs [data-baseweb="tab-highlight"] {{background: {TERRACOTTA};}}
[data-testid="stDataFrame"] {{border: 1px solid {HAIRLINE};}}
.cr-note {{font-size: 0.78rem; color: {BONE_DIM}; border-left: 2px solid {MOSS}; padding-left: 10px; margin-top: 10px;}}
</style>""", unsafe_allow_html=True)


def brand_bar(pages, current):
    left, right = st.columns([1.4, 1], vertical_alignment="center")
    left.markdown("""<div class="cr-brand"><img src="app/static/cranova_mark.png" alt="">
        <div><div class="cr-word">CRANOVA</div><div class="cr-tag">Reconstructing what remains</div></div></div>""",
                  unsafe_allow_html=True)
    for column, page in zip(right.columns(len(pages)), pages):
        if page.title == current.title:
            column.markdown(f'<span class="cr-nav-active">{page.title}</span>', unsafe_allow_html=True)
        else:
            column.page_link(page)
    st.markdown('<div class="cr-rule"></div>', unsafe_allow_html=True)


def label(text):
    st.markdown(f'<div class="cr-label">{text}</div>', unsafe_allow_html=True)


def display(title, lead=None):
    st.markdown(f'<div class="cr-display">{title}</div>' + (f'<div class="cr-lead">{lead}</div>' if lead else ""),
                unsafe_allow_html=True)


def stats(items):
    cards = "".join(f'<div class="cr-stat"><div class="l">{l}</div><div class="v{" accent" if a else ""}">{v}</div>'
                    f'<div class="n">{n}</div></div>' for l, v, n, a in items)
    st.markdown(f'<div class="cr-stats">{cards}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="cr-note">{text}</div>', unsafe_allow_html=True)
