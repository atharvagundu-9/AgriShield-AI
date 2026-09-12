"""AgriShield AI Streamlit entry point."""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AgriShield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

pages = [
    st.Page("pages/0_Dashboard.py", title="Dashboard", icon="🏡", default=True),
    st.Page("pages/1_Yield_Prediction.py", title="Yield Prediction", icon="📈"),
    st.Page("pages/2_Climate_Risk.py", title="Climate Risk", icon="🌦️"),
    st.Page("pages/3_What_If_Simulator.py", title="What-If Simulator", icon="🎛️"),
    st.Page("pages/4_Crop_Recommendation.py", title="Crop Recommendation", icon="🌱"),
    st.Page("pages/5_District_Analysis.py", title="District Analysis", icon="🗺️"),
    st.Page("pages/6_Historical_Trends.py", title="Historical Trends", icon="🕰️"),
    st.Page("pages/7_Explainability.py", title="Explainability", icon="🔎"),
    st.Page("pages/8_Advisory.py", title="Advisory", icon="📋"),
    st.Page("pages/9_About.py", title="About", icon="ℹ️"),
]

st.navigation(pages, position="sidebar", expanded=True).run()
