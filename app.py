
"""
DataForge AI — Entry Point
Streamlit will automatically show first page from pages/ folder.
No switch_page needed — avoids filename mismatch.
"""
import streamlit as st

st.set_page_config(
    page_title="DataForge AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("## DataForge AI")
st.markdown("Loading...")
