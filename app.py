"""
DataForge AI — Entry Point
"""
import streamlit as st

st.set_page_config(
    page_title="DataForge AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-redirect to first page
# Streamlit will show the first page in pages/ automatically
# No explicit switch_page needed — avoids filename mismatch errors
