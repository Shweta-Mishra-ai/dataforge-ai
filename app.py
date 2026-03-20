"""
DataForge AI — Entry Point
==========================
This file does ONE thing: configure the app and redirect to the first page.
All logic lives in pages/, core/, ai/, and components/.
"""

import streamlit as st

st.set_page_config(
    page_title="DataForge AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/1_Data_Upload.py")
