"""
components/styles.py
Centralized CSS for DataForge AI.
Import and call inject_global_css() at top of every page.
"""
import streamlit as st

# ── Brand tokens ──────────────────────────────────────────
BRAND_DARK      = "#0D1B2E"
BRAND_DARK2     = "#0F2240"
BRAND_ACCENT    = "#2563A8"
BRAND_ACCENT_LT = "#60A5FA"

SIDEBAR_CSS = """
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2E 0%, #0F2240 100%) !important;
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stButton button { color: white !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }
"""


def inject_global_css():
    """Call once per page — injects brand sidebar + base typography."""
    st.markdown(
        f"<style>{BASE_CSS}{SIDEBAR_CSS}</style>",
        unsafe_allow_html=True,
    )
