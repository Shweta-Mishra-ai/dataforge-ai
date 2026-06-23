"""
core/config.py — Centralized configuration accessor.
Single source of truth for all secrets and app-level settings.
All pages import from here — never access st.secrets directly.
"""
import os
import streamlit as st
import logging
logger = logging.getLogger(__name__)

# ── File size limits ──────────────────────────────────────
MAX_FILE_SIZE_MB   = 150          # hard reject above this
WARN_FILE_SIZE_MB  = 50           # warn above this
MAX_ROWS           = 500_000      # sample above this
MAX_COLS           = 200          # drop extra cols above this
MAX_CHART_COUNT    = 8            # max charts in PDF

# ── ML limits ─────────────────────────────────────────────
ML_MAX_ROWS        = 100_000      # sample for ML if larger
ML_MIN_ROWS        = 30           # refuse ML below this
ML_CV_FOLDS        = 5
ML_TEST_SIZE       = 0.20

# ── Insight thresholds ────────────────────────────────────
ATTRITION_CRITICAL = 20.0         # % above which is critical
ATTRITION_HIGH     = 15.0
ATTRITION_WARN     = 10.0
SATISFACTION_TARGET = 0.70        # internal planning target (0–1 scale)
OVERWORK_CRITICAL  = 240          # monthly hours
OVERWORK_WARN      = 220

def get_groq_key() -> str:
    """Single accessor for GROQ API key — never call st.secrets directly."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        logger.debug("st.secrets not available", exc_info=True)
    return os.environ.get("GROQ_API_KEY", "").strip()

def groq_available() -> bool:
    return bool(get_groq_key())

def validate_upload(size_mb: float, n_rows: int, n_cols: int) -> list[str]:
    """
    Returns list of error strings. Empty = upload is OK.
    Call in app.py before set_dataframe().
    """
    errors = []
    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File too large ({size_mb:.0f} MB). Maximum is {MAX_FILE_SIZE_MB} MB. "
            "Split the file or contact support."
        )
    if n_rows > MAX_ROWS:
        errors.append(
            f"Dataset has {n_rows:,} rows — will be sampled to {MAX_ROWS:,} "
            "for analysis. Results represent a random sample."
        )
    if n_cols > MAX_COLS:
        errors.append(
            f"Dataset has {n_cols} columns — first {MAX_COLS} will be used."
        )
    return errors
