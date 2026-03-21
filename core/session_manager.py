"""
session_manager.py
Centralized session state management.

Problems it solves:
1. df.to_json() destroys dtypes → use pickle-based caching instead
2. df_active mutated directly → maintain original + active separately  
3. No hash → stale cache served → use content hash for invalidation
4. Page switch resets state → centralized init guards against this
"""
import hashlib
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any


# ── Session Keys ──────────────────────────────────────────
KEY_DF_RAW        = "df_raw"
KEY_DF_ACTIVE     = "df_active"
KEY_DF_HASH       = "df_hash"
KEY_FILENAME      = "filename"
KEY_FILE_SIZE     = "file_size_mb"
KEY_PROFILE       = "profile"
KEY_CLEAN_REPORT  = "clean_report"
KEY_STATS         = "stats_cache"
KEY_STORY         = "story_cache"
KEY_ML_RESULT     = "ml_result"
KEY_DTYPES_MAP    = "original_dtypes"


def init_session():
    defaults = {
        KEY_DF_RAW:       None,
        KEY_DF_ACTIVE:    None,
        KEY_DF_HASH:      None,
        KEY_FILENAME:     "",
        KEY_FILE_SIZE:    0.0,
        KEY_PROFILE:      None,
        KEY_CLEAN_REPORT: None,
        KEY_STATS:        None,
        KEY_STORY:        None,
        KEY_ML_RESULT:    None,
        KEY_DTYPES_MAP:   {},
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def has_data() -> bool:
    return (
        KEY_DF_ACTIVE in st.session_state
        and st.session_state[KEY_DF_ACTIVE] is not None
        and len(st.session_state[KEY_DF_ACTIVE]) > 0
    )


def require_data(redirect_page: str = "pages/1_🏠_Data_Upload.py"):
    """
    FIXED: redirect_page now matches actual file name with emoji.
    Call at top of every analysis page.
    """
    init_session()
    if not has_data():
        st.warning("No dataset loaded. Please upload a file first.")
        try:
            st.page_link(redirect_page, label="Go to Upload")
        except Exception:
            st.info("Please go to the Upload page from the sidebar.")
        st.stop()


def get_df() -> pd.DataFrame:
    return st.session_state[KEY_DF_ACTIVE]


def get_raw_df() -> pd.DataFrame:
    return st.session_state[KEY_DF_RAW]


def set_dataframe(df_raw: pd.DataFrame, filename: str, size_mb: float):
    init_session()
    st.session_state[KEY_DF_RAW]     = df_raw.copy()
    st.session_state[KEY_DF_ACTIVE]  = df_raw.copy()
    st.session_state[KEY_FILENAME]   = filename
    st.session_state[KEY_FILE_SIZE]  = size_mb
    st.session_state[KEY_DF_HASH]    = _hash_df(df_raw)
    st.session_state[KEY_DTYPES_MAP] = {
        col: str(dtype) for col, dtype in df_raw.dtypes.items()
    }
    _invalidate_caches()


def update_active_df(df_cleaned: pd.DataFrame):
    st.session_state[KEY_DF_ACTIVE] = df_cleaned


def cache_stats(stats_obj):
    st.session_state[KEY_STATS] = stats_obj


def get_cached_stats():
    return st.session_state.get(KEY_STATS)


def cache_story(story_obj):
    st.session_state[KEY_STORY] = story_obj


def get_cached_story():
    return st.session_state.get(KEY_STORY)


def cache_ml_result(ml_obj):
    st.session_state[KEY_ML_RESULT] = ml_obj


def get_cached_ml():
    return st.session_state.get(KEY_ML_RESULT)


def get_filename() -> str:
    return st.session_state.get(KEY_FILENAME, "Dataset")


def get_file_size() -> float:
    return st.session_state.get(KEY_FILE_SIZE, 0.0)


def is_cache_valid(df: pd.DataFrame) -> bool:
    stored_hash = st.session_state.get(KEY_DF_HASH)
    if not stored_hash:
        return False
    return stored_hash == _hash_df(df)


def get_session_summary() -> Dict[str, Any]:
    df = st.session_state.get(KEY_DF_ACTIVE)
    return {
        "has_data":  has_data(),
        "filename":  st.session_state.get(KEY_FILENAME, ""),
        "rows":      len(df) if df is not None else 0,
        "cols":      len(df.columns) if df is not None else 0,
        "size_mb":   st.session_state.get(KEY_FILE_SIZE, 0),
        "has_stats": st.session_state.get(KEY_STATS) is not None,
        "has_story": st.session_state.get(KEY_STORY) is not None,
        "has_ml":    st.session_state.get(KEY_ML_RESULT) is not None,
        "df_hash":   (st.session_state.get(KEY_DF_HASH, "")[:8] + "..."
                      if st.session_state.get(KEY_DF_HASH) else "none"),
    }


# ── Internal helpers ──────────────────────────────────────

def _hash_df(df: pd.DataFrame) -> str:
    try:
        parts = [
            str(df.shape),
            str(list(df.columns)),
            str(list(df.dtypes.astype(str))),
            str(len(df)),
        ]
        try:
            parts.append(str(df.iloc[0].astype(str).tolist())  if len(df) > 0 else "")
            parts.append(str(df.iloc[-1].astype(str).tolist()) if len(df) > 0 else "")
        except Exception:
            pass
        content = "|".join(parts)
        return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return hashlib.md5(str(id(df)).encode()).hexdigest()


def _invalidate_caches():
    st.session_state[KEY_STATS]        = None
    st.session_state[KEY_STORY]        = None
    st.session_state[KEY_ML_RESULT]    = None
    st.session_state[KEY_CLEAN_REPORT] = None
    st.session_state[KEY_PROFILE]      = None
