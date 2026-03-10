import io
import pandas as pd
import numpy as np
import re
from typing import Tuple


def _try_numeric(series: pd.Series) -> Tuple[pd.Series, bool, str]:
    """
    Try to coerce an object series to numeric.
    All regex ops use .str accessor (Series-safe).
    Returns (converted_series, success, method_used).
    """
    s = series.astype(str).str.strip()

    # Step 1 — direct conversion
    direct = pd.to_numeric(s, errors="coerce")
    if direct.notna().mean() > 0.85:
        return direct, True, "direct"

    # Step 2 — strip currency symbols + commas using .str accessor
    cleaned = (s
               .str.replace(r"[₹$£€¥₩]", "", regex=True)
               .str.replace(r"(?<=\d),(?=\d)", "", regex=True)
               .str.strip())
    attempt = pd.to_numeric(cleaned, errors="coerce")
    if attempt.notna().mean() > 0.85:
        return attempt, True, "currency_strip"

    # Step 3 — strip trailing percent sign
    pct = cleaned.str.replace(r"%\s*$", "", regex=True).str.strip()
    attempt2 = pd.to_numeric(pct, errors="coerce")
    if attempt2.notna().mean() > 0.85:
        return attempt2, True, "percent_strip"

    # Step 4 — aggressive: strip all trailing non-numeric chars
    agg = pct.str.replace(r"[^\d.\-+eE]+$", "", regex=True).str.strip()
    attempt3 = pd.to_numeric(agg, errors="coerce")
    if attempt3.notna().mean() > 0.70:
        return attempt3, True, "junk_strip"

    return series, False, "none"


def _try_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Try to parse a series as datetime."""
    s = series.astype(str).str.strip()
    # Skip pure integer-like columns (IDs, counts)
    if s.str.match(r"^\d{1,6}$").mean() > 0.5:
        return series, False
    try:
        converted = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
        if converted.notna().mean() > 0.80:
            return converted, True
    except Exception:
        pass
    return series, False


def load_file(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """
    Load CSV or Excel, run smart type inference.
    Returns (df, load_info) where load_info logs what was done.
    """
    info = {
        "filename": "",
        "rows": 0,
        "cols": 0,
        "type_conversions": [],
        "parse_errors": [],
    }

    name = getattr(uploaded_file, "name", "file")
    info["filename"] = name

    # ── Read file ─────────────────────────────────────────
    try:
        if name.endswith(".csv"):
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
                    break
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
            else:
                df = pd.read_csv(uploaded_file, encoding="utf-8",
                                 errors="replace", low_memory=False)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
    except Exception as e:
        raise ValueError("File read error: {}".format(str(e)))

    info["rows"] = len(df)
    info["cols"] = len(df.columns)

    # ── Smart type inference on object columns ─────────────
    for col in df.select_dtypes(include="object").columns:
        # Skip high-cardinality columns — likely IDs or free text
        if df[col].nunique() / max(len(df), 1) > 0.90:
            continue

        converted, ok, method = _try_numeric(df[col])
        if ok:
            df[col] = converted
            info["type_conversions"].append({
                "column": col,
                "from":   "object",
                "to":     "numeric",
                "method": method,
            })
            continue

        dt_converted, dt_ok = _try_datetime(df[col])
        if dt_ok:
            df[col] = dt_converted
            info["type_conversions"].append({
                "column": col,
                "from":   "object",
                "to":     "datetime",
                "method": "datetime_parse",
            })

    return df, info
