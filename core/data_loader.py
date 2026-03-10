import pandas as pd
import numpy as np
from typing import Tuple


def _clean_numeric_string(val: str) -> str:
    """Remove currency symbols, commas, percent from a single string value."""
    if not isinstance(val, str):
        return val
    # Remove currency symbols
    for sym in ["₹", "$", "£", "€", "¥", "₩"]:
        val = val.replace(sym, "")
    # Remove commas between digits (1,234 → 1234)
    result = ""
    for i, ch in enumerate(val):
        if ch == "," and i > 0 and i < len(val) - 1:
            if val[i-1].isdigit() and val[i+1].isdigit():
                continue
        result += ch
    # Remove trailing percent
    result = result.strip()
    if result.endswith("%"):
        result = result[:-1].strip()
    return result


def _try_numeric(series: pd.Series) -> Tuple[pd.Series, bool, str]:
    """
    Try to coerce an object series to numeric.
    Returns (converted_series, success, method_used).
    """
    s = series.astype(str).str.strip()

    # Step 1 — direct
    direct = pd.to_numeric(s, errors="coerce")
    if direct.notna().mean() > 0.85:
        return direct, True, "direct"

    # Step 2 — clean currency/commas/percent then try
    cleaned = s.apply(_clean_numeric_string)
    attempt = pd.to_numeric(cleaned, errors="coerce")
    if attempt.notna().mean() > 0.80:
        return attempt, True, "currency_strip"

    return series, False, "none"


def _try_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Try to parse a series as datetime."""
    s = series.astype(str).str.strip()
    # Skip if mostly short integers (IDs, counts)
    short_int = s.str.match(r"^\d{1,6}$")
    if short_int.mean() > 0.5:
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
    Load CSV or Excel with smart type inference.
    Returns (df, load_info).
    """
    info = {
        "filename": "",
        "rows": 0,
        "cols": 0,
        "type_conversions": [],
    }

    name = getattr(uploaded_file, "name", "file")
    info["filename"] = name

    # ── Read file ─────────────────────────────────────────
    try:
        if name.lower().endswith(".csv"):
            df = None
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if df is None:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="utf-8",
                                 errors="replace", low_memory=False)
        elif name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("Could not read file: {}".format(str(e)))

    info["rows"] = len(df)
    info["cols"] = len(df.columns)

    # ── Type inference on object columns ──────────────────
    for col in list(df.select_dtypes(include="object").columns):
        try:
            n_unique = df[col].nunique()
            n_rows   = max(len(df), 1)

            # Skip high-cardinality (ID/free-text columns)
            if n_unique / n_rows > 0.90:
                continue

            # Try numeric
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

            # Try datetime
            dt_converted, dt_ok = _try_datetime(df[col])
            if dt_ok:
                df[col] = dt_converted
                info["type_conversions"].append({
                    "column": col,
                    "from":   "object",
                    "to":     "datetime",
                    "method": "auto_parse",
                })

        except Exception:
            # Never crash the whole upload for one column
            continue

    return df, info
