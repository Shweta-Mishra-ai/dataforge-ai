import pandas as pd
import numpy as np
from typing import Tuple


def _looks_like_id(series: pd.Series) -> bool:
    """
    Returns True only if column looks like an ID/free-text field.
    Logic: most values contain letters (product IDs, names, URLs etc.)
    Pure numeric-with-symbols like prices will NOT be flagged as ID.
    """
    sample = series.dropna().astype(str).head(30)
    if len(sample) == 0:
        return False
    has_letters = sample.str.contains(r"[A-Za-z]", regex=True)
    return bool(has_letters.mean() > 0.7)


def _clean_numeric_string(val) -> object:
    """Remove currency symbols, commas, percent from a single value."""
    if val is None:
        return np.nan
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    s = str(val).strip()
    # Remove currency symbols
    for sym in ["₹", "$", "£", "€", "¥", "₩", "Rs.", "Rs", "rs"]:
        s = s.replace(sym, "")
    # Remove commas (thousands separator)
    s = s.replace(",", "")
    # Remove trailing percent
    if s.endswith("%"):
        s = s[:-1]
    s = s.strip()
    # Common null strings → NaN
    if s in ("", "-", "N/A", "NA", "null", "None", "nan", "NaN", "#N/A"):
        return np.nan
    return s


def _try_numeric(series: pd.Series) -> Tuple[pd.Series, bool, str]:
    """Try to coerce an object series to numeric."""
    s = series.astype(str).str.strip()

    # Step 1 — direct (already clean numbers)
    direct = pd.to_numeric(s, errors="coerce")
    if direct.notna().mean() > 0.85:
        return direct, True, "direct"

    # Step 2 — strip currency / commas / percent then retry
    cleaned = s.apply(_clean_numeric_string)
    attempt = pd.to_numeric(cleaned, errors="coerce")
    if attempt.notna().mean() > 0.80:
        return attempt, True, "currency_strip"

    return series, False, "none"


def _try_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Try to parse a series as datetime."""
    s = series.astype(str).str.strip()
    # Skip columns that are mostly short integers
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
                except Exception:
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
            # Skip ID/free-text columns (contain letters like product_id, name, URL)
            if _looks_like_id(df[col]):
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
            continue   # never crash upload for one bad column

    return df, info
