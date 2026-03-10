import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class LoadResult:
    success: bool
    df: Optional[pd.DataFrame] = None
    filename: str = ""
    file_size_mb: float = 0.0
    sheet_names: List[str] = field(default_factory=list)
    type_conversions: List[dict] = field(default_factory=list)
    error: str = ""


def _looks_like_id(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(30)
    if len(sample) == 0:
        return False
    has_letters = sample.str.contains(r"[A-Za-z]", regex=True)
    return bool(has_letters.mean() > 0.7)


def _clean_numeric_string(val) -> object:
    if val is None:
        return np.nan
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    s = str(val).strip()
    for sym in ["₹", "$", "£", "€", "¥", "₩", "Rs.", "Rs", "rs"]:
        s = s.replace(sym, "")
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    s = s.strip()
    if s in ("", "-", "N/A", "NA", "null", "None", "nan", "NaN", "#N/A"):
        return np.nan
    return s


def _try_numeric(series: pd.Series) -> Tuple[pd.Series, bool, str]:
    s = series.astype(str).str.strip()
    direct = pd.to_numeric(s, errors="coerce")
    if direct.notna().mean() > 0.85:
        return direct, True, "direct"
    cleaned = s.apply(_clean_numeric_string)
    attempt = pd.to_numeric(cleaned, errors="coerce")
    if attempt.notna().mean() > 0.80:
        return attempt, True, "currency_strip"
    return series, False, "none"


def _try_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    s = series.astype(str).str.strip()
    if s.str.match(r"^\d{1,6}$").mean() > 0.5:
        return series, False
    try:
        converted = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
        if converted.notna().mean() > 0.80:
            return converted, True
    except Exception:
        pass
    return series, False


def _infer_types(df: pd.DataFrame) -> List[dict]:
    conversions = []
    for col in list(df.select_dtypes(include="object").columns):
        try:
            if _looks_like_id(df[col]):
                continue
            converted, ok, method = _try_numeric(df[col])
            if ok:
                df[col] = converted
                conversions.append({"column": col, "from": "object", "to": "numeric", "method": method})
                continue
            dt_converted, dt_ok = _try_datetime(df[col])
            if dt_ok:
                df[col] = dt_converted
                conversions.append({"column": col, "from": "object", "to": "datetime", "method": "auto_parse"})
        except Exception:
            continue
    return conversions


def load_file(uploaded_file, sheet_name=None) -> LoadResult:
    name = getattr(uploaded_file, "name", "file")
    try:
        uploaded_file.seek(0, 2)
        size_mb = round(uploaded_file.tell() / (1024 * 1024), 2)
        uploaded_file.seek(0)
    except Exception:
        size_mb = 0.0

    if name.lower().endswith(".csv"):
        try:
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
                df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace", low_memory=False)
            return LoadResult(success=True, df=df, filename=name, file_size_mb=size_mb,
                              sheet_names=[], type_conversions=_infer_types(df))
        except Exception as e:
            return LoadResult(success=False, error=str(e), filename=name)

    elif name.lower().endswith((".xlsx", ".xls")):
        try:
            uploaded_file.seek(0)
            xl = pd.ExcelFile(uploaded_file)
            sheets = xl.sheet_names
            target = sheet_name if sheet_name else sheets[0]
            df = xl.parse(target)
            return LoadResult(success=True, df=df, filename=name, file_size_mb=size_mb,
                              sheet_names=sheets, type_conversions=_infer_types(df))
        except Exception as e:
            return LoadResult(success=False, error=str(e), filename=name)

    elif name.lower().endswith(".json"):
        try:
            uploaded_file.seek(0)
            df = pd.read_json(uploaded_file)
            return LoadResult(success=True, df=df, filename=name, file_size_mb=size_mb,
                              sheet_names=[], type_conversions=_infer_types(df))
        except Exception as e:
            return LoadResult(success=False, error=str(e), filename=name)

    else:
        return LoadResult(success=False, filename=name,
                          error="Unsupported file. Use CSV, Excel, or JSON.")
