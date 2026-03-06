import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_id_like: bool
    is_constant: bool
    has_outliers: bool
    outlier_count: int
    outlier_pct: float
    sample_values: list
    stats: Dict = field(default_factory=dict)
    quality_score: float = 100.0


@dataclass
class DatasetProfile:
    rows: int
    cols: int
    total_cells: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    overall_quality_score: float
    column_profiles: List[ColumnProfile]
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    recommendations: List[str]


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols    = df.select_dtypes(include="datetime").columns.tolist()
    total_cells      = df.shape[0] * df.shape[1]
    missing_cells    = int(df.isnull().sum().sum())
    dup_rows         = int(df.duplicated().sum())
    col_profiles     = [_profile_column(df, col) for col in df.columns]
    completeness     = (1 - missing_cells / max(total_cells, 1)) * 100
    dedup_score      = (1 - dup_rows / max(len(df), 1)) * 100
    col_health       = sum(p.quality_score for p in col_profiles) / max(len(col_profiles), 1)
    overall          = round(completeness * 0.60 + dedup_score * 0.30 + col_health * 0.10, 1)
    recs             = _generate_recommendations(col_profiles, dup_rows, missing_cells, total_cells, len(df))

    return DatasetProfile(
        rows=len(df), cols=len(df.columns),
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=round(missing_cells / max(total_cells, 1) * 100, 1),
        duplicate_rows=dup_rows,
        duplicate_pct=round(dup_rows / max(len(df), 1) * 100, 1),
        overall_quality_score=overall,
        column_profiles=col_profiles,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        recommendations=recs,
    )


def _profile_column(df, col) -> ColumnProfile:
    s             = df[col]
    n             = len(s)
    missing       = int(s.isnull().sum())
    unique        = int(s.nunique())
    unique_pct    = round(unique / max(n, 1) * 100, 1)
    has_outliers  = False
    outlier_count = 0
    outlier_pct   = 0.0
    stats         = {}

    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        if len(clean) > 4:
            q1, q3        = clean.quantile(0.25), clean.quantile(0.75)
            iqr           = q3 - q1
            lo, hi        = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count = int(((s < lo) | (s > hi)).sum())
            has_outliers  = outlier_count > 0
            outlier_pct   = round(outlier_count / max(n, 1) * 100, 1)
            stats         = {
                "mean":   round(float(clean.mean()), 4),
                "median": round(float(clean.median()), 4),
                "std":    round(float(clean.std()), 4),
                "min":    round(float(clean.min()), 4),
                "max":    round(float(clean.max()), 4),
            }

    score  = 100.0
    score -= (missing / max(n, 1)) * 50
    if has_outliers:
        score -= min(outlier_pct * 0.5, 15)
    if unique <= 1:
        score -= 20
    score = max(0.0, round(score, 1))

    return ColumnProfile(
        name=col, dtype=str(s.dtype),
        missing_count=missing,
        missing_pct=round(missing / max(n, 1) * 100, 1),
        unique_count=unique, unique_pct=unique_pct,
        is_id_like=unique_pct > 95 and unique > 10,
        is_constant=unique <= 1,
        has_outliers=has_outliers,
        outlier_count=outlier_count,
        outlier_pct=outlier_pct,
        sample_values=s.dropna().unique()[:5].tolist(),
        stats=stats, quality_score=score,
    )


def _generate_recommendations(profiles, dup_rows, missing, total, n_rows) -> List[str]:
    recs = []
    if dup_rows > 0:
        recs.append(f"🔁 {dup_rows} duplicate rows found. Remove them.")
    if missing / max(total, 1) > 0.10:
        recs.append(f"⚠️ {missing/max(total,1)*100:.1f}% cells are missing.")
    for p in profiles:
        if p.is_constant:
            recs.append(f"🗑️ '{p.name}' has only 1 unique value — drop it.")
        elif p.missing_pct > 60:
            recs.append(f"🚨 '{p.name}' is {p.missing_pct}% empty — drop it.")
        elif p.missing_pct > 20:
            recs.append(f"⚠️ '{p.name}' has {p.missing_pct}% missing values.")
        if p.has_outliers and p.outlier_count > 5:
            recs.append(f"📊 '{p.name}' has {p.outlier_count} outliers.")
    return recs
