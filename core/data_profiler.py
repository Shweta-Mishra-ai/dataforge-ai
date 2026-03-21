"""
core/data_profiler.py — Data Quality Engine.
Fixed: robust handling of mixed-type columns, large files, Arrow serialization.
NO Streamlit imports allowed.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ColumnProfile:
    name:          str
    dtype:         str
    missing_count: int
    missing_pct:   float
    unique_count:  int
    unique_pct:    float
    is_id_like:    bool
    is_constant:   bool
    has_outliers:  bool
    outlier_count: int
    outlier_pct:   float
    sample_values: list
    stats:         Dict = field(default_factory=dict)
    quality_score: float = 100.0


@dataclass
class DatasetProfile:
    rows:                  int
    cols:                  int
    total_cells:           int
    missing_cells:         int
    missing_pct:           float
    duplicate_rows:        int
    duplicate_pct:         float
    overall_quality_score: float
    column_profiles:       List[ColumnProfile]
    numeric_cols:          List[str]
    categorical_cols:      List[str]
    datetime_cols:         List[str]
    recommendations:       List[str]


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Full quality profiling — robust to mixed types and large files."""

    # Sanitize first — replace infinities
    try:
        num_cols_raw = df.select_dtypes(include="number").columns
        if len(num_cols_raw) > 0:
            df = df.copy()
            df[num_cols_raw] = df[num_cols_raw].replace(
                [np.inf, -np.inf], np.nan)
    except Exception:
        pass

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols    = df.select_dtypes(include="datetime").columns.tolist()

    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    dup_rows      = int(df.duplicated().sum())

    col_profiles = []
    for col in df.columns:
        try:
            col_profiles.append(_profile_column(df, col))
        except Exception:
            # Never let one bad column crash the profiler
            col_profiles.append(ColumnProfile(
                name=col, dtype=str(df[col].dtype),
                missing_count=int(df[col].isna().sum()),
                missing_pct=round(df[col].isna().mean()*100,1),
                unique_count=0, unique_pct=0.0,
                is_id_like=False, is_constant=False,
                has_outliers=False, outlier_count=0, outlier_pct=0.0,
                sample_values=[], stats={}, quality_score=50.0,
            ))

    completeness = (1 - missing_cells / max(total_cells, 1)) * 100
    dedup_score  = (1 - dup_rows / max(len(df), 1)) * 100
    col_health   = sum(p.quality_score for p in col_profiles) / max(len(col_profiles), 1)
    overall      = round(completeness*0.60 + dedup_score*0.30 + col_health*0.10, 1)

    recs = _generate_recommendations(col_profiles, dup_rows, missing_cells, total_cells, len(df))

    return DatasetProfile(
        rows=len(df), cols=len(df.columns),
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=round(missing_cells/max(total_cells,1)*100,1),
        duplicate_rows=dup_rows,
        duplicate_pct=round(dup_rows/max(len(df),1)*100,1),
        overall_quality_score=overall,
        column_profiles=col_profiles,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        recommendations=recs,
    )


def _profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
    """Profile a single column — fully robust."""
    s = df[col]
    n = len(s)
    missing    = int(s.isnull().sum())
    unique     = int(s.nunique(dropna=True))
    unique_pct = round(unique / max(n, 1) * 100, 1)

    has_outliers  = False
    outlier_count = 0
    outlier_pct   = 0.0
    stats: Dict   = {}

    # Only run numeric stats on truly numeric, non-empty series
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        try:
            # Force to float64 and drop non-finite values
            clean = pd.to_numeric(s, errors="coerce").dropna()
            clean = clean[np.isfinite(clean)]   # remove inf/-inf

            if len(clean) > 4:
                q1  = float(np.percentile(clean.values, 25))
                q3  = float(np.percentile(clean.values, 75))
                iqr = q3 - q1
                lo  = q1 - 1.5 * iqr
                hi  = q3 + 1.5 * iqr

                outlier_mask  = (clean < lo) | (clean > hi)
                outlier_count = int(outlier_mask.sum())
                has_outliers  = outlier_count > 0
                outlier_pct   = round(outlier_count / max(n, 1) * 100, 1)

                stats = {
                    "mean":   round(float(clean.mean()), 4),
                    "median": round(float(clean.median()), 4),
                    "std":    round(float(clean.std()), 4),
                    "min":    round(float(clean.min()), 4),
                    "max":    round(float(clean.max()), 4),
                    "q25":    round(q1, 4),
                    "q75":    round(q3, 4),
                }
        except Exception:
            pass   # skip stats for this column silently

    # Quality score
    score = 100.0
    score -= (missing / max(n, 1)) * 50
    if has_outliers:
        score -= min(outlier_pct * 0.5, 15)
    if unique <= 1 and n > 1:
        score -= 20
    score = max(0.0, round(score, 1))

    # Safe sample values — convert to strings to avoid Arrow issues
    try:
        samples = [str(v) for v in s.dropna().unique()[:5].tolist()]
    except Exception:
        samples = []

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
        sample_values=samples,
        stats=stats,
        quality_score=score,
    )


def _generate_recommendations(
    profiles: List[ColumnProfile],
    dup_rows: int,
    missing: int,
    total: int,
    n_rows: int,
) -> List[str]:
    """Plain-English recommendations — no emojis (PDF compatible)."""
    recs = []

    if dup_rows > 0:
        recs.append(
            "{:,} duplicate rows found ({:.1f}%). "
            "Remove them to prevent skewed aggregations.".format(
                dup_rows, dup_rows/max(n_rows,1)*100)
        )

    if missing / max(total, 1) > 0.10:
        recs.append(
            "{:.1f}% of all cells are missing. "
            "Review imputation strategy for each column.".format(
                missing/max(total,1)*100)
        )

    for p in profiles:
        if p.is_constant:
            recs.append(
                "Column '{}' has only 1 unique value — useless for analysis. "
                "Consider dropping it.".format(p.name)
            )
        elif p.missing_pct > 60:
            recs.append(
                "Column '{}' is {:.0f}% empty. Consider dropping it.".format(
                    p.name, p.missing_pct)
            )
        elif p.missing_pct > 20:
            recs.append(
                "Column '{}' has {:.0f}% missing values. Choose a fill strategy.".format(
                    p.name, p.missing_pct)
            )
        if p.has_outliers and p.outlier_count > 5:
            recs.append(
                "'{}' has {:,} outliers ({:.1f}%). Investigate before analysis.".format(
                    p.name, p.outlier_count, p.outlier_pct)
            )

    return recs
