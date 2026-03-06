import pandas as pd
from typing import Dict


def clean_with_strategy(df: pd.DataFrame, strategies: Dict[str, str]) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    cols_to_drop = []

    for col, strategy in strategies.items():
        if col not in df_clean.columns:
            continue
        if strategy == "keep":
            continue
        elif strategy == "drop_col":
            cols_to_drop.append(col)
        elif strategy == "fill_mean":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == "fill_median":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == "fill_mode":
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
        elif strategy == "fill_zero":
            df_clean[col] = df_clean[col].fillna(0)
        elif strategy == "fill_unknown":
            df_clean[col] = df_clean[col].fillna("Unknown")
        elif strategy == "ffill":
            df_clean[col] = df_clean[col].ffill()
        elif strategy == "drop_rows":
            df_clean = df_clean.dropna(subset=[col])

    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop, errors="ignore")

    return df_clean.reset_index(drop=True)


def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    strategies = {}
    for col in df.columns:
        s           = df[col]
        missing_pct = s.isnull().mean() * 100
        if missing_pct > 60:
            strategies[col] = "drop_col"
        elif pd.api.types.is_numeric_dtype(s):
            strategies[col] = "fill_median"
        else:
            strategies[col] = "fill_mode"
    return clean_with_strategy(df, strategies)
