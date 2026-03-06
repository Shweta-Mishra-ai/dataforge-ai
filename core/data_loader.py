import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from config.settings import config


@dataclass
class LoadResult:
    df: Optional[pd.DataFrame]
    success: bool
    error: Optional[str] = None
    file_size_mb: float = 0.0
    sheet_names: list = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    filename: str = ""


def load_file(uploaded_file, sheet_name=0) -> LoadResult:
    try:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > config.max_file_mb:
            return LoadResult(
                df=None, success=False,
                error=f"File is {size_mb:.1f} MB. Maximum allowed is {config.max_file_mb} MB."
            )

        fname = uploaded_file.name.lower()
        sheets = []
        df = None

        if fname.endswith(".csv"):
            df = pd.read_csv(
                uploaded_file,
                encoding_errors="replace",
                low_memory=False
            )

        elif fname.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(uploaded_file)
            sheets = xl.sheet_names
            df = xl.parse(sheet_name)

        elif fname.endswith(".json"):
            df = pd.read_json(uploaded_file)

        else:
            return LoadResult(
                df=None, success=False,
                error="Unsupported format. Please upload CSV, Excel or JSON."
            )

        if df is None or df.empty:
            return LoadResult(df=None, success=False, error="File is empty.")

        if len(df) > config.max_rows_preview:
            df = df.head(config.max_rows_preview)

        df = _infer_dtypes(df)

        return LoadResult(
            df=df,
            success=True,
            file_size_mb=round(size_mb, 2),
            sheet_names=sheets,
            row_count=len(df),
            col_count=len(df.columns),
            filename=uploaded_file.name
        )

    except Exception as e:
        return LoadResult(df=None, success=False, error=f"Failed to load file: {str(e)}")


def _infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        try:
            converted = pd.to_datetime(df[col], infer_datetime_format=True)
            if converted.notna().sum() > len(df) * 0.7:
                df[col] = converted
                continue
        except Exception:
            pass
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.7:
                df[col] = converted
        except Exception:
            pass
    return df
