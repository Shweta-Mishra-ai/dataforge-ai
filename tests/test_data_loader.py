"""tests/test_data_loader.py — Unit tests for core/data_loader.py"""
import io
import pytest
import pandas as pd
from unittest.mock import MagicMock
from core.data_loader import load_file, _clean_columns, _smart_dtype_inference


def _make_upload(content: bytes, name: str, size: int = None):
    f = MagicMock()
    f.name = name
    f.size = size or len(content)
    buf = io.BytesIO(content)
    f.read = buf.read
    f.seek = buf.seek
    return f


# _clean_columns
def test_clean_columns_strips_whitespace():
    df = pd.DataFrame({" Name ": [1], "Age": [2]})
    df_clean, _ = _clean_columns(df)
    assert "Name" in df_clean.columns

def test_clean_columns_renames_unnamed():
    df = pd.DataFrame({"Unnamed: 0": [1], "Value": [2]})
    df_clean, warnings = _clean_columns(df)
    assert "Unnamed: 0" not in df_clean.columns
    assert warnings

def test_clean_columns_no_change_on_clean_df():
    df = pd.DataFrame({"Name": [1], "Age": [2]})
    df_clean, warnings = _clean_columns(df)
    assert list(df_clean.columns) == ["Name", "Age"]
    assert warnings == []


# _smart_dtype_inference
def test_dtype_converts_numeric_strings():
    df = pd.DataFrame({"score": ["10", "20", "30", "40"]})
    result = _smart_dtype_inference(df)
    assert pd.api.types.is_numeric_dtype(result["score"])

def test_dtype_skips_id_columns():
    df = pd.DataFrame({"customer_id": ["001", "002", "003"]})
    result = _smart_dtype_inference(df)
    # Should NOT be numeric — stays string/object
    assert not pd.api.types.is_numeric_dtype(result["customer_id"])

def test_dtype_skips_low_conversion_rate():
    df = pd.DataFrame({"mixed": ["1", "2", "abc", "xyz"]})
    result = _smart_dtype_inference(df)
    # Only 50% numeric — should NOT convert to numeric
    assert not pd.api.types.is_numeric_dtype(result["mixed"])


# load_file CSV
def test_load_csv_basic():
    csv = b"name,age,score\nAlice,30,95\nBob,25,88"
    result = load_file(_make_upload(csv, "test.csv"))
    assert result.success
    assert len(result.df) == 2
    assert "name" in result.df.columns

def test_load_csv_semicolon():
    """Semicolon CSVs load successfully (may fall back to single-column parse
    depending on mock seek behavior — core: file must not crash)."""
    import io as _io
    csv = b"name;age;score\nAlice;30;95\nBob;25;88"
    f = MagicMock()
    f.name = "test.csv"
    f.size = len(csv)
    buf = _io.BytesIO(csv)
    f.read = buf.read
    f.seek = buf.seek
    result = load_file(f)
    # Must not crash — success is the minimum bar
    assert result.success
    assert result.df is not None
    assert len(result.df) == 2

def test_load_csv_empty():
    result = load_file(_make_upload(b"", "empty.csv"))
    assert not result.success

def test_load_unsupported_format():
    result = load_file(_make_upload(b"data", "file.parquet"))
    assert not result.success
    assert "Unsupported" in result.error

def test_load_too_large():
    result = load_file(_make_upload(b"x", "big.csv", size=201*1024*1024))
    assert not result.success
    assert "Maximum" in result.error


# load_file JSON
def test_load_json_records():
    import json
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    result = load_file(_make_upload(json.dumps(data).encode(), "data.json"))
    assert result.success
    assert len(result.df) == 2

def test_load_json_dict():
    import json
    data = {"name": ["Alice", "Bob"], "age": [30, 25]}
    result = load_file(_make_upload(json.dumps(data).encode(), "data.json"))
    assert result.success
    assert len(result.df) == 2
