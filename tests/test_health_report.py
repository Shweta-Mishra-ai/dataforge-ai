"""Tests for pages/11 Health Report — compute_health, build_insights, build_health_pdf."""
import sys
import datetime
import io
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import importlib.util

sys.path.insert(0, ".")

# Mock Streamlit before importing page module
st_mock = mock.MagicMock()
st_mock.cache_data = lambda **kw: (lambda f: f)
st_mock.set_page_config = mock.MagicMock()
sys.modules["streamlit"] = st_mock
sys.modules["core.session_manager"] = mock.MagicMock()


def _load_health_module():
    spec = importlib.util.spec_from_file_location(
        "health_report", "pages/11_📋_Health_Report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    mod.st = st_mock
    mod.datetime = datetime
    mod.io = io
    mod.pd = pd
    mod.np = np
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod


@pytest.fixture(scope="module")
def hr_mod():
    return _load_health_module()


@pytest.fixture
def hr_df():
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "Department":            np.random.choice(["Eng", "Sales", "HR", "Finance"], n),
        "salary":                np.random.choice(["low", "medium", "high"], n),
        "satisfaction_level":    np.random.uniform(0.1, 1.0, n).round(3),
        "time_spend_company":    np.random.randint(0, 15, n),
        "average_montly_hours":  np.random.randint(140, 290, n),
        "promotion_last_5years": np.random.choice([0, 1], n, p=[0.92, 0.08]),
        "left":                  np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "last_evaluation":       np.random.uniform(0.4, 1.0, n).round(3),
        "number_project":        np.random.randint(2, 8, n),
    })


# ── compute_health ─────────────────────────────────────────────────────

class TestComputeHealth:
    def test_returns_dict_with_required_keys(self, hr_mod, hr_df):
        result = hr_mod.compute_health(hr_df)
        for key in ["score", "grade", "label", "color", "rows", "cols",
                    "missing_pct", "dup_pct", "outlier_pct"]:
            assert key in result, f"Missing key: {key}"

    def test_score_0_to_100(self, hr_mod, hr_df):
        result = hr_mod.compute_health(hr_df)
        assert 0 <= result["score"] <= 100

    def test_perfect_score_clean_data(self, hr_mod, hr_df):
        result = hr_mod.compute_health(hr_df)
        # Clean data → high score
        assert result["score"] >= 80

    def test_rows_and_cols_correct(self, hr_mod, hr_df):
        result = hr_mod.compute_health(hr_df)
        assert result["rows"] == len(hr_df)
        assert result["cols"] == len(hr_df.columns)

    def test_grade_is_letter(self, hr_mod, hr_df):
        result = hr_mod.compute_health(hr_df)
        assert result["grade"][0] in ["A", "B", "C", "D", "F"]  # A+ is valid

    def test_missing_data_reflected(self, hr_mod, hr_df):
        df_with_missing = hr_df.copy()
        df_with_missing.loc[:50, "satisfaction_level"] = None
        result = hr_mod.compute_health(df_with_missing)
        assert result["missing_pct"] > 0


# ── detect_niche ───────────────────────────────────────────────────────

class TestDetectNiche:
    def test_detects_hr(self, hr_mod, hr_df):
        niche, conf = hr_mod.detect_niche(hr_df)
        assert niche == "hr"
        assert isinstance(conf, (int, float))

    def test_returns_tuple(self, hr_mod, hr_df):
        result = hr_mod.detect_niche(hr_df)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ── build_insights ─────────────────────────────────────────────────────

class TestBuildInsights:
    def test_returns_list(self, hr_mod, hr_df):
        result = hr_mod.build_insights(hr_df.copy(), "hr")
        assert isinstance(result, list)

    def test_fires_multiple_insights_for_hr(self, hr_mod, hr_df):
        """Should fire at least 3 insights for HR data (attrition, sat, dept minimum)."""
        result = hr_mod.build_insights(hr_df.copy(), "hr")
        assert len(result) >= 3

    def test_each_insight_has_required_keys(self, hr_mod, hr_df):
        result = hr_mod.build_insights(hr_df.copy(), "hr")
        for ins in result:
            for key in ["tag", "title", "body", "action", "severity"]:
                assert key in ins, f"Missing key '{key}' in insight: {ins}"

    def test_severity_values_valid(self, hr_mod, hr_df):
        result = hr_mod.build_insights(hr_df.copy(), "hr")
        valid = {"critical", "warning", "positive", "info"}
        for ins in result:
            assert ins["severity"] in valid

    def test_attrition_insight_fires(self, hr_mod, hr_df):
        result = hr_mod.build_insights(hr_df.copy(), "hr")
        titles = [i["title"].lower() for i in result]
        assert any("attrition" in t for t in titles), "Attrition insight not firing"

    def test_flight_risk_fires_for_bad_data(self, hr_mod):
        """High attrition + low satisfaction → flight risk insight fires."""
        np.random.seed(1)
        n = 300
        df_bad = pd.DataFrame({
            "Department":           np.random.choice(["Eng", "Sales"], n),
            "satisfaction_level":   np.random.uniform(0.1, 0.4, n),  # very low sat
            "time_spend_company":   np.random.randint(3, 15, n),       # long tenure
            "left":                 np.random.choice([0, 1], n, p=[0.5, 0.5]),  # high atr
            "average_montly_hours": np.random.randint(200, 290, n),
        })
        result = hr_mod.build_insights(df_bad, "hr")
        tags = [i["tag"] for i in result]
        assert any("FLIGHT" in t.upper() for t in tags), \
            f"Flight risk not triggered. Tags: {tags}"

    def test_overwork_fires_for_high_hours(self, hr_mod):
        np.random.seed(2)
        n = 200
        df_overwork = pd.DataFrame({
            "Department":           ["Eng"] * n,
            "satisfaction_level":   np.random.uniform(0.3, 0.7, n),
            "average_montly_hours": np.random.randint(220, 290, n),  # >210 overwork
            "left":                 np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "time_spend_company":   np.random.randint(1, 10, n),
        })
        result = hr_mod.build_insights(df_overwork, "hr")
        tags = [i["tag"] for i in result]
        assert any("OVERWORK" in t.upper() for t in tags), \
            f"Overwork not triggered. Tags: {tags}"

    def test_no_crash_minimal_dataframe(self, hr_mod):
        df_min = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        result = hr_mod.build_insights(df_min, "general")
        assert isinstance(result, list)


# ── build_health_pdf ───────────────────────────────────────────────────

class TestBuildHealthPdf:
    def test_returns_bytes(self, hr_mod, hr_df):
        health = hr_mod.compute_health(hr_df)
        insights = hr_mod.build_insights(hr_df.copy(), "hr")
        result = hr_mod.build_health_pdf(hr_df, "hr", health, insights, "test.csv")
        assert isinstance(result, bytes)

    def test_pdf_non_empty(self, hr_mod, hr_df):
        health = hr_mod.compute_health(hr_df)
        insights = hr_mod.build_insights(hr_df.copy(), "hr")
        result = hr_mod.build_health_pdf(hr_df, "hr", health, insights, "test.csv")
        assert len(result) > 10_000  # at least 10KB

    def test_pdf_starts_with_pdf_header(self, hr_mod, hr_df):
        health = hr_mod.compute_health(hr_df)
        insights = hr_mod.build_insights(hr_df.copy(), "hr")
        result = hr_mod.build_health_pdf(hr_df, "hr", health, insights, "test.csv")
        assert result[:4] == b"%PDF"

    def test_no_crash_zero_insights(self, hr_mod, hr_df):
        health = hr_mod.compute_health(hr_df)
        result = hr_mod.build_health_pdf(hr_df, "hr", health, [], "test.csv")
        assert isinstance(result, bytes)
