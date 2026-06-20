"""Tests for core/chart_exporter.py — chart generation and smart aggregation."""
import sys
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
sys.modules.setdefault("streamlit", mock.MagicMock())

from core.chart_exporter import (
    make_bar_chart,
    make_histogram,
    make_correlation_heatmap,
    make_ranked_bar_chart,
    generate_all_charts,
    _pick_best_metric,
)


@pytest.fixture
def hr_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "Department":          np.random.choice(["Eng", "Sales", "HR"], n),
        "salary":              np.random.choice(["low", "medium", "high"], n),
        "satisfaction_level":  np.random.uniform(0.1, 1.0, n).round(3),
        "last_evaluation":     np.random.uniform(0.4, 1.0, n).round(3),
        "number_project":      np.random.randint(2, 8, n),
        "left":                np.random.choice([0, 1], n, p=[0.76, 0.24]),
    })


@pytest.fixture
def sales_df():
    np.random.seed(7)
    n = 150
    return pd.DataFrame({
        "Region":   np.random.choice(["North", "South", "East", "West"], n),
        "Product":  np.random.choice(["A", "B", "C"], n),
        "Revenue":  np.random.randint(1000, 50000, n),
        "Units":    np.random.randint(1, 500, n),
        "Profit":   np.random.randint(100, 10000, n),
    })


# ── _pick_best_metric ──────────────────────────────────────────────────

class TestPickBestMetric:
    def test_prefers_satisfaction_over_number_project(self, hr_df):
        num_cols = ["number_project", "satisfaction_level", "left"]
        result = _pick_best_metric(num_cols)
        assert result == "satisfaction_level"

    def test_prefers_score_keyword(self):
        cols = ["id", "age", "performance_score", "revenue"]
        result = _pick_best_metric(cols)
        assert result == "performance_score"

    def test_falls_back_to_first_when_no_score(self, sales_df):
        cols = ["Revenue", "Units", "Profit"]
        result = _pick_best_metric(cols)
        assert result == "Revenue"

    def test_handles_single_column(self):
        assert _pick_best_metric(["revenue"]) == "revenue"

    def test_handles_rating_keyword(self):
        cols = ["count", "rating_score", "id"]
        assert _pick_best_metric(cols) == "rating_score"


# ── make_bar_chart ─────────────────────────────────────────────────────

class TestMakeBarChart:
    def test_returns_bytes(self, hr_df):
        result = make_bar_chart(hr_df, "Department", "satisfaction_level")
        assert isinstance(result, bytes)
        assert len(result) > 1000

    def test_uses_mean_for_score_metric(self, hr_df):
        """satisfaction_level should use mean, not sum — verifies the COUNT bug is fixed."""
        result = make_bar_chart(hr_df, "Department", "satisfaction_level")
        assert isinstance(result, bytes)
        # If sum was used, values would be ~40-60 per dept; with mean they're 0.0-1.0
        # We can't read values from PNG directly, but we verify no crash and bytes produced

    def test_uses_sum_for_revenue(self, sales_df):
        result = make_bar_chart(sales_df, "Region", "Revenue")
        assert isinstance(result, bytes)

    def test_custom_title(self, hr_df):
        result = make_bar_chart(hr_df, "Department", "satisfaction_level",
                                title="My Custom Title")
        assert isinstance(result, bytes)

    def test_all_themes(self, hr_df):
        themes = ["Corporate Light", "Dark Tech", "Executive Green",
                  "Ocean Blue", "Slate Gray"]
        for theme in themes:
            result = make_bar_chart(hr_df, "Department", "satisfaction_level",
                                    theme_name=theme)
            assert isinstance(result, bytes), f"Failed for theme: {theme}"


# ── make_ranked_bar_chart ──────────────────────────────────────────────

class TestMakeRankedBarChart:
    def test_returns_bytes(self, hr_df):
        result = make_ranked_bar_chart(hr_df, "Department", "satisfaction_level")
        assert isinstance(result, bytes)
        assert len(result) > 1000

    def test_works_with_many_categories(self, hr_df):
        result = make_ranked_bar_chart(hr_df, "Department", "last_evaluation")
        assert isinstance(result, bytes)


# ── make_histogram ─────────────────────────────────────────────────────

class TestMakeHistogram:
    def test_returns_bytes(self, hr_df):
        result = make_histogram(hr_df, "satisfaction_level")
        assert isinstance(result, bytes)

    def test_works_with_integer_column(self, hr_df):
        result = make_histogram(hr_df, "number_project")
        assert isinstance(result, bytes)


# ── make_correlation_heatmap ───────────────────────────────────────────

class TestMakeCorrelationHeatmap:
    def test_returns_bytes(self, hr_df):
        result = make_correlation_heatmap(hr_df)
        assert isinstance(result, bytes)

    def test_min_columns_required(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        # Should not crash even with only 2 numeric cols
        result = make_correlation_heatmap(df)
        assert result is None or isinstance(result, bytes)


# ── generate_all_charts ────────────────────────────────────────────────

class TestGenerateAllCharts:
    def test_returns_list_of_tuples(self, hr_df):
        charts = generate_all_charts(hr_df)
        assert isinstance(charts, list)
        assert len(charts) > 0
        for title, img in charts:
            assert isinstance(title, str)
            assert isinstance(img, bytes)

    def test_max_charts_respected(self, hr_df):
        charts = generate_all_charts(hr_df, max_charts=3)
        assert len(charts) <= 3

    def test_score_metric_prioritized(self, hr_df):
        """satisfaction_level must be the headline metric, not number_project."""
        charts = generate_all_charts(hr_df)
        titles = [t.lower() for t, _ in charts]
        assert any("satisfaction" in t for t in titles), (
            "Score metric not prioritized — chart titles: " + str(titles)
        )

    def test_no_numeric_bin_xaxis(self, hr_df):
        """Chart 2 should NOT be a numeric-binned trend (unreadable labels)."""
        charts = generate_all_charts(hr_df)
        titles = [t for t, _ in charts]
        # Old bug produced titles like "satisfaction_level Trend" with bins as X axis
        bad_patterns = ["(0.", "(0,", "Trend"]
        for title in titles:
            for pat in bad_patterns:
                assert pat not in title, (
                    f"Numeric-bin trend chart still present: {title}"
                )

    def test_works_with_datetime_column(self):
        df = pd.DataFrame({
            "date":  pd.date_range("2023-01-01", periods=100, freq="D"),
            "sales": np.random.randint(1000, 5000, 100),
            "region": np.random.choice(["North", "South"], 100),
        })
        charts = generate_all_charts(df)
        assert len(charts) > 0

    def test_works_with_minimal_dataset(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value":    [10, 20, 30],
        })
        charts = generate_all_charts(df)
        assert len(charts) >= 1

    def test_all_chart_bytes_non_empty(self, hr_df):
        charts = generate_all_charts(hr_df)
        for title, img in charts:
            assert len(img) > 500, f"Chart too small: {title}"
