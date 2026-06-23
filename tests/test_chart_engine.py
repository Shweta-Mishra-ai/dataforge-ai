"""
tests/test_chart_engine.py
Coverage for:
  - chart_engine.py — recommend_charts, individual builders, _apply_contrast
  - Bug fix: _SCORE_KEYWORDS defined before use in chart_exporter
  - Bug fix: select_dtypes("object") → ["object","string"]
  - Bug fix: individual chart builders (make_bar etc.) now use _apply_contrast not _style
"""
import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def hr_df():
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "Department":         np.random.choice(["Sales", "IT", "HR", "Finance"], n),
        "salary":             np.random.choice(["low", "medium", "high"], n),
        "satisfaction_level": np.random.uniform(0.1, 1.0, n).round(3),
        "last_evaluation":    np.random.uniform(0.4, 1.0, n).round(3),
        "number_project":     np.random.randint(2, 7, n),
        "average_montly_hours": np.random.randint(140, 310, n),
        "left":               np.random.choice([0, 1], n, p=[0.76, 0.24]),
    })


@pytest.fixture
def sales_df():
    np.random.seed(7)
    n = 200
    return pd.DataFrame({
        "Region":   np.random.choice(["North", "South", "East", "West"], n),
        "Product":  np.random.choice(["A", "B", "C"], n),
        "Revenue":  np.random.uniform(1000, 50000, n).round(2),
        "Units":    np.random.randint(1, 200, n),
    })


@pytest.fixture
def datetime_df():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    np.random.seed(0)
    return pd.DataFrame({
        "date":    dates,
        "sales":   np.random.uniform(100, 5000, 120).round(2),
        "region":  np.random.choice(["North", "South"], 120),
    })


# ── chart_engine imports ──────────────────────────────────

from core.chart_engine import (
    recommend_charts,
    make_bar,
    make_horizontal_bar,
    make_line,
    make_scatter,
    make_histogram,
    make_pie,
    make_heatmap,
    _apply_contrast,
    _style,
    _is_identifier,
    _get_analysis_columns,
    safe_pct_gap,
)


# ── _SCORE_KEYWORDS defined-before-use fix ────────────────

class TestScoreKeywordsOrdering:
    """Regression: _SCORE_KEYWORDS was defined AFTER make_ranked_bar_chart which used it."""

    def test_score_keywords_defined_at_module_level(self):
        from core import chart_exporter
        assert hasattr(chart_exporter, "_SCORE_KEYWORDS"), \
            "_SCORE_KEYWORDS must be module-level, not after functions that use it"

    def test_make_ranked_bar_chart_does_not_crash(self, hr_df):
        from core.chart_exporter import make_ranked_bar_chart
        result = make_ranked_bar_chart(hr_df, "Department", "satisfaction_level")
        assert isinstance(result, bytes)
        assert len(result) > 500


# ── _apply_contrast vs _style ─────────────────────────────

class TestContrastStyling:
    """Regression: make_bar/make_line etc. called _style() (dark only) instead of _apply_contrast()."""

    def test_make_bar_light_theme_white_background(self, hr_df):
        fig = make_bar(hr_df, "Department", "satisfaction_level", theme_name="Corporate Light")
        layout = fig.layout
        assert layout.paper_bgcolor == "white", \
            "Corporate Light should have white background, not dark"

    def test_make_bar_dark_theme_dark_background(self, hr_df):
        fig = make_bar(hr_df, "Department", "satisfaction_level", theme_name="Dark Tech")
        assert fig.layout.paper_bgcolor == "#07080f", \
            "Dark Tech should keep dark background"

    def test_make_line_light_theme_readable(self, hr_df):
        fig = make_line(hr_df, "number_project", "satisfaction_level", theme_name="Corporate Light")
        assert fig.layout.paper_bgcolor == "white"

    def test_make_scatter_light_theme(self, hr_df):
        fig = make_scatter(hr_df, "number_project", "satisfaction_level", theme_name="Corporate Light")
        assert fig.layout.paper_bgcolor == "white"

    def test_make_histogram_light_theme(self, hr_df):
        fig = make_histogram(hr_df, "satisfaction_level", theme_name="Corporate Light")
        assert fig.layout.paper_bgcolor == "white"

    def test_make_heatmap_light_theme(self, hr_df):
        fig = make_heatmap(hr_df, theme_name="Corporate Light")
        assert fig.layout.paper_bgcolor == "white"

    def test_all_light_themes_white_background(self, hr_df):
        light_themes = ["Corporate Light", "Executive Green", "Ocean Blue", "Slate Gray"]
        for theme in light_themes:
            fig = make_bar(hr_df, "Department", "satisfaction_level", theme_name=theme)
            assert fig.layout.paper_bgcolor == "white", \
                f"Theme '{theme}' should produce white background, got {fig.layout.paper_bgcolor}"


# ── recommend_charts ──────────────────────────────────────

class TestRecommendCharts:
    def test_returns_list_of_tuples(self, hr_df):
        result = recommend_charts(hr_df, domain="hr")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], go.Figure)

    def test_max_5_charts(self, hr_df):
        result = recommend_charts(hr_df, domain="hr")
        assert len(result) <= 5

    def test_no_crash_minimal_df(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = recommend_charts(df)
        assert isinstance(result, list)

    def test_dark_tech_theme_dark_bg(self, hr_df):
        result = recommend_charts(hr_df, domain="hr", theme_name="Dark Tech")
        for _, fig in result:
            if fig.layout.paper_bgcolor:
                assert fig.layout.paper_bgcolor == "#07080f"

    def test_light_theme_white_bg(self, hr_df):
        result = recommend_charts(hr_df, domain="hr", theme_name="Corporate Light")
        for _, fig in result:
            if fig.layout.paper_bgcolor:
                assert fig.layout.paper_bgcolor == "white"

    def test_excludes_id_columns(self):
        df = pd.DataFrame({
            "emp_id":             range(100),
            "satisfaction_level": np.random.uniform(0.1, 1.0, 100),
            "Department":         np.random.choice(["A", "B", "C"], 100),
        })
        cols = _get_analysis_columns(df)
        assert "emp_id" in cols["id_cols"], "emp_id should be detected as identifier"
        assert "emp_id" not in cols["all_metrics"], "emp_id must not appear in metrics"

    def test_datetime_triggers_trend(self, datetime_df):
        result = recommend_charts(datetime_df)
        titles = [t for t, _ in result]
        assert any("trend" in t.lower() or "time" in t.lower() for t in titles), \
            "Datetime column should trigger a time-series chart"


# ── _is_identifier ────────────────────────────────────────

class TestIsIdentifier:
    def test_emp_id_detected(self):
        s = pd.Series(range(500))
        assert _is_identifier("emp_id", s, 500)

    def test_satisfaction_not_identifier(self):
        s = pd.Series(np.random.uniform(0, 1, 200))
        assert not _is_identifier("satisfaction_level", s, 200)

    def test_sequential_integers_detected(self):
        s = pd.Series(range(1, 201))
        assert _is_identifier("row_num", s, 200)

    def test_revenue_not_identifier(self):
        s = pd.Series(np.random.uniform(1000, 50000, 300))
        assert not _is_identifier("revenue", s, 300)


# ── make_pie guard ────────────────────────────────────────

class TestMakePieGuard:
    def test_score_metric_redirects_to_bar(self, hr_df):
        """make_pie on a score metric should return horizontal bar, not pie."""
        fig = make_pie(hr_df, "Department", "satisfaction_level")
        # A redirected bar has orientation="h" traces
        assert any(
            getattr(t, "orientation", None) == "h"
            for t in fig.data
        ), "Score metric in make_pie should redirect to horizontal bar"

    def test_revenue_makes_pie(self, sales_df):
        fig = make_pie(sales_df, "Region", "Revenue")
        # Pie chart has Pie traces
        assert any(t.type == "pie" for t in fig.data), \
            "Revenue metric should produce pie chart"


# ── safe_pct_gap ──────────────────────────────────────────

class TestSafePctGap:
    def test_zero_baseline(self):
        result = safe_pct_gap(100, 0)
        assert "N/A" in result

    def test_absurd_gap_capped(self):
        result = safe_pct_gap(10000, 1)
        assert ">999%" in result

    def test_normal_gap(self):
        result = safe_pct_gap(110, 100)
        assert "10.0%" in result


# ── select_dtypes deprecation fix ────────────────────────

class TestSelectDtypesDeprecationFix:
    """Regression: select_dtypes(include='object') triggers Pandas4Warning.
    All files should now use include=['object','string']."""

    def test_pdf_builder_no_object_only(self):
        src = open("core/pdf_builder.py").read()
        assert 'include="object"' not in src, \
            'pdf_builder.py still uses deprecated select_dtypes(include="object")'

    def test_chart_engine_no_object_only(self):
        src = open("core/chart_engine.py").read()
        assert 'include="object"' not in src

    def test_story_engine_no_object_only(self):
        src = open("core/story_engine.py").read()
        assert 'include="object"' not in src

    def test_bi_engine_no_object_only(self):
        src = open("core/bi_engine.py").read()
        assert 'include="object"' not in src

    def test_ml_engine_no_object_only(self):
        src = open("core/ml_engine.py").read()
        assert 'include="object"' not in src

    def test_data_loader_no_object_only(self):
        src = open("core/data_loader.py").read()
        assert 'include="object"' not in src

    def test_insights_builder_no_object_only(self):
        src = open("core/insights_builder.py").read()
        assert 'include="object"' not in src


# ── no duplicate _c in pdf_builder ───────────────────────

class TestPdfBuilderNoDuplicates:
    def test_no_local_c_function_inside_build_pdf(self):
        """Regression: _c() was redefined locally inside build_pdf.
        After refactor, _c lives in core/pdf/theme.py — verify it's there exactly once."""
        src = open("core/pdf/theme.py").read()
        occurrences = src.count("def _c(")
        assert occurrences == 1, (
            f"Expected exactly 1 'def _c(' in core/pdf/theme.py, found {occurrences}. "
            "Do not redefine _c locally inside any function."
        )
        # Also verify pdf_builder.py is now just a shim (no local _c)
        shim = open("core/pdf_builder.py").read()
        assert "def _c(" not in shim, "pdf_builder.py shim must not define _c locally"


# ── _apply_contrast font colors ──────────────────────────

class TestApplyContrastFontColors:
    def test_light_theme_dark_font(self):
        import plotly.express as px
        fig = px.bar(x=["A", "B"], y=[1, 2])
        fig = _apply_contrast(fig, "Corporate Light")
        assert fig.layout.font.color == "#0F172A", \
            "Light theme must set near-black font color for readability"

    def test_dark_tech_uses_style(self):
        import plotly.express as px
        fig = px.bar(x=["A", "B"], y=[1, 2])
        fig = _apply_contrast(fig, "Dark Tech")
        assert fig.layout.paper_bgcolor == "#07080f"
