"""
tests/test_codebase_health.py

Regression tests for:
  - Logging set up in all core modules (no more silent failures)
  - story_engine: key_findings always populated
  - story_engine: executive_summary is rich and specific
  - bi_engine: segments have strengths/weaknesses at 5% threshold
  - bi_engine: segment opportunities populated when weaknesses exist
  - No bare except:pass without logging in production code
  - data_profiler: profile_dataset returns expected shape
  - insight_engine: build_top_insights returns typed Insight objects
"""
import re
import os
import pytest
import pandas as pd
import numpy as np


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def hr_df():
    np.random.seed(42)
    n = 400
    return pd.DataFrame({
        "Department":             np.random.choice(["Sales", "IT", "HR", "Finance"], n),
        "satisfaction_level":     np.random.uniform(0.1, 1.0, n).round(3),
        "last_evaluation":        np.random.uniform(0.4, 1.0, n).round(3),
        "number_project":         np.random.randint(2, 7, n),
        "average_montly_hours":   np.random.randint(140, 310, n),
        "left":                   np.random.choice([0, 1], n, p=[0.76, 0.24]),
        "salary":                 np.random.choice(["low", "medium", "high"], n),
        "time_spend_company":     np.random.randint(1, 10, n),
    })


@pytest.fixture
def minimal_df():
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0], "cat": ["a", "b", "a", "b", "a"]})


# ── Logging regression: all core modules must have logger ────────────────

CORE_FILES_NEEDING_LOGGING = [
    "core/insight_engine.py",
    "core/insights_builder.py",
    "core/data_profiler.py",
    "core/story_engine.py",
    "core/engines/base.py",
    "core/engines/hr.py",
    "core/engines/ecommerce.py",
    "core/engines/sales.py",
    "core/engines/finance.py",
    "core/engines/general.py",
    "core/dashboards/hr.py",
    "core/dashboards/finance.py",
    "core/dashboards/ecommerce.py",
    "core/dashboards/sales.py",
    "core/dashboards/general.py",
    "core/bi_engine.py",
    "core/chart_exporter.py",
    "core/chart_engine.py",
    "core/pdf/theme.py",
    "core/pdf/primitives.py",
    "core/pdf/narrative_sections.py",
    "core/pdf/data_sections.py",
    "core/pdf/domain_sections.py",
    "core/pdf/builder.py",
    "core/health_pdf_builder.py",
    "core/eda_engine.py",
    "core/ml_engine.py",
    "core/domain_dashboards.py",
    "ai/report_narrator.py",
]


@pytest.mark.parametrize("filepath", CORE_FILES_NEEDING_LOGGING)
def test_module_has_logging(filepath):
    """Regression: silent except:pass replaced by logger.debug — logger must be set up."""
    src = open(filepath).read()
    assert "import logging" in src, f"{filepath} missing 'import logging'"
    assert "logger = logging.getLogger" in src, f"{filepath} missing logger instance"


@pytest.mark.parametrize("filepath", CORE_FILES_NEEDING_LOGGING)
def test_no_bare_silent_except(filepath):
    """Regression: no more 'except Exception:\\n    pass' without at least a log call."""
    src = open(filepath).read()
    lines = src.splitlines()
    violations = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("except Exception:", "except Exception as e:", "except:"):
            next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if next_stripped in ("pass", "continue", "..."):
                violations.append(f"line {i + 2}: {stripped} → {next_stripped}")
    assert not violations, (
        f"{filepath} has bare silent excepts (no logging):\n" + "\n".join(violations)
    )


# ── story_engine: key_findings always populated ──────────────────────────

class TestStoryEngineKeyFindings:
    def test_key_findings_not_empty_hr(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert story.key_findings, "key_findings must never be empty — was empty for HR domain"

    def test_key_findings_not_empty_minimal(self, minimal_df):
        from core.story_engine import generate_story
        story = generate_story(minimal_df)
        assert story.key_findings, "key_findings must never be empty even for minimal df"

    def test_key_findings_are_strings(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        for f in story.key_findings:
            assert isinstance(f, str), f"finding should be str, got {type(f)}"
            assert len(f) > 5, "finding string too short to be meaningful"

    def test_key_findings_max_6(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert len(story.key_findings) <= 6


class TestStoryEngineExecSummary:
    def test_exec_summary_not_boilerplate(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        summary = story.executive_summary.lower()
        # The old boilerplate was: "This X-row Y dataset analysis identified N critical issue(s)"
        # Pattern: starts with "this" and has "dataset analysis"
        assert not (summary.startswith("this") and "dataset analysis" in summary), \
            "exec summary still matches old boilerplate pattern (starts with 'this...dataset analysis')"

    def test_exec_summary_has_row_count(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert "400" in story.executive_summary, \
            "exec summary should include actual row count"

    def test_exec_summary_has_column_count(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert "8" in story.executive_summary, \
            "exec summary should include column count"

    def test_exec_summary_has_completeness(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert "100%" in story.executive_summary or "missing" in story.executive_summary.lower(), \
            "exec summary should mention data completeness"

    def test_exec_summary_has_attrition_when_present(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert "attrition" in story.executive_summary.lower() or "left" in story.executive_summary.lower(), \
            "exec summary should mention attrition for HR dataset"

    def test_exec_summary_minimal_df(self, minimal_df):
        from core.story_engine import generate_story
        story = generate_story(minimal_df)
        assert len(story.executive_summary) > 30, "exec summary too short"
        assert isinstance(story.executive_summary, str)

    def test_exec_summary_mentions_recommendations(self, hr_df):
        from core.story_engine import generate_story
        story = generate_story(hr_df)
        assert "recommendation" in story.executive_summary.lower(), \
            "exec summary should mention number of recommendations"


# ── bi_engine: segment strengths/weaknesses at 5% threshold ─────────────

class TestBIEngineSegments:
    def test_segments_populated(self, hr_df):
        from core.bi_engine import run_bi
        bi = run_bi(hr_df)
        assert bi.segments, "BI segments should not be empty for HR dataset"

    def test_some_segment_has_strength_or_weakness(self, hr_df):
        from core.bi_engine import run_bi
        bi = run_bi(hr_df)
        has_sw = any(seg.strengths or seg.weaknesses for seg in bi.segments)
        assert has_sw, (
            "At least one segment should have strengths or weaknesses at 5% threshold. "
            "Were all empty? Threshold may have reverted to 10%."
        )

    def test_segment_health_score_range(self, hr_df):
        from core.bi_engine import run_bi
        bi = run_bi(hr_df)
        for seg in bi.segments:
            assert 0 <= seg.health_score <= 100, \
                f"Segment health score out of range: {seg.health_score}"

    def test_segment_opportunity_set_when_weakness(self, hr_df):
        from core.bi_engine import run_bi
        bi = run_bi(hr_df)
        for seg in bi.segments:
            if seg.weaknesses:
                assert seg.opportunity and len(seg.opportunity) > 5, \
                    f"Segment '{seg.segment_name}' has weaknesses but empty opportunity text"

    def test_benchmarks_populated(self, hr_df):
        from core.bi_engine import run_bi
        bi = run_bi(hr_df)
        assert bi.benchmarks, "BI benchmarks should not be empty for HR dataset"


# ── data_profiler sanity ─────────────────────────────────────────────────

class TestDataProfiler:
    def test_profile_dataset_returns_dataclass(self, hr_df):
        from core.data_profiler import profile_dataset
        result = profile_dataset(hr_df)
        # profile_dataset returns a DatasetProfile dataclass, not a dict
        assert hasattr(result, "rows"), "profile should have .rows attribute"
        assert hasattr(result, "cols"), "profile should have .cols attribute"

    def test_profile_has_shape(self, hr_df):
        from core.data_profiler import profile_dataset
        result = profile_dataset(hr_df)
        assert result.rows == len(hr_df)
        assert result.cols == len(hr_df.columns)

    def test_profile_completeness(self, hr_df):
        from core.data_profiler import profile_dataset
        result = profile_dataset(hr_df)
        # hr_df has no missing values
        assert result.missing_pct == 0.0

    def test_profile_minimal_no_crash(self, minimal_df):
        from core.data_profiler import profile_dataset
        result = profile_dataset(minimal_df)
        assert result.rows == len(minimal_df)

    def test_profile_quality_grade(self, hr_df):
        from core.data_profiler import profile_dataset
        result = profile_dataset(hr_df)
        assert result.data_quality_grade in ("A", "B", "C", "D", "F")


# ── insights_builder returns typed objects ────────────────────────────────

class TestInsightsBuilder:
    def test_returns_insight_objects(self, hr_df):
        from core.insights_builder import build_top_insights, Insight as BuilderInsight
        insights = build_top_insights(hr_df, domain="hr")
        for ins in insights:
            # insights_builder has its own Insight class (separate from story_engine.Insight)
            assert hasattr(ins, "severity"), f"Expected Insight-like object, got {type(ins)}"
            assert hasattr(ins, "title")
            assert hasattr(ins, "problem")

    def test_insight_has_required_fields(self, hr_df):
        from core.insights_builder import build_top_insights
        insights = build_top_insights(hr_df, domain="hr")
        assert insights, "Should return at least 1 insight for HR dataset"
        for ins in insights:
            assert ins.title and len(ins.title) > 3
            assert ins.severity in ("critical", "high", "medium", "low", "warning", "positive")
            assert ins.problem and len(ins.problem) > 5

    def test_general_domain_no_crash(self, minimal_df):
        from core.insights_builder import build_top_insights
        result = build_top_insights(minimal_df, domain="general")
        assert isinstance(result, list)


# ── Narrative fallback computes real stats ────────────────────────────────

class TestReportNarrativeFallback:
    def test_chart_narrative_fallback_has_real_numbers(self, hr_df):
        """Regression: narrative fallback was 'Chart analysis computed from dataset statistics.'"""
        src = open("pages/8_Reports.py").read()
        assert "Chart analysis computed from dataset statistics." not in src, \
            "Boilerplate fallback string still present in pages/8_Reports.py"

    def test_report_page_imports_logging(self):
        src = open("pages/8_Reports.py").read()
        assert "import logging" in src


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: No bare except / logger.debug suppression in dashboards
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardSplit:
    def test_domain_dashboards_shim_re_exports(self):
        """Shim must re-export get_domain_kpis and get_domain_charts."""
        from core.domain_dashboards import get_domain_kpis, get_domain_charts
        assert callable(get_domain_kpis)
        assert callable(get_domain_charts)

    def test_dashboards_package_importable(self):
        """All dashboard submodules must import without error."""
        import core.dashboards.base
        import core.dashboards.hr
        import core.dashboards.finance
        import core.dashboards.ecommerce
        import core.dashboards.sales
        import core.dashboards.general
        assert True  # reaching here = no ImportError

    def test_no_bare_except_in_dashboards(self):
        """Dashboard files must not use bare except:."""
        import ast
        import pathlib
        for path in pathlib.Path("core/dashboards").glob("*.py"):
            if path.name == "__init__.py":
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError as e:
                pytest.fail(f"{path.name}: SyntaxError — {e}")
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    pytest.fail(f"{path.name} line {node.lineno}: bare `except:` not allowed")

    def test_no_silent_debug_in_dashboards(self):
        """Dashboard files must not use logger.debug for unexpected failures."""
        import pathlib
        for path in pathlib.Path("core/dashboards").glob("*.py"):
            if path.name == "__init__.py":
                continue
            src = path.read_text()
            if 'logger.debug("%s silent skip"' in src or 'logger.debug("%s skip"' in src:
                pytest.fail(
                    f"{path.name}: still has silent logger.debug. "
                    "Use logger.warning for unexpected failures."
                )


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Silent failure surfacing — logger.warning sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestSilentFailureSweep:
    CRITICAL_MODULES = [
        "core/engines/hr.py",
        "core/engines/finance.py",
        "core/engines/ecommerce.py",
        "core/engines/sales.py",
        "core/engines/general.py",
        "core/bi_engine.py",
        "core/insight_engine.py",
        "core/insights_builder.py",
        "core/eda_engine.py",
        "core/pdf/narrative_sections.py",
        "core/pdf/data_sections.py",
        "core/pdf/domain_sections.py",
        "core/health_pdf_builder.py",
    ]

    @pytest.mark.parametrize("filepath", CRITICAL_MODULES)
    def test_no_silent_debug_exceptions(self, filepath):
        """
        Critical modules must not swallow unexpected failures with logger.debug.
        All unexpected except blocks should use logger.warning or logger.error.
        """
        import pathlib
        src = pathlib.Path(filepath).read_text()
        bad_patterns = [
            'logger.debug("%s silent skip", exc_info=True)',
            'logger.debug("%s skip", exc_info=True)',
        ]
        for pattern in bad_patterns:
            assert pattern not in src, (
                f"{filepath}: still contains '{pattern}'. "
                "Use logger.warning for unexpected failures so operators see them."
            )
