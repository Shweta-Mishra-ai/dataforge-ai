"""
tests/test_property_based.py
Property-based tests using Hypothesis — generates hundreds of random inputs
per test to find edge cases that hand-written fixtures miss.

These tests target the EXACT failure class that caused the production crash
(core/data_cleaner.py quantile TypeError on mixed int/None columns) plus
other high-risk numeric/statistical functions across the codebase.
"""
from __future__ import annotations
import math
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# Suppress the "function-scoped fixture" health check — we use module fixtures
SUPPRESS = [HealthCheck.too_slow, HealthCheck.data_too_large]


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGIES — reusable generators for "messy real-world" data
# ─────────────────────────────────────────────────────────────────────────────

# Mixed int/None list — exact pattern that caused the production crash
mixed_int_none = st.lists(
    st.one_of(st.integers(min_value=-10_000, max_value=10_000), st.none()),
    min_size=15, max_size=200,
)

# Mixed float/None/nan list
mixed_float_nan = st.lists(
    st.one_of(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.none(),
    ),
    min_size=15, max_size=200,
)

# Numeric series with possible extreme outliers
numeric_with_outliers = st.lists(
    st.one_of(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e6, max_value=1e9, allow_nan=False, allow_infinity=False),
    ),
    min_size=15, max_size=100,
)

# Constant series (all same value) — division-by-zero risk class
constant_series = st.one_of(
    st.just([5.0] * 50),
    st.just([0.0] * 50),
    st.just([-3.5] * 30),
)


# ─────────────────────────────────────────────────────────────────────────────
#  core/data_cleaner.py — the exact production crash class
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCleanerProperty:

    @given(vals=mixed_int_none)
    @settings(max_examples=80, suppress_health_check=SUPPRESS, deadline=None)
    def test_auto_clean_never_crashes_on_mixed_int_none(self, vals):
        """
        Property: ANY mix of int/None in a column must not crash auto_clean.
        This is the exact pattern that caused the production TypeError:
        b - a in numpy _lerp when quantile() returns Series instead of scalar.
        """
        from core.data_cleaner import auto_clean
        df = pd.DataFrame({"col": vals, "other": list(range(len(vals)))})
        try:
            cleaned, report = auto_clean(df)
            assert cleaned is not None
        except Exception as e:
            pytest.fail(f"auto_clean crashed on mixed int/None: {e}\nInput: {vals[:10]}...")

    @given(vals=mixed_float_nan)
    @settings(max_examples=80, suppress_health_check=SUPPRESS, deadline=None)
    def test_auto_clean_never_crashes_on_mixed_float_nan(self, vals):
        """Property: ANY mix of float/None must not crash auto_clean."""
        from core.data_cleaner import auto_clean
        df = pd.DataFrame({"col": vals, "other": list(range(len(vals)))})
        try:
            cleaned, report = auto_clean(df)
            assert cleaned is not None
        except Exception as e:
            pytest.fail(f"auto_clean crashed on mixed float/nan: {e}\nInput: {vals[:10]}...")

    @given(vals=numeric_with_outliers)
    @settings(max_examples=50, suppress_health_check=SUPPRESS, deadline=None)
    def test_auto_clean_handles_extreme_outliers(self, vals):
        """Property: extreme outlier magnitudes must not crash quantile-based flagging."""
        from core.data_cleaner import auto_clean
        df = pd.DataFrame({"col": vals})
        cleaned, report = auto_clean(df)
        assert cleaned is not None
        assert len(cleaned) <= len(df)  # cleaning never adds rows


# ─────────────────────────────────────────────────────────────────────────────
#  core/engines/base.py — col_stats() and correlations()
# ─────────────────────────────────────────────────────────────────────────────

class TestColStatsProperty:

    @given(vals=mixed_float_nan)
    @settings(max_examples=80, suppress_health_check=SUPPRESS, deadline=None)
    def test_col_stats_never_crashes(self, vals):
        """Property: col_stats() must handle any float/None mix without crashing."""
        from core.engines.base import col_stats
        s = pd.Series(vals)
        try:
            result = col_stats(s)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"col_stats crashed: {e}\nInput: {vals[:10]}...")

    @given(vals=constant_series)
    @settings(max_examples=10, suppress_health_check=SUPPRESS, deadline=None)
    def test_col_stats_constant_series_no_crash(self, vals):
        """Property: constant series (zero variance) must not crash — common edge case."""
        from core.engines.base import col_stats
        s = pd.Series(vals)
        result = col_stats(s)
        assert isinstance(result, dict)
        if result:
            # std should be 0 or near-0 for constant data
            assert result.get("std", 0) < 1e-6

    @given(
        a=st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=15, max_size=100),
        b=st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=15, max_size=100),
    )
    @settings(max_examples=40, suppress_health_check=SUPPRESS, deadline=None)
    def test_correlations_never_crashes(self, a, b):
        """Property: correlations() must handle any two numeric columns without crashing."""
        from core.engines.base import correlations
        n = min(len(a), len(b))
        df = pd.DataFrame({"a": a[:n], "b": b[:n]})
        try:
            result = correlations(df)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"correlations crashed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  core/story_engine.py — generate_story() on arbitrary structured data
# ─────────────────────────────────────────────────────────────────────────────

class TestStoryEngineProperty:

    @given(
        n_rows=st.integers(min_value=20, max_value=300),
        missing_frac=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=20, suppress_health_check=SUPPRESS, deadline=None)
    def test_generate_story_never_crashes_on_random_structure(self, n_rows, missing_frac):
        """Property: generate_story() must handle any row count / missing fraction."""
        from core.story_engine import generate_story
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "value":    rng.standard_normal(n_rows),
            "category": rng.choice(["A", "B", "C"], n_rows),
            "score":    rng.uniform(0, 100, n_rows),
        })
        # Inject missing values
        mask = rng.random(n_rows) < missing_frac
        df.loc[mask, "value"] = np.nan
        try:
            story = generate_story(df)
            assert story is not None
            assert isinstance(story.key_findings, list)
        except Exception as e:
            pytest.fail(f"generate_story crashed: {e} (n_rows={n_rows}, missing_frac={missing_frac})")


# ─────────────────────────────────────────────────────────────────────────────
#  core/ml_engine.py — detect_task() threshold property
# ─────────────────────────────────────────────────────────────────────────────

class TestMLEngineProperty:

    @given(
        n_unique=st.integers(min_value=2, max_value=50),
        n_rows=st.integers(min_value=50, max_value=5000),
    )
    @settings(max_examples=40, suppress_health_check=SUPPRESS, deadline=None)
    def test_detect_task_never_crashes(self, n_unique, n_rows):
        """Property: detect_task() must handle any unique-value-count / row-count combo."""
        from core.ml_engine import detect_task
        rng = np.random.default_rng(0)
        n_unique = min(n_unique, n_rows)
        s = pd.Series(rng.choice(range(n_unique), n_rows))
        try:
            task, reason = detect_task(s)
            assert task in ("classification", "regression")
            assert isinstance(reason, str)
        except Exception as e:
            pytest.fail(f"detect_task crashed: {e} (n_unique={n_unique}, n_rows={n_rows})")

    @given(
        minority_frac=st.floats(min_value=0.01, max_value=0.5),
        n_rows=st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=15, suppress_health_check=SUPPRESS, deadline=None)
    def test_train_models_imbalance_detection_never_crashes(self, minority_frac, n_rows):
        """Property: any class imbalance ratio must not crash train_models()."""
        from core.ml_engine import train_models, prepare_features
        rng = np.random.default_rng(0)
        n_minority = max(2, int(n_rows * minority_frac))
        n_majority = n_rows - n_minority
        if n_minority < 2 or n_majority < 2:
            return  # skip degenerate splits
        target = np.array([1] * n_minority + [0] * n_majority)
        rng.shuffle(target)
        df = pd.DataFrame({
            "a": rng.standard_normal(n_rows),
            "b": rng.standard_normal(n_rows),
            "target": target,
        })
        X, y, _ = prepare_features(df, "target")
        try:
            results, X_test, y_test, target_enc, warning = train_models(X, y, "classification")
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"train_models crashed: {e} (minority_frac={minority_frac})")


# ─────────────────────────────────────────────────────────────────────────────
#  core/eda_engine.py — analyze_univariate() on arbitrary numeric series
# ─────────────────────────────────────────────────────────────────────────────

class TestEDAEngineProperty:

    @given(vals=mixed_float_nan)
    @settings(max_examples=50, suppress_health_check=SUPPRESS, deadline=None)
    def test_analyze_univariate_never_crashes(self, vals):
        """Property: analyze_univariate() must handle any float/None series."""
        from core.eda_engine import analyze_univariate
        s = pd.Series(vals, name="test_col")
        try:
            result = analyze_univariate(s)
            assert result is not None
            # Backward-compat properties must work
            _ = result.mean
            _ = result.skewness
            _ = result.outlier_pct
        except Exception as e:
            pytest.fail(f"analyze_univariate crashed: {e}\nInput: {vals[:10]}...")


# ─────────────────────────────────────────────────────────────────────────────
#  core/config.py — validate_upload() boundary property
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigProperty:

    @given(
        size_mb=st.floats(min_value=0, max_value=1000, allow_nan=False),
        n_rows=st.integers(min_value=0, max_value=2_000_000),
        n_cols=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=60, suppress_health_check=SUPPRESS, deadline=None)
    def test_validate_upload_never_crashes(self, size_mb, n_rows, n_cols):
        """Property: validate_upload() must handle any size/row/col combination."""
        from core.config import validate_upload, MAX_FILE_SIZE_MB
        errors = validate_upload(size_mb, n_rows, n_cols)
        assert isinstance(errors, list)
        if size_mb > MAX_FILE_SIZE_MB:
            assert any("too large" in e.lower() for e in errors)
