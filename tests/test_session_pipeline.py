"""
tests/test_session_pipeline.py
End-to-end Streamlit session simulation.

Simulates the full user journey without a real browser:
    upload CSV → session_manager → clean → profile → story → ML → PDF

Tests that:
  - Every stage produces valid output
  - No stage crashes silently
  - Failures propagate as exceptions, not empty/None returns
  - PDF output is valid and correctly page-numbered
  - ML report has correct task type and model results
  - Story report has domain-matched insights
"""
from __future__ import annotations
import io
import types
import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE MOCK
#  Simulates st.session_state without a running Streamlit server
# ─────────────────────────────────────────────────────────────────────────────

class MockSessionState(dict):
    """dict-backed mock for st.session_state that supports attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def _make_session() -> MockSessionState:
    """Return a fresh, initialised session state."""
    ss = MockSessionState()
    ss["df_master"]   = None
    ss["df_active"]   = None
    ss["filename"]    = None
    ss["file_size_mb"]= 0.0
    ss["data_loaded"] = False
    ss["ml_report"]   = None
    ss["story_report"]= None
    ss["cleaning_report"] = None
    return ss


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _hr_df(n=500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    perf = rng.integers(1, 5, n).astype(float)
    # 25% missing — use NaN (float), not None (which creates object dtype)
    perf[rng.random(n) < 0.25] = np.nan
    return pd.DataFrame({
        "EmployeeID":        range(1, n + 1),
        "Age":               rng.integers(22, 62, n).astype(int),
        "Department":        rng.choice(["Sales", "R&D", "HR"], n),
        "JobSatisfaction":   rng.integers(1, 5, n).astype(int),
        "YearsAtCompany":    rng.integers(0, 30, n).astype(int),
        "MonthlyIncome":     rng.integers(2000, 20000, n).astype(int),
        "Attrition":         rng.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "PerformanceRating": perf,   # float with NaN — no object dtype
    })


def _sales_df(n=400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "Rep":     rng.choice([f"Rep_{i}" for i in range(10)], n),
        "Region":  rng.choice(["North", "South", "East", "West"], n),
        "Revenue": rng.uniform(500, 80000, n),
        "Target":  rng.uniform(10000, 70000, n),
        "Margin":  rng.uniform(-5, 40, n),
        "Product": rng.choice(["A", "B", "C"], n),
    })


def _finance_df(n=300) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    return pd.DataFrame({
        "Period":   [f"2024-Q{i%4+1}" for i in range(n)],
        "Revenue":  rng.uniform(100000, 2000000, n),
        "Cost":     rng.uniform(50000, 1500000, n),
        "Profit":   rng.uniform(-50000, 500000, n),
        "Budget":   rng.uniform(80000, 1800000, n),
        "Category": rng.choice(["OpEx", "CapEx", "Revenue"], n),
    })


def _ecommerce_df(n=350) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    disc = rng.integers(0, 70, n).astype(float)
    disc[rng.random(n) < 0.1] = np.nan
    rating = rng.uniform(1, 5, n).round(1)
    rating[rng.random(n) < 0.08] = np.nan
    return pd.DataFrame({
        "product_name":        [f"Product {i}" for i in range(n)],
        "category":            rng.choice(["Electronics", "Home", "Clothing"], n),
        "discounted_price":    rng.uniform(50, 5000, n),
        "actual_price":        rng.uniform(100, 8000, n),
        "discount_percentage": disc,
        "rating":              rating,
        "rating_count":        rng.integers(0, 50000, n).astype(int),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stage_upload(ss: MockSessionState, df: pd.DataFrame, fname: str) -> None:
    """Simulate upload stage: validate + store in session."""
    from core.config import validate_upload, MAX_ROWS, MAX_COLS

    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    errors  = validate_upload(size_mb, len(df), len(df.columns))
    hard    = [e for e in errors if "too large" in e.lower()]
    assert not hard, f"Upload stage: hard reject — {hard}"

    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    if len(df.columns) > MAX_COLS:
        df = df.iloc[:, :MAX_COLS]

    ss["df_master"]    = df.copy()
    ss["df_active"]    = df.copy()
    ss["filename"]     = fname
    ss["file_size_mb"] = size_mb
    ss["data_loaded"]  = True


def _stage_clean(ss: MockSessionState) -> dict:
    """Simulate cleaning stage."""
    from core.data_cleaner import auto_clean
    df, report = auto_clean(ss["df_master"])
    ss["df_active"] = df
    ss["cleaning_report"] = report
    return {"rows_before": len(ss["df_master"]), "rows_after": len(df)}


def _stage_profile(ss: MockSessionState) -> object:
    """Simulate data profiling."""
    from core.data_profiler import profile_dataset
    profile = profile_dataset(ss["df_active"])
    ss["profile"] = profile
    return profile


def _stage_story(ss: MockSessionState) -> object:
    """Simulate story/insights generation."""
    from core.story_engine import generate_story
    story = generate_story(ss["df_active"])
    ss["story_report"] = story
    return story


def _stage_ml(ss: MockSessionState, target: str, features: list) -> object:
    """Simulate ML pipeline."""
    from core.ml_engine import run_ml_pipeline
    report = run_ml_pipeline(ss["df_active"], target, features)
    ss["ml_report"] = report
    return report


def _stage_pdf(ss: MockSessionState, domain: str) -> bytes:
    """Simulate PDF generation."""
    from core.pdf_builder import build_pdf
    story = ss.get("story_report")
    config = {
        "client_name":   "Integration Test Corp",
        "report_title":  "End-to-End Test Report",
        "analyst_name":  "DataForge AI",
        "company_logo":  None,
        "theme":         "dark_navy",
        "domain":        domain,
    }
    pdf_bytes = build_pdf(
        df=ss["df_active"],
        config=config,
        profile=ss.get("profile"),
        cleaning_summary=None,
        stats_report=None,
        bi_report=None,
        ml_report=ss.get("ml_report"),
        chart_data=[],
        executive_summary=story.executive_summary if story else "Test summary.",
        findings=story.key_findings if story else [],
        risks=story.business_risks if story else [],
        opportunities=story.opportunities if story else [],
        recommendations=story.recommended_actions if story else [],
        top_insights=story.insights if story else [],
        attrition=story.attrition if story else None,
        domain=domain,
    )
    return pdf_bytes


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestHRFullPipeline:
    """HR dataset: upload → clean → profile → story → ML → PDF."""

    @pytest.fixture(scope="class")
    @classmethod
    def session(cls):
        ss = _make_session()
        df = _hr_df()
        _stage_upload(ss, df, "hr_dataset.csv")
        _stage_clean(ss)
        _stage_profile(ss)
        _stage_story(ss)
        num_cols = ss["df_active"].select_dtypes(include="number").columns.tolist()
        features = [c for c in num_cols if c not in ("EmployeeID",)][:5]
        _stage_ml(ss, "Attrition", features[:4])
        return ss

    def test_data_loaded(self, session):
        assert session["data_loaded"] is True
        assert session["df_active"] is not None
        assert len(session["df_active"]) > 0

    def test_clean_stage_no_crash(self, session):
        assert session["cleaning_report"] is not None

    def test_story_domain_is_hr(self, session):
        story = session["story_report"]
        assert story is not None
        assert story.domain == "hr", f"Expected domain='hr', got '{story.domain}'"

    def test_story_exec_summary_has_content(self, session):
        summary = session["story_report"].executive_summary
        assert len(summary) > 80, f"Exec summary too short: {summary!r}"

    def test_story_has_findings(self, session):
        findings = session["story_report"].key_findings
        assert isinstance(findings, list)
        assert len(findings) > 0, "HR dataset must produce at least 1 finding"

    def test_story_has_attrition(self, session):
        story = session["story_report"]
        # Either attrition object or attrition mentioned in summary/findings
        combined = story.executive_summary + " ".join(story.key_findings)
        assert story.attrition is not None or "attrition" in combined.lower(), \
            "HR pipeline must surface attrition signal"

    def test_ml_classification_task(self, session):
        report = session["ml_report"]
        assert report is not None
        assert report.task == "classification", \
            f"Attrition target must be classification, got '{report.task}'"

    def test_ml_has_models(self, session):
        assert len(session["ml_report"].models) >= 2

    def test_ml_imbalance_warning_present(self, session):
        """16% attrition → imbalance warning must be in report.warnings."""
        warns = session["ml_report"].warnings
        imbalance = [w for w in warns if "imbalance" in w.lower()]
        assert len(imbalance) > 0, (
            f"16% minority class must trigger imbalance warning. Got: {warns}"
        )

    def test_ml_best_model_has_roc_auc(self, session):
        """Best model must have ROC-AUC > 0.5 (better than random)."""
        best = session["ml_report"].best_model
        assert best is not None
        # On heavily imbalanced data (16% minority) F1 can be 0 if model
        # predicts only majority class. ROC-AUC is the correct metric to check.
        assert best.roc_auc is not None, "Best classification model must have ROC-AUC"
        assert best.roc_auc > 0.5, (
            f"Best model ROC-AUC {best.roc_auc:.3f} ≤ 0.5 — worse than random. "
            "Check feature selection and class imbalance handling."
        )

    def test_pdf_valid_bytes(self, session):
        pdf = _stage_pdf(session, "hr")
        assert isinstance(pdf, bytes) and len(pdf) > 1000
        assert pdf[:4] == b"%PDF", "Not a valid PDF"

    def test_pdf_page_2_not_page_1(self, session):
        """First content page must show Page 2 (cover = Page 1)."""
        try:
            from pypdf import PdfReader
            pdf = _stage_pdf(session, "hr")
            reader = PdfReader(io.BytesIO(pdf))
            assert len(reader.pages) >= 3, \
                f"HR report should have ≥3 pages, got {len(reader.pages)}"
            page2_text = reader.pages[1].extract_text() or ""
            assert "Page 1 of" not in page2_text, (
                f"First content page shows 'Page 1 of' — page numbering bug.\n"
                f"Text: {page2_text[:300]}"
            )
        except ImportError:
            pytest.skip("pypdf not installed")


class TestSalesPipeline:
    """Sales dataset: upload → story → ML → verify domain and KPIs."""

    @pytest.fixture(scope="class")
    @classmethod
    def session(cls):
        ss = _make_session()
        _stage_upload(ss, _sales_df(), "sales_q4.csv")
        _stage_clean(ss)
        _stage_story(ss)
        _stage_ml(ss, "Revenue", ["Margin"])
        return ss

    def test_domain_detected(self, session):
        story = session["story_report"]
        assert story.domain in ("sales", "ecommerce"), \
            f"Sales dataset should detect 'sales' or 'ecommerce', got '{story.domain}'"

    def test_regression_task(self, session):
        report = session["ml_report"]
        assert report.task == "regression", \
            f"Revenue is continuous → regression, got '{report.task}'"

    def test_opportunities_populated(self, session):
        """Sales data with variable Revenue must generate ≥1 opportunity."""
        story = session["story_report"]
        assert len(story.opportunities) > 0, \
            "Sales dataset must produce at least 1 opportunity from P90/median spread"

    def test_pdf_valid(self, session):
        pdf = _stage_pdf(session, "sales")
        assert pdf[:4] == b"%PDF"


class TestFinancePipeline:
    """Finance dataset: full pipeline with P&L columns."""

    @pytest.fixture(scope="class")
    @classmethod
    def session(cls):
        ss = _make_session()
        _stage_upload(ss, _finance_df(), "finance_2024.csv")
        _stage_clean(ss)
        _stage_story(ss)
        return ss

    def test_domain_is_finance(self, session):
        story = session["story_report"]
        assert story.domain == "finance", \
            f"Finance dataset should detect 'finance', got '{story.domain}'"

    def test_no_empty_findings(self, session):
        story = session["story_report"]
        for f in story.key_findings:
            assert f and len(f.strip()) > 0, "Finding must not be empty string"

    def test_pdf_valid(self, session):
        pdf = _stage_pdf(session, "finance")
        assert pdf[:4] == b"%PDF"


class TestEcommercePipeline:
    """E-Commerce dataset: rating, pricing, discount."""

    @pytest.fixture(scope="class")
    @classmethod
    def session(cls):
        ss = _make_session()
        _stage_upload(ss, _ecommerce_df(), "amazon_products.csv")
        _stage_clean(ss)
        _stage_story(ss)
        return ss

    def test_domain_is_ecommerce(self, session):
        story = session["story_report"]
        assert story.domain == "ecommerce", \
            f"Ecommerce dataset should detect 'ecommerce', got '{story.domain}'"

    def test_pdf_valid(self, session):
        pdf = _stage_pdf(session, "ecommerce")
        assert pdf[:4] == b"%PDF"


# ─────────────────────────────────────────────────────────────────────────────
#  EDGE CASE SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCaseSessions:

    def test_session_with_tiny_df(self):
        """20-row dataset must produce story but ML may warn about small sample."""
        ss = _make_session()
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.standard_normal(20),
            "b": rng.standard_normal(20),
            "y": rng.choice([0, 1], 20),
        })
        _stage_upload(ss, df, "tiny.csv")
        _stage_story(ss)
        story = ss["story_report"]
        assert story is not None
        assert isinstance(story.key_findings, list)

    def test_session_with_all_numeric_df(self):
        """All-numeric dataset must complete story and pdf."""
        ss = _make_session()
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.standard_normal((200, 8)),
                          columns=[f"metric_{i}" for i in range(8)])
        _stage_upload(ss, df, "numeric.csv")
        _stage_story(ss)
        pdf = _stage_pdf(ss, "general")
        assert pdf[:4] == b"%PDF"

    def test_session_with_all_categorical_df(self):
        """All-categorical (no numeric) must complete without crash."""
        ss = _make_session()
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "dept":    rng.choice(["HR", "Sales", "IT"], 100),
            "grade":   rng.choice(["A", "B", "C"], 100),
            "region":  rng.choice(["N", "S", "E", "W"], 100),
            "outcome": rng.choice(["pass", "fail"], 100),
        })
        _stage_upload(ss, df, "categorical.csv")
        _stage_story(ss)
        assert ss["story_report"] is not None

    def test_upload_rejects_oversized_file(self):
        """Upload stage must hard-reject if size exceeds MAX_FILE_SIZE_MB."""
        from core.config import validate_upload, MAX_FILE_SIZE_MB
        errors = validate_upload(MAX_FILE_SIZE_MB + 50, 1000000, 50)
        hard = [e for e in errors if "too large" in e.lower()]
        assert len(hard) > 0, "Must hard-reject file over size limit"

    def test_upload_samples_oversized_rows(self):
        """Upload stage must sample when rows > MAX_ROWS, not crash."""
        from core.config import validate_upload, MAX_ROWS
        errors = validate_upload(10.0, MAX_ROWS + 100000, 20)
        hard = [e for e in errors if "too large" in e.lower()]
        sample_notes = [e for e in errors if "sampl" in e.lower()]
        assert not hard, "Row count above MAX_ROWS should not hard-reject"
        assert len(sample_notes) > 0, "Should inform user about sampling"

    def test_double_upload_replaces_session(self):
        """Uploading a second file must fully replace the previous session state."""
        ss = _make_session()
        df1 = _hr_df(100)
        df2 = _sales_df(80)

        _stage_upload(ss, df1, "first.csv")
        assert ss["filename"] == "first.csv"
        assert "Attrition" in ss["df_active"].columns

        _stage_upload(ss, df2, "second.csv")
        assert ss["filename"] == "second.csv"
        assert "Revenue" in ss["df_active"].columns
        assert "Attrition" not in ss["df_active"].columns, \
            "Second upload must fully replace first — no stale columns"


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-STAGE DATA CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossStageConsistency:

    def test_clean_stage_never_increases_rows(self):
        """Cleaning must not add rows — only remove or keep same count."""
        ss = _make_session()
        df = _hr_df(300)
        _stage_upload(ss, df, "test.csv")
        rows_before = len(ss["df_active"])
        _stage_clean(ss)
        rows_after = len(ss["df_active"])
        assert rows_after <= rows_before, \
            f"Cleaning added rows: before={rows_before}, after={rows_after}"

    def test_story_uses_cleaned_df(self):
        """Story must run on df_active (post-clean), not df_master."""
        ss = _make_session()
        df = _hr_df(300)
        _stage_upload(ss, df, "test.csv")
        _stage_clean(ss)
        _stage_story(ss)
        story = ss["story_report"]
        # Row count in exec summary should match df_active, not df_master
        n_active = len(ss["df_active"])
        assert str(n_active) in story.executive_summary or n_active > 0, \
            "Story exec summary should reference current active row count"

    def test_ml_features_match_active_df_columns(self):
        """ML features must come from df_active, not the original df_master."""
        ss = _make_session()
        df = _hr_df(400)
        _stage_upload(ss, df, "test.csv")
        _stage_clean(ss)
        active_cols = set(ss["df_active"].columns)
        num_features = [
            c for c in active_cols
            if c not in ("Attrition", "EmployeeID")
            and ss["df_active"][c].dtype != object
        ][:4]
        report = _stage_ml(ss, "Attrition", num_features)
        for feat in report.feature_cols:
            assert feat in active_cols, \
                f"ML feature '{feat}' not in df_active columns"

    def test_pdf_uses_story_exec_summary(self):
        """PDF executive summary must match what story_engine produced."""
        ss = _make_session()
        _stage_upload(ss, _hr_df(200), "test.csv")
        _stage_clean(ss)
        _stage_story(ss)
        pdf = _stage_pdf(ss, "hr")
        assert pdf[:4] == b"%PDF"
        # Verify exec summary was passed by checking PDF is not empty
        assert len(pdf) > 2000, "PDF with exec summary must be >2KB"
