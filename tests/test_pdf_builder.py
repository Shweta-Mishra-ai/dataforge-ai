"""tests/test_pdf_builder.py — Tests for critical pdf_builder paths"""
import io
import pytest
import pandas as pd


def _sample_df():
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "Dave"],
        "age":    [30, 25, 35, 28],
        "salary": [50000, 60000, 55000, 62000],
        "dept":   ["HR", "Eng", "HR", "Eng"],
    })


def test_build_pdf_returns_bytes():
    """build_pdf must return non-empty bytes without raising."""
    from core.pdf_builder import build_pdf
    df = _sample_df()
    config = {
        "title": "Test Report", "subtitle": "Automated",
        "client_name": "Test Co", "analyst_name": "Tester",
        "confidential": False, "theme_name": "Corporate Light",
        "logo_path": "", "logo_bytes": None, "avg_salary_k": 50,
    }
    pdf = build_pdf(
        df=df, config=config, profile=None,
        executive_summary="Test summary.",
        findings=["Finding 1"], risks=[], opportunities=[],
        recommendations=["Action 1"], top_insights=[],
        domain="general",
    )
    assert isinstance(pdf, bytes)
    assert len(pdf) > 1000  # real PDF, not empty


def test_build_pdf_with_none_profile():
    """profile=None must not cause NameError (was the shipped bug)."""
    from core.pdf_builder import build_pdf
    df = _sample_df()
    config = {
        "title": "Test", "subtitle": "", "client_name": "",
        "analyst_name": "", "confidential": False,
        "theme_name": "Corporate Light", "logo_path": "",
        "logo_bytes": None, "avg_salary_k": 50,
    }
    # Must NOT raise NameError
    pdf = build_pdf(df=df, config=config, profile=None, domain="general")
    assert isinstance(pdf, bytes)


def test_build_pdf_all_themes():
    """Every theme must produce a valid PDF."""
    from core.pdf_builder import build_pdf, THEMES
    df = _sample_df()
    for theme_name in THEMES:
        config = {
            "title": f"Test {theme_name}", "subtitle": "",
            "client_name": "", "analyst_name": "",
            "confidential": False, "theme_name": theme_name,
            "logo_path": "", "logo_bytes": None, "avg_salary_k": 50,
        }
        pdf = build_pdf(df=df, config=config, profile=None, domain="general")
        assert len(pdf) > 500, f"Theme {theme_name} produced empty PDF"


def test_appendix_does_not_crash_without_profile():
    """Regression test for the shipped NameError bug."""
    from core.pdf_builder import _appendix
    # Must not raise NameError for profile or df
    story = []
    s = {"h1": None, "h2": None, "h3": None, "body": None, "note": None}
    T = {"primary": "#000000", "accent": "#000000"}
    # This was crashing before the fix
    try:
        _appendix(story, s, T, {}, 400, domain="general",
                  df=_sample_df(), profile=None)
    except Exception as e:
        # ReportLab style errors are OK (we're not building a real doc)
        # NameError is NOT OK
        assert "profile" not in str(e) and "df" not in str(e), \
            f"NameError still present: {e}"
