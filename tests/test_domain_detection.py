"""tests/test_domain_detection.py — Tests for domain/niche detection."""
import pandas as pd
import pytest
from core.story_engine import detect_domain


def test_detects_hr_domain():
    df = pd.DataFrame({"employee_id": [1], "attrition": ["Yes"],
                        "department": ["HR"], "salary": [50000]})
    domain, conf = detect_domain(df)
    assert domain == "hr"
    assert conf > 0


def test_detects_ecommerce_domain():
    df = pd.DataFrame({"product": ["X"], "price": [9.99],
                        "order_id": [1], "customer": ["Alice"],
                        "rating": [4.5], "category": ["Electronics"]})
    domain, conf = detect_domain(df)
    assert domain == "ecommerce"


def test_detects_finance_domain():
    df = pd.DataFrame({"revenue": [1000], "expense": [800],
                        "profit": [200], "margin": [0.2],
                        "budget": [1100], "cost": [800]})
    domain, conf = detect_domain(df)
    assert domain == "finance"


def test_detects_sales_domain():
    df = pd.DataFrame({"revenue": [5000], "deal": ["Won"],
                        "pipeline": ["Q1"], "quota": [10000],
                        "lead": ["ABC Corp"], "forecast": [8000]})
    domain, conf = detect_domain(df)
    assert domain in ("sales", "finance")  # overlap is acceptable


def test_general_for_unknown():
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    domain, conf = detect_domain(df)
    # Should not crash; may return "general" or any domain with low conf
    assert isinstance(domain, str)
    assert 0.0 <= conf <= 1.0


def test_returns_tuple():
    df = pd.DataFrame({"employee": [1], "salary": [50000]})
    result = detect_domain(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
