import pandas as pd
import numpy as np
import pytest



# ── Regression: quantile crash on mixed-type / object-backed numeric columns ──
class TestQuantileRegressionPy314:
    """
    Regression tests for the production crash on Python 3.14 + pandas 2.x:
        TypeError: b - a  (in numpy _lerp)
    Caused by quantile() returning a Series/DataFrame instead of a scalar
    when the column has object dtype even after is_numeric_dtype() passes.
    """

    def test_clean_integer_column_with_none_values(self):
        """Column built from [int, None] produces object dtype — must not crash."""
        from core.data_cleaner import auto_clean
        n = 100
        rng = np.random.default_rng(0)
        # None in a list forces object dtype even for integer data
        vals = [int(x) if i % 10 != 0 else None
                for i, x in enumerate(rng.integers(1, 1000, n))]
        df = pd.DataFrame({"salary": vals, "dept": ["HR"] * n})
        cleaned, report = auto_clean(df)
        assert cleaned is not None
        assert len(cleaned) > 0

    def test_clean_float_column_with_string_mixed(self):
        """Column with currency strings mixed in (e.g. '$1,234') must not crash."""
        from core.data_cleaner import auto_clean
        df = pd.DataFrame({
            "revenue": ["$1,234", "5000", "3200.50", None, "8900", "$500"],
            "region":  ["N", "S", "E", "W", "N", "S"],
        })
        cleaned, report = auto_clean(df)
        assert cleaned is not None

    def test_clean_normal_numeric_still_works(self):
        """Normal float column must still flag extreme outliers after the fix."""
        from core.data_cleaner import auto_clean
        rng = np.random.default_rng(42)
        vals = list(rng.normal(50, 5, 200))
        vals[0] = 99999.0  # extreme outlier
        vals[1] = -99999.0
        df = pd.DataFrame({"score": vals})
        cleaned, report = auto_clean(df)
        actions_text = " ".join(str(a.action) for a in report.actions)
        assert "outlier" in actions_text.lower() or "flagged" in actions_text.lower(), \
            "Extreme outliers must still be flagged after the quantile fix"

    def test_clean_large_messy_dataset_no_crash(self):
        """Real-world messy HR dataset must complete without TypeError."""
        from core.data_cleaner import auto_clean
        rng = np.random.default_rng(7)
        n = 500
        # Mix of None and int → object dtype
        salary = [int(rng.integers(2000, 20000)) if i % 8 != 0 else None
                  for i in range(n)]
        df = pd.DataFrame({
            "salary":      salary,
            "age":         rng.integers(20, 65, n).tolist(),
            "dept":        rng.choice(["HR", "Sales", "IT"], n).tolist(),
            "left":        rng.choice(["Yes", "No"], n).tolist(),
            "satisfaction": [round(float(x), 2) if i % 5 != 0 else None
                             for i, x in enumerate(rng.uniform(0, 1, n))],
        })
        cleaned, report = auto_clean(df)
        assert cleaned is not None
        assert len(cleaned) > 0
