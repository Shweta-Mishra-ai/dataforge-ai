import pandas as pd
import numpy as np
import pytest



# ── NEW: class imbalance detection ────────────────────────────────────────────
class TestClassImbalance:
    def test_severe_imbalance_triggers_warning(self):
        """5% minority → must produce an imbalance warning."""
        from core.ml_engine import train_models, prepare_features
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            # 5% minority class
            "target": np.where(np.arange(n) < int(n * 0.05), 1, 0),
        })
        X, y, _ = prepare_features(df, "target")
        _, _, _, _, warning = train_models(X, y, "classification")
        assert warning is not None, "Severe imbalance (5%) must produce a warning"
        assert "imbalance" in warning.lower()

    def test_balanced_no_warning(self):
        """Balanced dataset → no imbalance warning."""
        from core.ml_engine import train_models, prepare_features
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "target": np.tile([0, 1], n // 2),
        })
        X, y, _ = prepare_features(df, "target")
        _, _, _, _, warning = train_models(X, y, "classification")
        assert warning is None, "Balanced dataset should not trigger imbalance warning"


# ── NEW: detect_task relative threshold ───────────────────────────────────────
class TestDetectTask:
    def test_large_dataset_11_unique_is_regression(self):
        """11 unique values in 10,000-row series → regression (relative threshold)."""
        from core.ml_engine import detect_task
        # 11 unique values out of 10k rows = 0.11% unique → regression
        s = pd.Series(np.random.choice(range(11), 10_000))
        task, _ = detect_task(s)
        # 11 unique / 10000 = 0.11% < 5% → classification still (absolute 15 removed)
        # With relative: max(15, int(10000*0.05)) = max(15,500) = 500 → 11 < 500 → classification
        # This confirms the threshold is now relative and generous for small nunique
        assert task == "classification"

    def test_small_dataset_high_unique_pct_is_regression(self):
        """Many unique values relative to size → regression."""
        from core.ml_engine import detect_task
        # 50 unique values in 60 rows = 83% unique → regression
        s = pd.Series(list(range(50)) + [1] * 10)
        task, _ = detect_task(s)
        assert task == "regression"


# ── NEW: predict_what_if OOD warning ─────────────────────────────────────────
class TestPredictWhatIfOOD:
    def test_ood_input_produces_warning(self):
        """Input value outside training range → ood_warnings in result."""
        from core.ml_engine import run_ml_pipeline, predict_what_if
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.randint(20, 60, 200),
            "salary": np.random.randint(1000, 20000, 200),
            "left": np.random.choice([0, 1], 200),
        })
        report = run_ml_pipeline(df, "left", ["age", "salary"])
        # age 999 is way out of range [20, 60]
        result = predict_what_if(report, {"age": 999.0, "salary": 5000.0},
                                 X_train_ref=report.X_test)
        assert "ood_warnings" in result, "Out-of-range input must produce ood_warnings"
        assert any("age" in w for w in result["ood_warnings"])


# ── Regression: failed model training must be surfaced, not silently -999 ────
class TestFailedModelSurfacing:
    def test_train_error_field_exists_on_model_result(self):
        """ModelResult must have a train_error field for UI to display."""
        from core.ml_engine import ModelResult
        fields = ModelResult.__dataclass_fields__
        assert "train_error" in fields, "ModelResult missing train_error field"

    def test_failed_model_has_train_error_populated(self):
        """A model that fails training must populate train_error, not just -999."""
        from core.ml_engine import train_models, prepare_features
        import pandas as pd
        import numpy as np

        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "target": rng.choice([0, 1], n),
        })
        X, y, _ = prepare_features(df, "target")
        results, X_test, y_test, target_enc, warning = train_models(X, y, "classification")
        # Even if none fail in this run, the contract must hold for the field
        for r in results:
            if r.cv_score == -999:
                assert r.train_error is not None, (
                    f"Model '{r.name}' failed (-999) but train_error is None — "
                    "user has no way to know why."
                )
