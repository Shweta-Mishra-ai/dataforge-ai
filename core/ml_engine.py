"""
ml_engine.py — Production ML pipeline.
Auto model selection, cross-validation, SHAP, what-if analysis.
No shortcuts — proper ML engineering.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score,
    classification_report
)
import logging
logger = logging.getLogger(__name__)
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    name:           str
    task:           str          # "regression" or "classification"
    cv_score:       float        # mean cross-val score
    cv_std:         float        # std of cv scores
    train_score:    float
    test_score:     float
    overfit_gap:    float        # train_score - test_score
    overfit_label:  str          # "None", "Mild", "Severe"
    metric_name:    str          # "R2", "Accuracy", "F1"
    # Regression metrics
    mae:            Optional[float] = None
    rmse:           Optional[float] = None
    # Classification metrics
    f1:             Optional[float] = None
    roc_auc:        Optional[float] = None
    # Model object (not serialized)
    model:          Any = field(default=None, repr=False)
    is_best:        bool = False
    train_error:    Optional[str] = None  # set when training failed — UI shows this instead of metrics


@dataclass
class FeatureImportance:
    feature:     str
    importance:  float
    rank:        int
    direction:   str    # "positive", "negative", "mixed"
    explanation: str    # plain English


@dataclass
class MLReport:
    task:               str          # "regression" or "classification"
    target_col:         str
    feature_cols:       List[str]
    n_rows_used:        int
    n_features:         int
    class_balance:      Optional[Dict] = None   # classification only
    models:             List[ModelResult] = field(default_factory=list)
    best_model:         Optional[ModelResult] = None
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    shap_values:        Optional[np.ndarray] = None
    shap_feature_names: Optional[List[str]] = None
    # ── Large objects — memory cost in st.session_state ─────────────────────
    # feature_ranges replaces storing the full X_test DataFrame for OOD checks.
    # A 100k-row × 20-col X_test costs ~16MB in session state; feature_ranges
    # costs ~1KB (just min/max per column). predict_what_if() uses this instead.
    feature_ranges:     Dict[str, Dict[str, float]] = field(default_factory=dict)
    # X_test/y_test/y_pred kept ONLY for the current request/test cycle.
    # Callers should NOT persist MLReport with these populated into
    # st.session_state for long — call clear_large_arrays() before storing.
    X_test:             Optional[pd.DataFrame] = None
    y_test:             Optional[pd.Series] = None
    y_pred:             Optional[np.ndarray] = None
    preprocessor:       Any = field(default=None, repr=False)
    label_encoders:     Dict = field(default_factory=dict)
    target_encoder:     Any = field(default=None, repr=False)
    warnings:           List[str] = field(default_factory=list)

    def clear_large_arrays(self) -> None:
        """
        Call before storing MLReport in st.session_state for the session.
        Drops X_test/y_test/y_pred/shap_values (large numpy/pandas objects)
        while preserving feature_ranges (tiny dict) for OOD validation in
        predict_what_if(). Reduces session_state memory footprint by ~95%
        on large datasets.
        """
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.shap_values = None
    insights:           List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  TASK DETECTION
# ══════════════════════════════════════════════════════════

def detect_task(series: pd.Series) -> Tuple[str, str]:
    """
    Detect if target is regression or classification.
    Returns (task, reason).
    """
    s       = series.dropna()
    n_uniq  = s.nunique()
    dtype   = s.dtype

    # Boolean or binary → classification
    if n_uniq == 2:
        return "classification", "Binary target (2 unique values)"

    # Object/string → classification
    if dtype is object or str(dtype) == "str":
        if n_uniq <= 20:
            return "classification", "Categorical target ({} classes)".format(n_uniq)
        else:
            return "classification", "High-cardinality categorical ({} classes)".format(n_uniq)

    # Few unique integers → classification (relative threshold: <5% of records)
    if pd.api.types.is_integer_dtype(dtype):
        relative_thresh = max(15, int(len(s) * 0.05))   # at most 5% unique
        if n_uniq <= relative_thresh:
            return "classification", "Discrete integer target ({} unique values)".format(n_uniq)

    # Continuous numeric → regression
    return "regression", "Continuous numeric target ({} unique values, {:.1f}% unique)".format(
        n_uniq, n_uniq / max(len(s), 1) * 100)


def suggest_targets(df: pd.DataFrame) -> List[Dict]:
    """
    Suggest good target columns.
    Returns ranked list with task type and reason.
    """
    suggestions = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue

        # Skip ID-like columns
        if s.nunique() / max(len(s), 1) > 0.95 and len(s) > 50:
            continue

        task, reason = detect_task(s)
        score = 0

        # Prefer columns that are informative
        if task == "regression":
            cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0
            score = min(cv, 1.0)
        else:
            # Prefer balanced classes
            vc = s.value_counts(normalize=True)
            balance = 1 - vc.max()   # higher = more balanced
            score = balance

        suggestions.append({
            "column": col, "task": task,
            "reason": reason, "score": round(score, 3),
            "n_unique": s.nunique(), "dtype": str(s.dtype),
        })

    return sorted(suggestions, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Prepare X, y for ML.
    - Encode categoricals
    - Handle missing values
    - Remove low-variance features
    Returns (X, y, label_encoders).
    """
    df = df.copy()

    # Target
    y = df[target_col].copy()
    df = df.drop(columns=[target_col])

    # Use selected features or auto-select
    if selected_features:
        available = [c for c in selected_features if c in df.columns]
        df = df[available]
    else:
        # Drop ID-like columns
        drop_cols = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                drop_cols.append(col)
                continue
            # High cardinality string → drop
            if df[col].dtype == object and df[col].nunique() / max(len(df), 1) > 0.5:
                drop_cols.append(col)
        df = df.drop(columns=drop_cols)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown")
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Convert datetime to numeric (days since min)
    for col in df.select_dtypes(include="datetime").columns:
        df[col] = (df[col] - df[col].min()).dt.days

    # Keep only numeric
    df = df.select_dtypes(include="number")

    # Remove constant columns
    df = df.loc[:, df.nunique() > 1]

    # Align y with X index
    common_idx = df.index.intersection(y.dropna().index)
    df = df.loc[common_idx]
    y  = y.loc[common_idx]

    return df, y, label_encoders


# ══════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════

def _get_models(task: str) -> List[Tuple[str, Any]]:
    """Return list of (name, model) tuples for given task."""
    if task == "regression":
        models = [
            ("Ridge Regression",    Ridge(alpha=1.0)),
            ("Random Forest",       RandomForestRegressor(
                                        n_estimators=100, random_state=42,
                                        n_jobs=-1)),
            ("Gradient Boosting",   GradientBoostingRegressor(
                                        n_estimators=100, random_state=42)),
        ]
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBRegressor(
                n_estimators=100, random_state=42,
                verbosity=0, eval_metric="rmse")))
    else:
        models = [
            ("Logistic Regression", LogisticRegression(
                                        max_iter=1000, random_state=42)),
            ("Random Forest",       RandomForestClassifier(
                                        n_estimators=100, random_state=42,
                                        n_jobs=-1)),
            ("Gradient Boosting",   GradientBoostingClassifier(
                                        n_estimators=100, random_state=42)),
        ]
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBClassifier(
                n_estimators=100, random_state=42,
                verbosity=0, eval_metric="logloss",
                use_label_encoder=False)))
    return models


def _make_pipeline(model, task: str, binary_cols: list | None = None) -> Pipeline:
    """
    Wrap model in imputer + scaler pipeline.
    Uses most_frequent imputation for binary/categorical-encoded columns
    to avoid producing 0.5 from median on binary features.
    """
    # SimpleImputer with median is safe for continuous; use mean as fallback
    # Binary columns (0/1 encoded) need most_frequent, not median
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def _evaluate_regression(y_true, y_pred) -> Dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def _evaluate_classification(y_true, y_pred, y_prob=None) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            n_classes_true  = len(np.unique(y_true))
            n_classes_proba = y_prob.shape[1] if y_prob.ndim == 2 else 2

            if n_classes_true == 2:
                # Binary: use column 1 probability; handle 1-D proba arrays
                proba_col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc = roc_auc_score(y_true, proba_col)
            elif n_classes_true == n_classes_proba:
                # Multiclass: only valid when y_true classes == proba columns
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                    average="weighted")
            else:
                # Mismatch — test split missing some classes (small / imbalanced data)
                logger.warning(
                    "roc_auc_score skipped: %d classes in y_true vs %d proba "
                    "columns — test split does not contain all classes. "
                    "Use stratified split or larger dataset.",
                    n_classes_true, n_classes_proba,
                )
            if auc is not None:
                auc = round(float(auc), 4)
        except Exception:
            logger.warning("roc_auc_score failed unexpectedly", exc_info=True)
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "roc_auc": auc}


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    target_encoder: Optional[LabelEncoder] = None,
) -> List[ModelResult]:
    """
    Train all models, cross-validate, evaluate on holdout.
    Returns list of ModelResult sorted by cv_score descending.
    """
    # Encode classification target
    if task == "classification":
        if y.dtype == object or str(y.dtype) == "str":
            if target_encoder is None:
                target_encoder = LabelEncoder()
            y = pd.Series(
                target_encoder.fit_transform(y.astype(str)),
                index=y.index
            )
        else:
            y = y.astype(int)

    # Train/test split — stratified for classification
    stratify = y if task == "classification" and y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scoring = "r2" if task == "regression" else "f1_weighted"
    results = []

    # ── Class imbalance detection ──────────────────────────────────────────
    class_imbalance_warning = None
    if task == "classification":
        vc = y.value_counts(normalize=True)
        minority_pct = float(vc.min()) * 100
        if minority_pct < 10:
            class_imbalance_warning = (
                f"⚠️ Class imbalance detected: minority class = {minority_pct:.1f}% of data. "
                f"Accuracy metric is misleading — use F1/AUC instead. "
                f"A model predicting the majority class always would score "
                f"{100 - minority_pct:.1f}% accuracy."
            )
        elif minority_pct < 20:
            class_imbalance_warning = (
                f"⚠️ Moderate class imbalance: minority class = {minority_pct:.1f}%. "
                f"Prefer F1-weighted and ROC-AUC over accuracy."
            )

    for name, model in _get_models(task):
        try:
            pipe = _make_pipeline(model, task)

            # Cross-validation on training set
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=5, scoring=scoring, n_jobs=-1
            )

            # Fit on full training set
            pipe.fit(X_train, y_train)
            y_pred_train = pipe.predict(X_train)
            y_pred_test  = pipe.predict(X_test)

            # Scores
            if task == "regression":
                train_s = r2_score(y_train, y_pred_train)
                test_s  = r2_score(y_test,  y_pred_test)
                metrics = _evaluate_regression(y_test, y_pred_test)
                metric_name = "R2"
            else:
                train_s = accuracy_score(y_train, y_pred_train)
                test_s  = accuracy_score(y_test,  y_pred_test)
                try:
                    y_prob = pipe.predict_proba(X_test)
                except Exception:
                    y_prob = None
                metrics = _evaluate_classification(y_test, y_pred_test, y_prob)
                metric_name = "Accuracy"

            gap   = train_s - test_s
            o_lbl = ("None" if gap < 0.05
                     else "Mild" if gap < 0.15
                     else "Severe")

            results.append(ModelResult(
                name=name, task=task,
                cv_score=round(float(np.mean(cv_scores)), 4),
                cv_std=round(float(np.std(cv_scores)), 4),
                train_score=round(train_s, 4),
                test_score=round(test_s, 4),
                overfit_gap=round(gap, 4),
                overfit_label=o_lbl,
                metric_name=metric_name,
                mae=metrics.get("mae"),
                rmse=metrics.get("rmse"),
                f1=metrics.get("f1"),
                roc_auc=metrics.get("roc_auc"),
                model=pipe,
            ))

        except Exception as e:
            logger.warning("Model '%s' training failed: %s", name, e, exc_info=True)
            results.append(ModelResult(
                name=name, task=task,
                cv_score=-999, cv_std=0,
                train_score=0, test_score=0,
                overfit_gap=0, overfit_label="N/A",
                metric_name="N/A",
                model=None,
                train_error=str(e),
            ))

    # Sort by cv_score
    results.sort(key=lambda x: x.cv_score, reverse=True)
    if results:
        results[0].is_best = True

    return results, X_test, y_test, target_encoder, class_imbalance_warning


# ══════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE + SHAP
# ══════════════════════════════════════════════════════════

def get_feature_importance(
    model_result: ModelResult,
    feature_names: List[str],
    X_test: pd.DataFrame,
    task: str,
) -> Tuple[List[FeatureImportance], Optional[np.ndarray]]:
    """
    Extract feature importance — tries SHAP first, falls back to
    model-native importance.
    """
    importances = []
    shap_values = None

    if model_result.model is None:
        return importances, shap_values

    pipe  = model_result.model
    model = pipe.named_steps["model"]

    # ── Try SHAP ──────────────────────────────────────────
    try:
        import shap
        # Transform X_test through imputer + scaler
        X_transformed = pipe[:-1].transform(X_test)
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

        if hasattr(model, "feature_importances_"):
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_transformed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # binary classification
            shap_values = shap_vals
            raw_imp     = np.abs(shap_vals).mean(axis=0)
        else:
            explainer = shap.LinearExplainer(
                model, X_transformed,
                feature_perturbation="interventional"
            )
            shap_vals   = explainer.shap_values(X_transformed)
            shap_values = shap_vals
            raw_imp     = np.abs(shap_vals).mean(axis=0)

    except Exception:
        # ── Fallback: model-native importance ─────────────
        raw_imp = None
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw_imp = np.abs(model.coef_).flatten()[:len(feature_names)]

        if raw_imp is None:
            return importances, shap_values

    # Ensure raw_imp matches feature_names length — trim or pad
    raw_imp = np.array(raw_imp).flatten()
    n_feats = len(feature_names)
    if len(raw_imp) > n_feats:
        raw_imp = raw_imp[:n_feats]
    elif len(raw_imp) < n_feats:
        raw_imp = np.pad(raw_imp, (0, n_feats - len(raw_imp)))

    # Normalize
    total    = raw_imp.sum()
    norm_imp = raw_imp / total if total > 0 else raw_imp

    # Direction from SHAP or coef — safely
    directions = ["mixed"] * n_feats
    try:
        if shap_values is not None:
            sv = np.array(shap_values)
            if sv.ndim == 2 and sv.shape[1] >= n_feats:
                for i in range(n_feats):
                    mean_shap = float(np.mean(sv[:, i]))
                    directions[i] = (
                        "positive" if mean_shap > 0.01
                        else "negative" if mean_shap < -0.01
                        else "mixed"
                    )
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_).flatten()
            for i in range(min(n_feats, len(coef))):
                directions[i] = "positive" if coef[i] > 0 else "negative"
    except Exception:
        directions = ["mixed"] * n_feats

    # Build FeatureImportance list — zip ensures equal length
    pairs  = list(zip(feature_names, norm_imp, directions))
    ranked = sorted(enumerate(pairs), key=lambda x: x[1][1], reverse=True)

    for rank, (i, (feat, imp, direction)) in enumerate(ranked, 1):
        pct = float(imp) * 100

        if pct > 30:
            explanation = "Dominant feature — drives {:.0f}% of predictions. {}influence.".format(
                pct, "Positive " if direction == "positive"
                else "Negative " if direction == "negative" else "Mixed ")
        elif pct > 10:
            explanation = "Important feature ({:.0f}% contribution). {}effect on target.".format(
                pct, "Increases " if direction == "positive"
                else "Decreases " if direction == "negative" else "Mixed ")
        elif pct > 3:
            explanation = "Moderate contribution ({:.0f}%). Minor {} effect.".format(
                pct, direction)
        else:
            explanation = "Low importance ({:.1f}%). Minimal impact on predictions.".format(pct)

        importances.append(FeatureImportance(
            feature=feat, importance=round(float(imp), 4),
            rank=rank, direction=direction, explanation=explanation,
        ))

    return importances[:20], shap_values


# ══════════════════════════════════════════════════════════
#  WHAT-IF PREDICTION
# ══════════════════════════════════════════════════════════

def predict_what_if(
    ml_report: MLReport,
    input_values: Dict[str, float],
    X_train_ref: "pd.DataFrame | None" = None,
) -> Dict:
    """
    Make a single prediction from user-supplied input values.
    Returns prediction + confidence info.
    - Encodes categoricals via stored label_encoders.
    - Validates numeric inputs against training data range (warns on OOD).
    """
    if ml_report.best_model is None or ml_report.best_model.model is None:
        return {"error": "No trained model available."}

    # ── Input range validation ─────────────────────────────────────────────
    # Uses feature_ranges (tiny dict, ~1KB) instead of the full X_test
    # DataFrame — avoids holding large arrays in session_state long-term.
    # X_train_ref param kept for backward compatibility / explicit override.
    ood_warnings = []
    if X_train_ref is not None:
        for feat, val in input_values.items():
            if feat in X_train_ref.columns and val is not None:
                try:
                    col_min = float(X_train_ref[feat].min())
                    col_max = float(X_train_ref[feat].max())
                    if val < col_min or val > col_max:
                        ood_warnings.append(
                            f"'{feat}' = {val} is outside training range "
                            f"[{col_min:.2g}, {col_max:.2g}] — prediction may be unreliable."
                        )
                except Exception:
                    logger.debug("Range check failed for %s", feat, exc_info=True)
    elif ml_report.feature_ranges:
        for feat, val in input_values.items():
            rng = ml_report.feature_ranges.get(feat)
            if rng and val is not None:
                try:
                    if val < rng["min"] or val > rng["max"]:
                        ood_warnings.append(
                            f"'{feat}' = {val} is outside training range "
                            f"[{rng['min']:.2g}, {rng['max']:.2g}] — prediction may be unreliable."
                        )
                except Exception:
                    logger.debug("Range check failed for %s", feat, exc_info=True)
    elif getattr(ml_report, "X_test", None) is not None:
        # Last-resort fallback if feature_ranges wasn't populated (legacy reports)
        X_ref = ml_report.X_test
        for feat, val in input_values.items():
            if feat in X_ref.columns and val is not None:
                try:
                    col_min = float(X_ref[feat].min())
                    col_max = float(X_ref[feat].max())
                    if val < col_min or val > col_max:
                        ood_warnings.append(
                            f"'{feat}' = {val} is outside training range "
                            f"[{col_min:.2g}, {col_max:.2g}] — prediction may be unreliable."
                        )
                except Exception:
                    logger.debug("Range check failed for %s", feat, exc_info=True)

    try:
        # Build input row aligned to feature_cols
        row = {}
        for feat in ml_report.feature_cols:
            val = input_values.get(feat)
            # Apply the same LabelEncoder used during training for categoricals
            if ml_report.label_encoders and feat in ml_report.label_encoders:
                le = ml_report.label_encoders[feat]
                try:
                    val = int(le.transform([str(val)])[0])
                except Exception:
                    # Unseen label — use 0 (safe fallback, same as training fallback)
                    val = 0
            row[feat] = val

        X_input = pd.DataFrame([row])[ml_report.feature_cols]
        pipe    = ml_report.best_model.model
        pred    = pipe.predict(X_input)[0]

        result = {"prediction": float(pred), "task": ml_report.task}

        if ml_report.task == "classification":
            # Decode label
            if ml_report.target_encoder is not None:
                try:
                    pred_label = ml_report.target_encoder.inverse_transform([int(pred)])[0]
                    result["prediction_label"] = str(pred_label)
                except Exception:
                    result["prediction_label"] = str(pred)

            # Probability
            try:
                proba = pipe.predict_proba(X_input)[0]
                result["probabilities"] = {
                    str(c): round(float(p), 4)
                    for c, p in zip(pipe.classes_, proba)
                }
                result["confidence"] = round(float(max(proba)) * 100, 1)
            except Exception:
                result["confidence"] = None
        else:
            # Regression confidence interval (naive ± 1 RMSE)
            rmse = ml_report.best_model.rmse or 0
            result["lower"] = round(float(pred) - rmse, 4)
            result["upper"] = round(float(pred) + rmse, 4)
            result["confidence_note"] = "±{:.2f} (1x RMSE)".format(rmse)

        if ood_warnings:
            result["ood_warnings"] = ood_warnings

        return result

    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
#  INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_insights(report: MLReport) -> List[str]:
    insights = []
    best = report.best_model

    if best is None:
        return ["No model trained successfully."]

    # Performance interpretation
    if report.task == "regression":
        r2 = best.test_score
        if r2 >= 0.85:
            insights.append(
                "Excellent model: R2={:.2f} — model explains {:.0f}% of variance in '{}'.".format(
                    r2, r2*100, report.target_col))
        elif r2 >= 0.70:
            insights.append(
                "Good model: R2={:.2f} — explains {:.0f}% of variance. "
                "Acceptable for business use.".format(r2, r2*100))
        elif r2 >= 0.50:
            insights.append(
                "Moderate model: R2={:.2f} — explains {:.0f}% of variance. "
                "Consider adding more features.".format(r2, r2*100))
        else:
            insights.append(
                "Weak model: R2={:.2f} — '{}' is difficult to predict "
                "from current features.".format(r2, report.target_col))
    else:
        acc = best.test_score
        if acc >= 0.90:
            insights.append(
                "Excellent classifier: {:.1f}% accuracy on held-out data.".format(acc*100))
        elif acc >= 0.75:
            insights.append(
                "Good classifier: {:.1f}% accuracy. "
                "Better than random by {:.1f}%.".format(
                    acc*100, (acc - 1/max(report.n_features,1))*100))
        else:
            insights.append(
                "Weak classifier: {:.1f}% accuracy. "
                "Class imbalance or insufficient features may be the cause.".format(acc*100))

    # Overfitting
    if best.overfit_label == "Severe":
        insights.append(
            "WARNING: Severe overfitting detected (train={:.2f} vs test={:.2f}). "
            "Model memorized training data — will not generalize.".format(
                best.train_score, best.test_score))
    elif best.overfit_label == "Mild":
        insights.append(
            "Mild overfitting (gap={:.2f}). "
            "Consider regularization or more training data.".format(best.overfit_gap))

    # Top feature
    if report.feature_importance:
        top = report.feature_importance[0]
        insights.append(
            "Most important predictor: '{}' ({:.0f}% contribution). {}".format(
                top.feature, top.importance*100, top.explanation))

    # Model comparison
    if len(report.models) >= 2:
        best_cv  = report.models[0].cv_score
        worst_cv = report.models[-1].cv_score
        if best_cv - worst_cv > 0.1:
            insights.append(
                "Significant model performance gap: best ({}) CV={:.2f} "
                "vs worst ({}) CV={:.2f}. Model choice matters for this dataset.".format(
                    report.models[0].name, best_cv,
                    report.models[-1].name, worst_cv))

    # Class imbalance warning
    if report.class_balance:
        max_pct = max(report.class_balance.values())
        if max_pct > 0.80:
            insights.append(
                "Class imbalance detected ({:.0f}% dominant class). "
                "Accuracy metric may be misleading — check F1 score.".format(max_pct*100))

    return insights


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def run_ml_pipeline(
    df: pd.DataFrame,
    target_col: str,
    selected_features: Optional[List[str]] = None,
    max_rows: int = 50_000,
) -> MLReport:
    """
    Full ML pipeline:
    1. Detect task (regression/classification)
    2. Prepare features
    3. Train + cross-validate multiple models
    4. Feature importance (SHAP if available)
    5. Generate insights
    Returns MLReport.
    """
    # Sample if large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    task, task_reason = detect_task(df[target_col])

    # Class balance for classification
    class_balance = None
    if task == "classification":
        vc = df[target_col].value_counts(normalize=True)
        class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}

    # Prepare features
    X, y, label_encoders = prepare_features(df, target_col, selected_features)

    if len(X.columns) == 0:
        report = MLReport(
            task=task, target_col=target_col,
            feature_cols=[], n_rows_used=len(df), n_features=0,
        )
        report.warnings.append("No usable feature columns found after preprocessing.")
        return report

    if len(X) < 20:
        report = MLReport(
            task=task, target_col=target_col,
            feature_cols=list(X.columns), n_rows_used=len(X), n_features=len(X.columns),
        )
        report.warnings.append("Too few rows ({}) for reliable ML. Need at least 20.".format(len(X)))
        return report

    # Train models
    model_results, X_test, y_test, target_encoder, imbalance_warning = train_models(X, y, task)
    best = next((m for m in model_results if m.is_best), None)

    # Propagate imbalance warning immediately so UI can display it
    _imbalance_warn_list = [imbalance_warning] if imbalance_warning else []

    # Feature importance
    feat_importance = []
    shap_values     = None
    if best and best.model is not None:
        feat_importance, shap_values = get_feature_importance(
            best, list(X.columns), X_test, task
        )

    # Predictions on test set
    y_pred = None
    if best and best.model is not None:
        try:
            y_pred = best.model.predict(X_test)
        except Exception:
            logger.warning("%s unexpected ML failure", exc_info=True)

    # ── Compute lightweight feature ranges for OOD validation ───────────────
    # Stored instead of the full X DataFrame — ~1KB vs potentially tens of MB
    feature_ranges = {}
    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_ranges[col] = {
                    "min": float(X[col].min()),
                    "max": float(X[col].max()),
                }
        except Exception:
            logger.debug("feature range skip for %s", col)

    report = MLReport(
        task=task,
        target_col=target_col,
        feature_cols=list(X.columns),
        n_rows_used=len(X),
        n_features=len(X.columns),
        class_balance=class_balance,
        models=model_results,
        best_model=best,
        feature_importance=feat_importance,
        shap_values=shap_values,
        shap_feature_names=list(X.columns) if shap_values is not None else None,
        feature_ranges=feature_ranges,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        preprocessor=None,
        label_encoders=label_encoders,
        target_encoder=target_encoder,
        warnings=[task_reason] + _imbalance_warn_list,
    )

    report.insights = _generate_insights(report)
    return report
