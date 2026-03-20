"""
pages/5_ML_Predictions.py
Auto ML — model selection, SHAP, what-if simulator.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.session_manager import require_data, get_df, get_filename, cache_ml_result, get_cached_ml

st.set_page_config(page_title="ML Predictions", layout="wide")

require_data()
df      = get_df()
fname   = get_filename()

from core.ml_engine import run_ml_pipeline, suggest_targets, detect_task, predict_what_if

COLORS = ["#1a4a8a", "#2196F3", "#22d3a5", "#f7934f", "#a78bfa", "#f77070"]

st.markdown("## ML Predictions")
st.caption("{} — {:,} rows, {} columns".format(fname, len(df), len(df.columns)))
st.divider()

# ══════════════════════════════════════════════════════════
#  STEP 1 — TARGET SELECTION
# ══════════════════════════════════════════════════════════
st.markdown("### Step 1 — Select Target Column")
st.caption("What do you want to predict?")

suggestions = suggest_targets(df)
suggested_cols = [s["column"] for s in suggestions[:5]]

col_a, col_b = st.columns([2, 1])
with col_a:
    all_cols   = df.columns.tolist()
    default_ix = 0
    if suggestions:
        best_col = suggestions[0]["column"]
        if best_col in all_cols:
            default_ix = all_cols.index(best_col)

    target_col = st.selectbox(
        "Target column (what to predict)",
        all_cols, index=default_ix,
        help="DataForge AI will auto-detect if this is regression or classification."
    )

with col_b:
    if target_col:
        task, reason = detect_task(df[target_col])
        if task == "regression":
            st.success("REGRESSION\n\n{}".format(reason))
        else:
            st.info("CLASSIFICATION\n\n{}".format(reason))

# Top suggestions
if suggestions[:3]:
    st.markdown("**Suggested targets:**")
    cols_s = st.columns(min(3, len(suggestions)))
    for i, sug in enumerate(suggestions[:3]):
        with cols_s[i]:
            st.markdown(
                "<div style='background:#f0f4ff;border-radius:8px;"
                "padding:10px 14px;border:1px solid #d0d8f0'>"
                "<b>{}</b><br>"
                "<span style='font-size:12px;color:#646882'>{} | {}</span>"
                "</div>".format(sug["column"], sug["task"].title(), sug["reason"]),
                unsafe_allow_html=True
            )

st.divider()

# ══════════════════════════════════════════════════════════
#  STEP 2 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════
st.markdown("### Step 2 — Select Features")
st.caption("Which columns should the model use to predict? (Unselect ID columns)")

available_features = [c for c in df.columns if c != target_col]

# Auto-suggest: exclude obvious IDs
smart_defaults = []
for col in available_features:
    s = df[col].dropna()
    if s.nunique() / max(len(s), 1) > 0.95 and len(s) > 50:
        continue
    if any(kw in col.lower() for kw in ["_id", "id_", "url", "link", "uuid", "img"]):
        continue
    smart_defaults.append(col)

selected_features = st.multiselect(
    "Feature columns",
    available_features,
    default=smart_defaults[:15],
    help="Start with smart defaults. Remove any ID or irrelevant columns."
)

if not selected_features:
    st.warning("Select at least one feature column.")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════
#  STEP 3 — TRAIN
# ══════════════════════════════════════════════════════════
st.markdown("### Step 3 — Train Models")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    train_btn = st.button("Train All Models", type="primary",
                          use_container_width=True)
with col_info:
    st.caption(
        "Trains Ridge/Logistic Regression, Random Forest, "
        "Gradient Boosting{} with 5-fold cross-validation.".format(
            ", XGBoost" if True else ""
        )
    )

# Check cache
cached = get_cached_ml()
report = None

if train_btn:
    with st.spinner("Training models... this may take 30-60 seconds."):
        report = run_ml_pipeline(df, target_col, selected_features)
        cache_ml_result(report)
    st.success("Training complete!")
elif cached is not None:
    report = cached
    st.info("Showing cached results. Click 'Train All Models' to retrain.")

if report is None:
    st.stop()

# Warnings
if report.warnings:
    with st.expander("Warnings ({})".format(len(report.warnings))):
        for w in report.warnings:
            st.warning(w)

if not report.models:
    st.error("No models trained. Check warnings above.")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════
#  RESULTS TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Model Comparison",
    "Feature Importance",
    "What-If Simulator",
    "Insights",
])

# ── Tab 1: Model Comparison ───────────────────────────────
with tab1:
    st.markdown("### Model Comparison")
    st.caption("{} models trained | Target: '{}' | Task: {}".format(
        len(report.models), report.target_col, report.task.title()))

    # Summary table
    rows = []
    for m in report.models:
        if m.cv_score == -999:
            continue
        row = {
            "Model":         m.name,
            "CV Score":      "{:.4f} ±{:.4f}".format(m.cv_score, m.cv_std),
            "Train Score":   "{:.4f}".format(m.train_score),
            "Test Score":    "{:.4f}".format(m.test_score),
            "Overfit":       m.overfit_label,
            "Best?":         "YES" if m.is_best else "",
        }
        if report.task == "regression":
            row["MAE"]  = "{:.4f}".format(m.mae) if m.mae else "-"
            row["RMSE"] = "{:.4f}".format(m.rmse) if m.rmse else "-"
        else:
            row["F1 Score"] = "{:.4f}".format(m.f1) if m.f1 else "-"
            row["ROC AUC"]  = "{:.4f}".format(m.roc_auc) if m.roc_auc else "-"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    # Bar chart comparison
    valid_models = [m for m in report.models if m.cv_score != -999]
    if valid_models:
        fig = go.Figure()
        names   = [m.name for m in valid_models]
        cv_sc   = [m.cv_score for m in valid_models]
        test_sc = [m.test_score for m in valid_models]

        fig.add_trace(go.Bar(
            name="CV Score (5-fold)", x=names, y=cv_sc,
            marker_color=COLORS[0], text=["{:.3f}".format(s) for s in cv_sc],
            textposition="outside"
        ))
        fig.add_trace(go.Bar(
            name="Test Score", x=names, y=test_sc,
            marker_color=COLORS[2], text=["{:.3f}".format(s) for s in test_sc],
            textposition="outside"
        ))
        fig.update_layout(
            title="Model Performance Comparison",
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8faff",
            font=dict(family="Helvetica", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Best model highlight
    if report.best_model:
        best = report.best_model
        st.markdown("#### Best Model: {}".format(best.name))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CV Score",   "{:.4f}".format(best.cv_score),
                  help="5-fold cross-validation mean score")
        c2.metric("Test Score", "{:.4f}".format(best.test_score),
                  help="Held-out 20% test set")
        c3.metric("Overfit",    best.overfit_label,
                  delta="Gap: {:.3f}".format(best.overfit_gap),
                  delta_color="inverse" if best.overfit_label != "None" else "off")
        if report.task == "regression":
            c4.metric("RMSE", "{:.4f}".format(best.rmse) if best.rmse else "-")
        else:
            c4.metric("F1 Score", "{:.4f}".format(best.f1) if best.f1 else "-")

    # Classification extras
    if report.task == "classification" and report.class_balance:
        st.markdown("#### Class Distribution")
        labels = list(report.class_balance.keys())
        values = list(report.class_balance.values())
        fig_pie = px.pie(
            values=values, names=labels,
            title="Target Class Balance",
            color_discrete_sequence=COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        if max(values) > 0.80:
            st.warning(
                "Imbalanced classes detected ({:.0f}% dominant). "
                "Accuracy can be misleading — rely on F1 score.".format(
                    max(values)*100))

# ── Tab 2: Feature Importance ─────────────────────────────
with tab2:
    st.markdown("### Feature Importance")
    if not report.feature_importance:
        st.info("Feature importance not available for this model.")
    else:
        st.caption("Based on {} — top 20 features shown.".format(
            "SHAP values" if report.shap_values is not None
            else "model-native importance"
        ))

        fi = report.feature_importance[:15]

        # Horizontal bar chart
        feat_names = [f.feature for f in fi]
        feat_imp   = [f.importance * 100 for f in fi]
        feat_dir   = [f.direction for f in fi]
        bar_colors = [
            "#2196F3" if d == "positive"
            else "#f77070" if d == "negative"
            else "#a78bfa"
            for d in feat_dir
        ]

        fig = go.Figure(go.Bar(
            x=feat_imp, y=feat_names,
            orientation="h",
            marker_color=bar_colors,
            text=["{:.1f}%".format(v) for v in feat_imp],
            textposition="outside",
        ))
        fig.update_layout(
            title="Feature Importance (% contribution)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8faff",
            font=dict(family="Helvetica", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(autorange="reversed"),
            height=max(300, len(fi) * 35),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        c1, c2, c3 = st.columns(3)
        c1.info("Blue — Positive effect on target")
        c2.error("Red — Negative effect on target")
        c3.warning("Purple — Mixed/nonlinear effect")

        # Detail table
        st.markdown("#### Feature Detail")
        rows = []
        for f in fi:
            rows.append({
                "Rank":        f.rank,
                "Feature":     f.feature,
                "Importance":  "{:.1f}%".format(f.importance * 100),
                "Direction":   f.direction.title(),
                "Explanation": f.explanation,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

# ── Tab 3: What-If Simulator ──────────────────────────────
with tab3:
    st.markdown("### What-If Simulator")
    st.caption(
        "Change feature values below and see how the prediction changes. "
        "Uses the best model: {}.".format(
            report.best_model.name if report.best_model else "N/A"
        )
    )

    if not report.best_model or report.best_model.model is None:
        st.info("Train models first to use the simulator.")
    elif not report.feature_cols:
        st.info("No features available.")
    else:
        # Show top 8 features as sliders
        top_features = (
            [f.feature for f in report.feature_importance[:8]]
            if report.feature_importance
            else report.feature_cols[:8]
        )

        input_vals = {}
        n_cols = 2
        pairs  = [top_features[i:i+n_cols]
                  for i in range(0, len(top_features), n_cols)]

        for pair in pairs:
            cols_ui = st.columns(n_cols)
            for j, feat in enumerate(pair):
                if feat not in df.columns:
                    continue
                s = df[feat].dropna()
                if pd.api.types.is_numeric_dtype(s):
                    lo  = float(s.min())
                    hi  = float(s.max())
                    med = float(s.median())
                    if lo < hi:
                        val = cols_ui[j].slider(
                            feat,
                            min_value=round(lo, 2),
                            max_value=round(hi, 2),
                            value=round(med, 2),
                            key="whatif_{}".format(feat),
                            help="Median: {:.2f}".format(med)
                        )
                    else:
                        val = med
                        cols_ui[j].metric(feat, "{:.2f}".format(val))
                    input_vals[feat] = val
                else:
                    unique_vals = s.unique().tolist()[:20]
                    val = cols_ui[j].selectbox(
                        feat, unique_vals,
                        key="whatif_{}".format(feat)
                    )
                    input_vals[feat] = val

        # Fill remaining features with median
        for feat in report.feature_cols:
            if feat not in input_vals and feat in df.columns:
                s = df[feat].dropna()
                if pd.api.types.is_numeric_dtype(s):
                    input_vals[feat] = float(s.median())
                else:
                    if len(s) > 0:
                        input_vals[feat] = s.mode()[0]

        st.divider()
        if st.button("Predict", type="primary"):
            result_pred = predict_what_if(report, input_vals)

            if "error" in result_pred:
                st.error("Prediction error: {}".format(result_pred["error"]))
            else:
                st.markdown("#### Prediction Result")
                if report.task == "regression":
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Predicted {}".format(report.target_col),
                        "{:.4f}".format(result_pred["prediction"])
                    )
                    if result_pred.get("lower") is not None:
                        c2.metric("Lower Bound",
                                  "{:.4f}".format(result_pred["lower"]))
                        c3.metric("Upper Bound",
                                  "{:.4f}".format(result_pred["upper"]))
                    st.caption(result_pred.get("confidence_note", ""))
                else:
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Predicted Class",
                        result_pred.get("prediction_label",
                                        str(result_pred["prediction"]))
                    )
                    if result_pred.get("confidence"):
                        c2.metric("Confidence",
                                  "{:.1f}%".format(result_pred["confidence"]))

                    if result_pred.get("probabilities"):
                        st.markdown("**Class Probabilities**")
                        prob_df = pd.DataFrame([
                            {"Class": k, "Probability": "{:.1f}%".format(v*100)}
                            for k, v in result_pred["probabilities"].items()
                        ])
                        st.dataframe(prob_df, use_container_width=True,
                                     hide_index=True)

# ── Tab 4: Insights ───────────────────────────────────────
with tab4:
    st.markdown("### ML Insights")
    st.caption("Auto-generated analysis of model results.")

    if report.insights:
        for insight in report.insights:
            if "WARNING" in insight or "Severe" in insight or "Weak" in insight:
                st.warning(insight)
            elif "Excellent" in insight or "Good" in insight:
                st.success(insight)
            else:
                st.info(insight)
    else:
        st.info("No insights generated.")

    # Training summary
    st.divider()
    st.markdown("#### Training Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Used",     "{:,}".format(report.n_rows_used))
    c2.metric("Features Used", str(report.n_features))
    c3.metric("Task Type",     report.task.title())
    c4.metric("Models Tested", str(len([m for m in report.models
                                        if m.cv_score != -999])))

    if report.n_rows_used < 1000:
        st.warning(
            "Small dataset ({:,} rows). Results are indicative — "
            "more data will improve reliability.".format(report.n_rows_used))

