"""
core/chart_theme.py — DataForge AI
Professional board-level chart styling.
Apply to every Plotly chart for consistent, premium output.
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

# ── Palette ────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1B4FD8",
    "secondary": "#0EA5E9",
    "accent":    "#6366F1",
    "success":   "#059669",
    "warning":   "#D97706",
    "danger":    "#DC2626",
    "neutral":   "#64748B",
    "light":     "#F1F5F9",
    "dark":      "#0F172A",
}

CATEGORICAL = [
    "#1B4FD8", "#059669", "#D97706", "#DC2626",
    "#6366F1", "#0EA5E9", "#7C3AED", "#DB2777",
]

RAG = {
    "critical": "#DC2626",
    "high":     "#D97706",
    "ok":       "#059669",
    "info":     "#1B4FD8",
}


def base_layout(
    title: str = "",
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    height: int = 420,
    show_legend: bool = True,
    bg: str = "rgba(0,0,0,0)",   # transparent — inherits Streamlit theme bg
) -> dict:
    """Return a consistent Plotly layout dict for board-level charts."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;opacity:.6'>{subtitle}</span>"

    return dict(
        title=dict(
            text=title_text,
            font=dict(family="Inter, Arial, sans-serif", size=16, color="#0F172A"),
            x=0.0, xanchor="left", pad=dict(l=4),
        ),
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#374151"),
        showlegend=show_legend,
        legend=dict(
            font=dict(size=11, color="#374151"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0", borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True, gridcolor="rgba(128,128,128,.15)", gridwidth=1,
            showline=True, linecolor="#E2E8F0", linewidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True, gridcolor="rgba(128,128,128,.15)", gridwidth=1,
            showline=False, zeroline=False,
        ),
    )


def apply_theme(fig: go.Figure, title: str = "", subtitle: str = "",
                x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply professional theme to any Plotly figure."""
    fig.update_layout(**base_layout(title, subtitle, x_title, y_title))
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
        selector=dict(type="bar")
    )
    return fig


def bar_chart(df, x_col: str, y_col: str, title: str,
              color_col: Optional[str] = None, horizontal: bool = False,
              reference_line: Optional[float] = None,
              reference_label: str = "Reference",
              color_map: Optional[dict] = None) -> go.Figure:
    """Professional bar chart with optional reference line."""
    import pandas as pd

    if horizontal:
        fig = px.bar(df, x=y_col, y=x_col, orientation="h",
                     color=color_col,
                     color_discrete_map=color_map or {},
                     color_discrete_sequence=CATEGORICAL,
                     text=y_col)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          selector=dict(type="bar"))
    else:
        fig = px.bar(df, x=x_col, y=y_col,
                     color=color_col,
                     color_discrete_map=color_map or {},
                     color_discrete_sequence=CATEGORICAL,
                     text=y_col)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                          selector=dict(type="bar"))

    if reference_line is not None:
        if horizontal:
            fig.add_vline(x=reference_line, line_dash="dash",
                          line_color=PALETTE["neutral"], line_width=1.5,
                          annotation_text=reference_label,
                          annotation_font_size=10)
        else:
            fig.add_hline(y=reference_line, line_dash="dash",
                          line_color=PALETTE["neutral"], line_width=1.5,
                          annotation_text=reference_label,
                          annotation_font_size=10)

    fig = apply_theme(fig, title)
    fig.update_layout(bargap=0.3)
    return fig


def scatter_chart(df, x_col: str, y_col: str, title: str,
                  color_col: Optional[str] = None,
                  size_col: Optional[str] = None,
                  trendline: bool = True) -> go.Figure:
    """Professional scatter with optional trendline."""
    kwargs = dict(color=color_col, size=size_col,
                  color_discrete_sequence=CATEGORICAL)
    if trendline:
        kwargs["trendline"] = "ols"
    fig = px.scatter(df, x=x_col, y=y_col, **kwargs)
    return apply_theme(fig, title, x_title=x_col, y_title=y_col)


def heatmap(pivot_df, title: str, fmt: str = ".1f",
            colorscale: str = "RdYlGn") -> go.Figure:
    """Clean correlation / pivot heatmap."""
    import numpy as np
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(c) for c in pivot_df.columns],
        y=[str(i) for i in pivot_df.index],
        colorscale=colorscale,
        text=np.round(pivot_df.values, 2),
        texttemplate=f"%{{text:{fmt}}}",
        textfont=dict(size=11),
        showscale=True,
    ))
    return apply_theme(fig, title)


def line_chart(df, x_col: str, y_cols: list, title: str,
               reference_line: Optional[float] = None,
               reference_label: str = "") -> go.Figure:
    """Multi-line chart with optional reference line."""
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], name=col, mode="lines+markers",
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2.5),
            marker=dict(size=6),
        ))
    if reference_line is not None:
        fig.add_hline(y=reference_line, line_dash="dot",
                      line_color=PALETTE["neutral"], line_width=1.5,
                      annotation_text=reference_label, annotation_font_size=10)
    return apply_theme(fig, title)


def rag_color(value: float, critical: float, high: float, ok: float,
              higher_is_worse: bool = True) -> str:
    """Return RAG color string for a value."""
    if higher_is_worse:
        if value >= critical:   return RAG["critical"]
        elif value >= high:     return RAG["high"]
        else:                   return RAG["ok"]
    else:
        if value <= critical:   return RAG["critical"]
        elif value <= high:     return RAG["high"]
        else:                   return RAG["ok"]

