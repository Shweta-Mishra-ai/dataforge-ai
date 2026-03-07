import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — Streamlit Cloud pe zaroori
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Optional


# ── Default color palettes ─────────────────────────────────
LIGHT_COLORS = ["#1a4a8a","#2196F3","#42A5F5","#90CAF9","#0D47A1","#1565C0"]
DARK_COLORS  = ["#4f8ef7","#22d3a5","#f7934f","#a78bfa","#f77070","#ffd43b"]
GREEN_COLORS = ["#1a6b4a","#2ecc71","#27ae60","#82e0aa","#145a32","#1e8449"]


def _get_style(theme_name: str) -> dict:
    """Return matplotlib rcParams style dict based on theme."""
    if theme_name == "Dark Tech":
        return {
            "figure.facecolor":  "#07080f",
            "axes.facecolor":    "#0e0f1a",
            "axes.edgecolor":    "#1e2035",
            "axes.labelcolor":   "#dde1f5",
            "xtick.color":       "#636a8a",
            "ytick.color":       "#636a8a",
            "text.color":        "#dde1f5",
            "grid.color":        "#1e2035",
            "grid.alpha":        0.8,
        }
    else:
        return {
            "figure.facecolor":  "#ffffff",
            "axes.facecolor":    "#f8faff",
            "axes.edgecolor":    "#d0d8f0",
            "axes.labelcolor":   "#1e1e28",
            "xtick.color":       "#646882",
            "ytick.color":       "#646882",
            "text.color":        "#1e1e28",
            "grid.color":        "#e0e8f5",
            "grid.alpha":        0.8,
        }


def _get_colors(theme_name: str) -> list:
    if theme_name == "Dark Tech":
        return DARK_COLORS
    elif theme_name == "Executive Green":
        return GREEN_COLORS
    return LIGHT_COLORS


def _apply_style(ax, style: dict):
    """Apply style dict to axes."""
    ax.set_facecolor(style["axes.facecolor"])
    ax.tick_params(colors=style["xtick.color"])
    ax.grid(True, color=style["grid.color"],
            alpha=style["grid.alpha"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style["axes.edgecolor"])
    ax.spines["bottom"].set_color(style["axes.edgecolor"])


def fig_to_bytes(
    fig: plt.Figure,
    dpi: int = 150
) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


# ── Chart makers ──────────────────────────────────────────

def make_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    top_n: int = 15,
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    agg = (df.groupby(x_col)[y_col]
             .sum()
             .reset_index()
             .sort_values(y_col, ascending=False)
             .head(top_n))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bars = ax.bar(
        range(len(agg)), agg[y_col],
        color=colors[0], alpha=0.85,
        edgecolor=style["axes.edgecolor"], linewidth=0.5
    )

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            f"{h:,.0f}",
            ha="center", va="bottom",
            fontsize=7, color=style["text.color"]
        )

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(
        [str(v)[:12] for v in agg[x_col]],
        rotation=35, ha="right", fontsize=8
    )
    ax.set_ylabel(y_col, fontsize=9,
                  color=style["axes.labelcolor"])
    ax.set_title(title or f"{y_col} by {x_col}",
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    data = df[[x_col, y_col]].dropna().sort_values(x_col)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    ax.plot(
        range(len(data)), data[y_col],
        color=colors[0], linewidth=2,
        marker="o", markersize=3, markerfacecolor=colors[1]
    )
    ax.fill_between(
        range(len(data)), data[y_col],
        alpha=0.15, color=colors[0]
    )

    step = max(1, len(data) // 8)
    ax.set_xticks(range(0, len(data), step))
    ax.set_xticklabels(
        [str(data[x_col].iloc[i])[:10]
         for i in range(0, len(data), step)],
        rotation=35, ha="right", fontsize=8
    )
    ax.set_ylabel(y_col, fontsize=9,
                  color=style["axes.labelcolor"])
    ax.set_title(title or f"{y_col} over {x_col}",
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_histogram(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    bins: int = 30,
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)
    data   = df[col].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    ax.hist(data, bins=bins, color=colors[0],
            alpha=0.8, edgecolor=style["axes.edgecolor"],
            linewidth=0.4)

    # Mean line
    mean_val = data.mean()
    ax.axvline(mean_val, color=colors[2],
               linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:,.2f}")
    ax.legend(fontsize=8)

    ax.set_xlabel(col, fontsize=9,
                  color=style["axes.labelcolor"])
    ax.set_ylabel("Frequency", fontsize=9,
                  color=style["axes.labelcolor"])
    ax.set_title(title or f"Distribution: {col}",
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    agg = (df.groupby(names_col)[values_col]
             .sum()
             .reset_index()
             .sort_values(values_col, ascending=False)
             .head(8))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(style["figure.facecolor"])

    wedges, texts, autotexts = ax.pie(
        agg[values_col],
        labels=None,
        autopct="%1.1f%%",
        colors=colors[:len(agg)],
        startangle=90,
        pctdistance=0.82,
        wedgeprops={"edgecolor": style["figure.facecolor"],
                    "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(style["figure.facecolor"])

    # Legend
    ax.legend(
        wedges,
        [str(v)[:15] for v in agg[names_col]],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        framealpha=0,
        labelcolor=style["text.color"]
    )
    ax.set_title(title or f"{values_col} by {names_col}",
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    theme_name: str = "Corporate Light",
) -> bytes:
    style    = _get_style(theme_name)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        return None

    corr = df[num_cols[:10]].corr().round(2)
    n    = len(corr)

    fig, ax = plt.subplots(figsize=(max(7, n), max(5, n - 1)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    ax.set_facecolor(style["axes.facecolor"])

    cmap = "RdBu_r"
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1,
                     aspect="auto")

    # Annotations
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(val) > 0.5 else style["text.color"])

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c[:10] for c in corr.columns],
                       rotation=45, ha="right", fontsize=8,
                       color=style["xtick.color"])
    ax.set_yticklabels([c[:10] for c in corr.index],
                       fontsize=8, color=style["ytick.color"])
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_box_plot(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)
    data   = df[col].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        notch=False,
        widths=0.4,
        flierprops=dict(
            marker="o", markerfacecolor=colors[4],
            markersize=4, alpha=0.5
        )
    )
    bp["boxes"][0].set_facecolor(colors[0])
    bp["boxes"][0].set_alpha(0.7)
    bp["medians"][0].set_color(colors[1])
    bp["medians"][0].set_linewidth(2)

    ax.set_xticklabels([col], fontsize=9,
                       color=style["xtick.color"])
    ax.set_title(title or f"Outlier Analysis: {col}",
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def generate_all_charts(
    df: pd.DataFrame,
    theme_name: str = "Corporate Light",
    max_charts: int = 5,
) -> List[Tuple[str, bytes]]:
    """
    Auto-generate best charts for this dataset.
    Returns list of (title, png_bytes) tuples.
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    charts    = []

    # 1. Bar chart
    if cat_cols and num_cols:
        best_cat = next(
            (c for c in cat_cols if 2 <= df[c].nunique() <= 25),
            cat_cols[0]
        )
        title = f"{num_cols[0]} by {best_cat}"
        try:
            charts.append((
                title,
                make_bar_chart(df, best_cat, num_cols[0],
                               title, theme_name)
            ))
        except Exception:
            pass

    # 2. Line chart
    if date_cols and num_cols:
        title = f"{num_cols[0]} Over Time"
        try:
            charts.append((
                title,
                make_line_chart(df, date_cols[0], num_cols[0],
                                title, theme_name)
            ))
        except Exception:
            pass
    elif len(num_cols) >= 2:
        title = f"{num_cols[1]} Trend"
        try:
            charts.append((
                title,
                make_line_chart(df, num_cols[0], num_cols[1],
                                title, theme_name)
            ))
        except Exception:
            pass

    # 3. Histogram
    if num_cols:
        title = f"Distribution: {num_cols[0]}"
        try:
            charts.append((
                title,
                make_histogram(df, num_cols[0], title, theme_name)
            ))
        except Exception:
            pass

    # 4. Correlation heatmap
    if len(num_cols) >= 3:
        try:
            charts.append((
                "Correlation Matrix",
                make_correlation_heatmap(df, "Correlation Matrix",
                                         theme_name)
            ))
        except Exception:
            pass

    # 5. Pie chart
    if cat_cols and num_cols:
        best_cat = next(
            (c for c in cat_cols if 2 <= df[c].nunique() <= 10),
            None
        )
        if best_cat:
            title = f"{num_cols[0]} Share by {best_cat}"
            try:
                charts.append((
                    title,
                    make_pie_chart(df, best_cat, num_cols[0],
                                   title, theme_name)
                ))
            except Exception:
                pass

    return charts[:max_charts]
