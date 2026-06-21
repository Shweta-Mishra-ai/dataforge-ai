import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import logging
logger = logging.getLogger(__name__)


LIGHT_COLORS = ["#1565C0", "#0D47A1", "#B71C1C", "#1B5E20", "#4527A0", "#E65100"]
DARK_COLORS  = ["#64B5F6", "#4DB6AC", "#FFB74D", "#CE93D8", "#EF9A9A", "#FFF176"]
GREEN_COLORS = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#1A237E", "#0D47A1"]

# Must be defined at module level before any function references it
_SCORE_KEYWORDS = {"satisfaction", "rating", "score", "evaluation", "performance",
                   "sentiment", "nps", "csat", "quality", "health", "level", "index"}


def _get_style(theme_name: str) -> dict:
    if theme_name == "Dark Tech":
        return {
            "figure.facecolor": "#07080f",
            "axes.facecolor":   "#0e0f1a",
            "axes.edgecolor":   "#1e2035",
            "axes.labelcolor":  "#dde1f5",
            "xtick.color":      "#1e3a5f",
            "ytick.color":      "#1e3a5f",
            "text.color":       "#dde1f5",
            "grid.color":       "#1e2035",
            "grid.alpha":       0.8,
        }
    else:
        return {
            "figure.facecolor": "#ffffff",
            "axes.facecolor":   "#F8FAFF",
            "axes.edgecolor":   "#CBD5E1",
            "axes.labelcolor":  "#0F172A",   # near-black for labels
            "xtick.color":      "#0F172A",   # dark tick labels
            "ytick.color":      "#0F172A",   # dark tick labels
            "text.color":       "#0A1628",   # very dark for titles
            "grid.color":       "#CBD5E1",
            "grid.alpha":       0.5,
        }


def _get_colors(theme_name: str) -> list:
    if theme_name == "Dark Tech":
        return DARK_COLORS
    elif theme_name == "Executive Green":
        return GREEN_COLORS
    return LIGHT_COLORS


def _apply_style(ax, style: dict):
    ax.set_facecolor(style["axes.facecolor"])
    ax.tick_params(colors=style["xtick.color"])
    ax.grid(True, color=style["grid.color"],
            alpha=style["grid.alpha"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style["axes.edgecolor"])
    ax.spines["bottom"].set_color(style["axes.edgecolor"])


def fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def make_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    top_n: int = 15,
) -> bytes:
    """Smart bar: mean for score/rating metrics, sum for revenue/count metrics.
    Fixes the bug where satisfaction_level (0-1 range) displayed as 2,544 (count)."""
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    _SCORE_KW = {"satisfaction","rating","score","evaluation","performance",
                 "sentiment","nps","csat","quality","health","level","index"}
    is_score = any(kw in y_col.lower() for kw in _SCORE_KW)
    agg_func = "mean" if is_score else "sum"
    fmt      = "{:.3f}" if is_score else "{:,.0f}"
    y_label  = ("Avg " if is_score else "Total ") + y_col.replace("_", " ").title()

    agg = (df.groupby(x_col)[y_col]
             .agg(agg_func)
             .reset_index()
             .sort_values(y_col, ascending=False)
             .head(top_n))

    org_avg = float(agg[y_col].mean())
    bar_colors = [colors[0] if v >= org_avg else "#64748B"  # slate-500 — below avg
                  for v in agg[y_col]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bars = ax.bar(
        range(len(agg)), agg[y_col],
        color=bar_colors, alpha=0.88,
        edgecolor=style["axes.edgecolor"], linewidth=0.5
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.012,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=8, color=style["text.color"], fontweight="bold"
        )

    ax.axhline(org_avg, color=(colors[1] if len(colors) > 1 else "#888888"),
               linestyle="--", linewidth=1.2, alpha=0.7,
               label="Avg: {}".format(fmt.format(org_avg)))
    ax.legend(fontsize=8, framealpha=0)

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(
        [str(v)[:14] for v in agg[x_col]],
        rotation=35, ha="right", fontsize=8.5
    )
    ax.set_ylabel(y_label, fontsize=9, color=style["axes.labelcolor"])
    ax.set_title(title or "{} by {}".format(y_label, x_col.replace("_", " ").title()),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=12)
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    """
    Smart line chart:
    - If x is datetime: aggregate by month
    - If x is numeric with >50 unique values: bin into 20 buckets
    - If x is categorical: bar chart style instead
    - Otherwise: plot directly
    """
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    data = df[[x_col, y_col]].dropna().copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    x_is_datetime = pd.api.types.is_datetime64_any_dtype(data[x_col])
    x_is_numeric  = pd.api.types.is_numeric_dtype(data[x_col])
    n_unique      = data[x_col].nunique()

    if x_is_datetime:
        # Aggregate by month
        data[x_col] = pd.to_datetime(data[x_col])
        data = data.set_index(x_col).resample("ME")[y_col].mean().reset_index()
        x_vals = range(len(data))
        y_vals = data[y_col].values
        labels = [str(d)[:7] for d in data[x_col]]

    elif x_is_numeric and n_unique > 50:
        # Bin into 20 buckets for clean trend
        data["_bin"] = pd.cut(data[x_col], bins=20)
        agg = data.groupby("_bin")[y_col].mean().reset_index()
        agg = agg.dropna()
        x_vals = range(len(agg))
        y_vals = agg[y_col].values
        labels = [str(b)[:10] for b in agg["_bin"]]

    elif x_is_numeric and n_unique <= 50:
        # Sort and plot directly
        data = data.sort_values(x_col)
        x_vals = range(len(data))
        y_vals = data[y_col].values
        labels = [str(v) for v in data[x_col]]

    else:
        # Categorical x — aggregate by mean
        agg = data.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(15)
        x_vals = range(len(agg))
        y_vals = agg[y_col].values
        labels = [str(v)[:12] for v in agg[x_col]]

    ax.plot(
        x_vals, y_vals,
        color=colors[0], linewidth=2.5,
        marker="o", markersize=4,
        markerfacecolor=colors[1],
        markeredgecolor=colors[0]
    )
    ax.fill_between(x_vals, y_vals, alpha=0.12, color=colors[0])

    # X axis labels — max 10
    step = max(1, len(labels) // 10)
    ax.set_xticks(list(x_vals)[::step])
    ax.set_xticklabels(labels[::step], rotation=35, ha="right", fontsize=8)

    ax.set_ylabel(y_col, fontsize=9, color=style["axes.labelcolor"])
    ax.set_title(title or "{} Trend".format(y_col),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_histogram(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    bins: int = 25,
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

    mean_val   = data.mean()
    median_val = data.median()

    ax.axvline(mean_val, color=colors[2],
               linestyle="--", linewidth=1.8,
               label="Mean: {:.2f}".format(mean_val))
    ax.axvline(median_val, color=colors[3] if len(colors) > 3 else colors[1],
               linestyle=":", linewidth=1.8,
               label="Median: {:.2f}".format(median_val))
    ax.legend(fontsize=8)

    ax.set_xlabel(col, fontsize=9, color=style["axes.labelcolor"])
    ax.set_ylabel("Frequency", fontsize=9, color=style["axes.labelcolor"])
    ax.set_title(title or "Distribution: {}".format(col),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_ranked_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    """Horizontal ranked bar — clearer than a pie chart for comparing
    many roughly-equal categories on a score metric."""
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    is_score = any(kw in y_col.lower() for kw in _SCORE_KEYWORDS)
    agg_func = "mean" if is_score else "sum"
    fmt      = "{:.3f}" if is_score else "{:,.0f}"

    agg = (df.groupby(x_col)[y_col]
             .agg(agg_func)
             .reset_index()
             .sort_values(y_col, ascending=True)
             .head(15))

    org_avg = float(agg[y_col].mean())
    bar_colors = [colors[0] if v >= org_avg else "#64748B"
                  for v in agg[y_col]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(agg) * 0.45)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bars = ax.barh(
        range(len(agg)), agg[y_col],
        color=bar_colors, alpha=0.88,
        edgecolor=style["axes.edgecolor"], linewidth=0.5
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va="center", ha="left", fontsize=8,
                color=style["text.color"], fontweight="bold")

    ax.axvline(org_avg, color=(colors[3] if len(colors) > 3 else colors[1]),
               linestyle="--", linewidth=1.2, alpha=0.7,
               label="Avg: {}".format(fmt.format(org_avg)))
    ax.legend(fontsize=8, framealpha=0)

    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([str(v)[:16] for v in agg[x_col]], fontsize=9)
    ax.set_xlabel(("Avg " if is_score else "Total ") + y_col.replace("_", " ").title(),
                  fontsize=9, color=style["axes.labelcolor"])
    ax.set_title(title or "{} Ranking by {}".format(
                     y_col.replace("_", " ").title(), x_col.replace("_", " ").title()),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=12)
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


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

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(style["figure.facecolor"])

    wedges, texts, autotexts = ax.pie(
        agg[values_col],
        labels=None,
        autopct="%1.1f%%",
        colors=colors[:len(agg)],
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": style["figure.facecolor"], "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(
        wedges,
        [str(v)[:15] for v in agg[names_col]],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0,
        labelcolor=style["text.color"]
    )
    ax.set_title(title or "{} by {}".format(values_col, names_col),
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

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, "{:.2f}".format(val),
                    ha="center", va="center", fontsize=8,
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
        data, vert=True, patch_artist=True,
        notch=False, widths=0.4,
        flierprops=dict(marker="o", markerfacecolor=colors[4] if len(colors) > 4 else colors[0],
                        markersize=4, alpha=0.5)
    )
    bp["boxes"][0].set_facecolor(colors[0])
    bp["boxes"][0].set_alpha(0.7)
    bp["medians"][0].set_color(colors[1])
    bp["medians"][0].set_linewidth(2)

    ax.set_xticklabels([col], fontsize=9, color=style["xtick.color"])
    ax.set_title(title or "Outlier Analysis: {}".format(col),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)



def _pick_best_metric(num_cols, cat_cols=None):
    """Prefer score/rating columns for headline charts over raw counts/IDs."""
    for c in num_cols:
        if any(kw in c.lower() for kw in _SCORE_KEYWORDS):
            return c
    return num_cols[0] if num_cols else None


def generate_all_charts(
    df: pd.DataFrame,
    theme_name: str = "Corporate Light",
    max_charts: int = 5,
) -> List[Tuple[str, bytes]]:
    """Auto-generate best charts for this dataset.

    FIX: previously always used num_cols[0] for every chart and a numeric-bin
    line chart with unreadable x-axis labels when no datetime column existed.
    Now picks the best score metric and falls back to a second categorical
    breakdown or ranked bar chart instead of meaningless numeric bins.
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "string"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    charts    = []

    best_metric = _pick_best_metric(num_cols)
    best_cat = next(
        (c for c in cat_cols if 2 <= df[c].nunique() <= 25), cat_cols[0]
    ) if cat_cols else None

    # 1. Bar chart — best metric by best category (mean/sum auto-detected)
    if best_cat and best_metric:
        title = "{} by {}".format(best_metric.replace("_", " ").title(), best_cat)
        try:
            charts.append((title, make_bar_chart(
                df, best_cat, best_metric, title, theme_name
            )))
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # 2. Second view — datetime trend OR second categorical breakdown
    #    (avoids numeric-binned x-axis with unreadable labels like "(0.089, 0.)")
    if date_cols and best_metric:
        title = "{} Over Time".format(best_metric.replace("_", " ").title())
        try:
            charts.append((title, make_line_chart(
                df, date_cols[0], best_metric, title, theme_name
            )))
        except Exception:
            logger.debug("%s silent skip", exc_info=True)
    else:
        second_cat = next(
            (c for c in cat_cols if c != best_cat and 2 <= df[c].nunique() <= 12),
            None
        )
        if second_cat and best_metric:
            title = "{} by {}".format(best_metric.replace("_", " ").title(), second_cat)
            try:
                charts.append((title, make_bar_chart(
                    df, second_cat, best_metric, title, theme_name
                )))
            except Exception:
                logger.debug("%s silent skip", exc_info=True)
        elif len(num_cols) >= 2:
            second_metric = next((c for c in num_cols if c != best_metric), num_cols[0])
            title = "Distribution: {}".format(second_metric.replace("_", " ").title())
            try:
                charts.append((title, make_histogram(
                    df, second_metric, title, theme_name
                )))
            except Exception:
                logger.debug("%s silent skip", exc_info=True)

    # 3. Histogram — distribution of primary metric
    if best_metric:
        title = "Distribution: {}".format(best_metric.replace("_", " ").title())
        try:
            charts.append((title, make_histogram(
                df, best_metric, title, theme_name
            )))
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # 4. Correlation heatmap
    if len(num_cols) >= 3:
        try:
            charts.append(("Correlation Matrix", make_correlation_heatmap(
                df, "Correlation Matrix", theme_name
            )))
        except Exception:
            logger.debug("%s silent skip", exc_info=True)

    # 5. Ranked horizontal bar — clearer comparison than a pie for many categories
    if best_cat and best_metric:
        title = "{} Ranking by {}".format(best_metric.replace("_", " ").title(), best_cat)
        try:
            charts.append((title, make_ranked_bar_chart(
                df, best_cat, best_metric, title, theme_name
            )))
        except Exception:
            # Fall back to pie only if ranked bar genuinely fails
            try:
                charts.append((title, make_pie_chart(
                    df, best_cat, best_metric, title, theme_name
                )))
            except Exception:
                logger.debug("%s silent skip", exc_info=True)

    return charts[:max_charts]
