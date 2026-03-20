"""
pdf_builder.py — Client-grade PDF report builder.
Integrates: stats, cleaning report, BI insights, ML results, EDA findings.
Narrative-driven — not just data dump.
"""
import io
from datetime import datetime
import pandas as pd
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether,
    BaseDocTemplate, Frame, PageTemplate
)


# ══════════════════════════════════════════════════════════
#  COLOR HELPERS
# ══════════════════════════════════════════════════════════

def _rgb(hex_or_tuple):
    if isinstance(hex_or_tuple, tuple):
        return colors.Color(
            hex_or_tuple[0]/255,
            hex_or_tuple[1]/255,
            hex_or_tuple[2]/255
        )
    h = hex_or_tuple.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return colors.Color(r/255, g/255, b/255)


# ── Themes ────────────────────────────────────────────────
THEMES = {
    "Corporate Light": {
        "header_bg":    "#1a4a8a",
        "header_text":  "#ffffff",
        "accent":       "#2196F3",
        "accent2":      "#22d3a5",
        "text":         "#1e1e28",
        "text_muted":   "#646882",
        "bg_light":     "#f0f4ff",
        "bg_card":      "#f8faff",
        "border":       "#d0d8f0",
        "positive":     "#22d3a5",
        "negative":     "#f77070",
        "warning":      "#f7934f",
    },
    "Dark Tech": {
        "header_bg":    "#07080f",
        "header_text":  "#4f8ef7",
        "accent":       "#4f8ef7",
        "accent2":      "#22d3a5",
        "text":         "#dde1f5",
        "text_muted":   "#636a8a",
        "bg_light":     "#0e0f1a",
        "bg_card":      "#12132a",
        "border":       "#1e2035",
        "positive":     "#22d3a5",
        "negative":     "#f77070",
        "warning":      "#f7934f",
    },
    "Executive Green": {
        "header_bg":    "#1a6b4a",
        "header_text":  "#ffffff",
        "accent":       "#2ecc71",
        "accent2":      "#27ae60",
        "text":         "#1a2e1a",
        "text_muted":   "#5a7a5a",
        "bg_light":     "#f0fff4",
        "bg_card":      "#f8fff8",
        "border":       "#c8e6c9",
        "positive":     "#27ae60",
        "negative":     "#e74c3c",
        "warning":      "#f39c12",
    },
}


# ══════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════

def _styles(theme: dict, CW: float) -> dict:
    T   = theme
    txt = _rgb(T["text"])
    mut = _rgb(T["text_muted"])
    acc = _rgb(T["accent"])

    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold", fontSize=28,
            textColor=_rgb(T["header_text"]),
            alignment=TA_CENTER, spaceAfter=6*mm,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="Helvetica", fontSize=13,
            textColor=_rgb(T["header_text"]),
            alignment=TA_CENTER, spaceAfter=4*mm,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            fontName="Helvetica", fontSize=10,
            textColor=_rgb(T["header_text"]),
            alignment=TA_CENTER, spaceAfter=3*mm,
        ),
        "section_h": ParagraphStyle(
            "section_h",
            fontName="Helvetica-Bold", fontSize=16,
            textColor=acc, spaceAfter=2*mm, spaceBefore=4*mm,
        ),
        "sub_h": ParagraphStyle(
            "sub_h",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=txt, spaceAfter=2*mm, spaceBefore=3*mm,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica", fontSize=9,
            textColor=txt, leading=14,
            spaceAfter=2*mm, alignment=TA_JUSTIFY,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=txt, leading=14, spaceAfter=1*mm,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica", fontSize=8,
            textColor=mut, leading=11, spaceAfter=1*mm,
        ),
        "finding": ParagraphStyle(
            "finding",
            fontName="Helvetica", fontSize=9,
            textColor=txt, leading=14,
            leftIndent=4*mm, spaceAfter=2*mm,
        ),
        "toc": ParagraphStyle(
            "toc",
            fontName="Helvetica", fontSize=10,
            textColor=txt, leading=16, spaceAfter=1*mm,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label",
            fontName="Helvetica", fontSize=7,
            textColor=mut, alignment=TA_CENTER,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value",
            fontName="Helvetica-Bold", fontSize=20,
            textColor=acc, alignment=TA_CENTER,
        ),
        "kpi_sub": ParagraphStyle(
            "kpi_sub",
            fontName="Helvetica", fontSize=7,
            textColor=mut, alignment=TA_CENTER,
        ),
        "risk": ParagraphStyle(
            "risk",
            fontName="Helvetica", fontSize=9,
            textColor=_rgb(T["negative"]),
            leading=13, leftIndent=4*mm, spaceAfter=1*mm,
        ),
        "opportunity": ParagraphStyle(
            "opportunity",
            fontName="Helvetica", fontSize=9,
            textColor=_rgb(T["positive"]),
            leading=13, leftIndent=4*mm, spaceAfter=1*mm,
        ),
        "action": ParagraphStyle(
            "action",
            fontName="Helvetica", fontSize=9,
            textColor=txt, leading=13,
            leftIndent=6*mm, spaceAfter=2*mm,
        ),
        "narrative": ParagraphStyle(
            "narrative",
            fontName="Helvetica", fontSize=9,
            textColor=txt, leading=14,
            spaceAfter=2*mm, alignment=TA_JUSTIFY,
            leftIndent=3*mm, rightIndent=3*mm,
        ),
        "insight": ParagraphStyle(
            "insight",
            fontName="Helvetica", fontSize=9,
            textColor=txt, leading=13,
            leftIndent=4*mm, spaceAfter=2*mm,
        ),
    }


# ══════════════════════════════════════════════════════════
#  LAYOUT HELPERS
# ══════════════════════════════════════════════════════════

def _page_h(story, ST, theme, title):
    """Section page header with underline."""
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(title, ST["section_h"]))
    story.append(HRFlowable(
        width="100%", thickness=1.5,
        color=_rgb(theme["accent"]),
        spaceAfter=3*mm, spaceBefore=1*mm,
    ))


def _divider(story, theme, light=False):
    story.append(HRFlowable(
        width="100%", thickness=0.5 if light else 1,
        color=_rgb(theme["border"]),
        spaceAfter=2*mm, spaceBefore=2*mm,
    ))


def _table(story, theme, data, col_widths=None, header=True):
    """Styled table with alternating rows."""
    if not data or len(data) < 1:
        return
    T = theme
    style = [
        ("FONTNAME",  (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",  (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 8),
        ("TEXTCOLOR", (0,0), (-1,0), _rgb(T["header_text"])),
        ("BACKGROUND",(0,0), (-1,0), _rgb(T["header_bg"])),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [_rgb(T["bg_card"]), _rgb(T["bg_light"])]),
        ("GRID",      (0,0), (-1,-1), 0.3, _rgb(T["border"])),
        ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("WORDWRAP",  (0,0), (-1,-1), True),
    ]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 2*mm))


def _kpi_row(story, ST, theme, kpis, CW):
    """Row of KPI cards: [{label, value, sub}]."""
    n   = len(kpis)
    cw  = CW / n
    T   = theme
    row = []
    for kpi in kpis:
        cell = [
            Paragraph(kpi.get("label",""), ST["kpi_label"]),
            Spacer(1, 1*mm),
            Paragraph(str(kpi.get("value","")), ST["kpi_value"]),
            Spacer(1, 1*mm),
            Paragraph(str(kpi.get("sub","")), ST["kpi_sub"]),
        ]
        row.append(cell)

    tbl_style = [
        ("BACKGROUND",  (0,0), (-1,-1), _rgb(T["bg_card"])),
        ("BOX",         (0,0), (-1,-1), 0.5, _rgb(T["border"])),
        ("LINEAFTER",   (0,0), (-2,-1), 0.5, _rgb(T["border"])),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]
    t = Table([row], colWidths=[cw]*n)
    t.setStyle(TableStyle(tbl_style))
    story.append(t)
    story.append(Spacer(1, 3*mm))


def _narrative_box(story, ST, theme, text):
    """Highlighted narrative text box."""
    T = theme
    box_data = [[Paragraph(text, ST["narrative"])]]
    box = Table(box_data, colWidths=["100%"])
    box.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), _rgb(T["bg_light"])),
        ("LINEAFTER",    (0,0), (0,-1),  3, _rgb(T["accent"])),
        ("LINEBEFORE",   (0,0), (0,-1),  3, _rgb(T["accent"])),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(box)
    story.append(Spacer(1, 2*mm))


# ══════════════════════════════════════════════════════════
#  PAGE BUILDERS
# ══════════════════════════════════════════════════════════

def _cover(story, ST, theme, config, CW):
    """Full-page cover."""
    T   = theme
    now = datetime.now().strftime("%B %d, %Y")

    # Cover background block
    cover_data = [[
        Paragraph(config.get("title","Data Analysis Report"), ST["cover_title"]),
        Spacer(1, 3*mm),
        Paragraph(config.get("subtitle",""), ST["cover_sub"]),
        Spacer(1, 8*mm),
        Paragraph("Prepared for: {}".format(
            config.get("client_name","Client")), ST["cover_meta"]),
        Paragraph("Prepared by: DataForge AI", ST["cover_meta"]),
        Paragraph("Date: {}".format(now), ST["cover_meta"]),
        Spacer(1, 4*mm),
        Paragraph("Theme: {}".format(
            config.get("theme_name","Corporate Light")), ST["cover_meta"]),
    ]]
    if config.get("confidential", True):
        cover_data[0].append(Spacer(1, 6*mm))
        cover_data[0].append(Paragraph("CONFIDENTIAL DOCUMENT", ParagraphStyle(
            "conf", fontName="Helvetica-Bold", fontSize=10,
            textColor=_rgb(T["negative"]), alignment=TA_CENTER,
        )))
        cover_data[0].append(Paragraph(
            "This report was automatically generated by DataForge AI.",
            ST["cover_meta"]))

    tbl = Table([cover_data], colWidths=[CW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), _rgb(T["header_bg"])),
        ("TOPPADDING", (0,0), (-1,-1), 20*mm),
        ("BOTTOMPADDING",(0,0), (-1,-1), 20*mm),
        ("LEFTPADDING", (0,0), (-1,-1), 10*mm),
        ("RIGHTPADDING",(0,0), (-1,-1), 10*mm),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl)


def _toc(story, ST, theme, sections, CW):
    """Table of contents."""
    _page_h(story, ST, theme, "Table of Contents")
    for i, (num, title, page) in enumerate(sections):
        row = Paragraph(
            "{}. {}{}{}".format(
                num, title,
                " " * max(1, 60 - len(title)),
                page
            ),
            ST["toc"]
        )
        story.append(row)
    story.append(Spacer(1, 4*mm))


def _exec_summary(story, ST, theme, summary_text, findings, risks,
                  opportunities, CW):
    """Executive summary with findings, risks, opportunities."""
    _page_h(story, ST, theme, "Executive Summary")

    if summary_text:
        _narrative_box(story, ST, theme, summary_text)

    if findings:
        story.append(Paragraph("Key Findings", ST["sub_h"]))
        for i, f in enumerate(findings[:6], 1):
            story.append(Paragraph(
                "+ {}".format(f), ST["finding"]))

    if risks:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Business Risks", ST["sub_h"]))
        for r in risks[:4]:
            story.append(Paragraph("! {}".format(r), ST["risk"]))

    if opportunities:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Opportunities", ST["sub_h"]))
        for o in opportunities[:4]:
            story.append(Paragraph("* {}".format(o), ST["opportunity"]))


def _dataset_overview(story, ST, theme, df, profile, cleaning_summary, CW):
    """Dataset overview with KPI cards + before/after cleaning."""
    _page_h(story, ST, theme, "Dataset Overview")

    # KPI cards — post-cleaning
    _kpi_row(story, ST, theme, [
        {"label": "TOTAL ROWS",    "value": "{:,}".format(len(df)),
         "sub": "records"},
        {"label": "COLUMNS",       "value": str(len(df.columns)),
         "sub": "features"},
        {"label": "MISSING",
         "value": "{:.1f}%".format(
             df.isna().sum().sum() / max(df.shape[0]*df.shape[1],1)*100),
         "sub": "{} cells".format(int(df.isna().sum().sum()))},
        {"label": "QUALITY SCORE", "value": "{}".format(
             getattr(profile, "overall_quality_score", "N/A")),
         "sub": "out of 100"},
    ], CW)

    # Cleaning summary
    if cleaning_summary:
        story.append(Paragraph("Cleaning Actions Applied", ST["sub_h"]))
        rows = [
            [Paragraph("Before", ST["body_bold"]),
             Paragraph("After", ST["body_bold"]),
             Paragraph("Action", ST["body_bold"])],
        ]
        dup = cleaning_summary.get("duplicates_removed", 0)
        rows.append([
            Paragraph("{:,} rows".format(
                cleaning_summary.get("original_rows", len(df))), ST["body"]),
            Paragraph("{:,} rows".format(
                cleaning_summary.get("cleaned_rows", len(df))), ST["body"]),
            Paragraph("{} duplicates removed".format(dup), ST["body"]),
        ])
        miss_actions = cleaning_summary.get("groups", {}).get("missing", [])
        if miss_actions:
            for a in miss_actions[:3]:
                rows.append([
                    Paragraph(a.column, ST["body"]),
                    Paragraph(a.action, ST["body"]),
                    Paragraph("Missing values handled", ST["body"]),
                ])
        _table(story, theme, rows, [CW*0.25, CW*0.35, CW*0.40])

    # Column type breakdown
    story.append(Paragraph("Column Type Breakdown", ST["sub_h"]))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

    rows = [
        [Paragraph("Type", ST["body_bold"]),
         Paragraph("Count", ST["body_bold"]),
         Paragraph("Columns (sample)", ST["body_bold"])],
        [Paragraph("Numeric", ST["body"]),
         Paragraph(str(len(num_cols)), ST["body"]),
         Paragraph(", ".join(num_cols[:5]), ST["body"])],
        [Paragraph("Categorical", ST["body"]),
         Paragraph(str(len(cat_cols)), ST["body"]),
         Paragraph(", ".join(cat_cols[:5]), ST["body"])],
        [Paragraph("DateTime", ST["body"]),
         Paragraph(str(len(dt_cols)), ST["body"]),
         Paragraph(", ".join(dt_cols[:5]) or "None", ST["body"])],
    ]
    _table(story, theme, rows, [CW*0.2, CW*0.15, CW*0.65])

    # Descriptive stats
    if num_cols:
        story.append(Paragraph("Descriptive Statistics", ST["sub_h"]))
        desc  = df[num_cols[:6]].describe().round(2)
        hdr   = [Paragraph("Stat", ST["body_bold"])] + [
            Paragraph(c[:12], ST["body_bold"]) for c in num_cols[:6]]
        rows  = [hdr]
        for stat in ["mean","std","min","25%","50%","75%","max"]:
            if stat in desc.index:
                row = [Paragraph(stat, ST["body"])] + [
                    Paragraph(str(desc.loc[stat, c]), ST["body"])
                    for c in num_cols[:6]]
                rows.append(row)
        cw_stat = CW / (len(num_cols[:6]) + 1)
        _table(story, theme, rows,
               [cw_stat] + [cw_stat]*len(num_cols[:6]))


def _statistical_analysis(story, ST, theme, stats_report, CW):
    """Statistical analysis section — normality, outliers, correlations."""
    if stats_report is None:
        return
    _page_h(story, ST, theme, "Statistical Analysis")

    # Normality summary
    story.append(Paragraph("Distribution Analysis", ST["sub_h"]))
    num_insights = []
    for col, cs in list(stats_report.column_stats.items())[:8]:
        if cs.mean is None:
            continue
        normal = "Normal" if cs.is_normal else "Non-normal"
        skew   = cs.skew_label or "N/A"
        num_insights.append(
            "• '{}': {} | {} | Outliers (IQR): {}".format(
                col, normal, skew, cs.outlier_count_iqr)
        )
    for ins in num_insights[:6]:
        story.append(Paragraph(ins, ST["body"]))

    # Significant correlations
    sig_corrs = [c for c in stats_report.correlations
                 if c.is_significant and abs(c.pearson_r) >= 0.3]
    if sig_corrs:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Significant Correlations", ST["sub_h"]))
        rows = [[
            Paragraph("Column A", ST["body_bold"]),
            Paragraph("Column B", ST["body_bold"]),
            Paragraph("Pearson r", ST["body_bold"]),
            Paragraph("p-value", ST["body_bold"]),
            Paragraph("Strength", ST["body_bold"]),
        ]]
        for c in sig_corrs[:8]:
            rows.append([
                Paragraph(c.col_a, ST["body"]),
                Paragraph(c.col_b, ST["body"]),
                Paragraph(str(c.pearson_r), ST["body"]),
                Paragraph(str(c.p_value), ST["body"]),
                Paragraph(c.strength.title(), ST["body"]),
            ])
        _table(story, theme, rows,
               [CW*0.22, CW*0.22, CW*0.15, CW*0.15, CW*0.26])

    # Insights
    if stats_report.dataset_insights:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Statistical Insights", ST["sub_h"]))
        for ins in stats_report.dataset_insights[:5]:
            story.append(Paragraph("• " + ins, ST["insight"]))


def _bi_section(story, ST, theme, bi_report, CW):
    """Business Intelligence section."""
    if bi_report is None:
        return
    _page_h(story, ST, theme, "Business Intelligence")

    # Executive brief
    if bi_report.executive_brief:
        _narrative_box(story, ST, theme, bi_report.executive_brief)

    # Benchmarks
    if bi_report.benchmarks:
        story.append(Paragraph("Benchmarking Summary", ST["sub_h"]))
        rows = [[
            Paragraph("Metric", ST["body_bold"]),
            Paragraph("Mean", ST["body_bold"]),
            Paragraph("Median", ST["body_bold"]),
            Paragraph("Top 10%", ST["body_bold"]),
            Paragraph("Bottom 10%", ST["body_bold"]),
            Paragraph("Variation", ST["body_bold"]),
        ]]
        for bm in bi_report.benchmarks[:4]:
            rows.append([
                Paragraph(bm.column, ST["body"]),
                Paragraph(str(bm.mean), ST["body"]),
                Paragraph(str(bm.median), ST["body"]),
                Paragraph(str(bm.top_10_pct), ST["body"]),
                Paragraph(str(bm.bottom_10_pct), ST["body"]),
                Paragraph(bm.benchmark_label.split("—")[0].strip(), ST["body"]),
            ])
        _table(story, theme, rows,
               [CW*0.20, CW*0.13, CW*0.13, CW*0.13, CW*0.13, CW*0.28])

    # Cohort findings
    sig_cohorts = [c for c in bi_report.cohorts if c.is_significant]
    if sig_cohorts:
        story.append(Paragraph("Cohort Analysis Findings", ST["sub_h"]))
        for c in sig_cohorts[:3]:
            story.append(Paragraph("• " + c.interpretation, ST["body"]))
            if c.recommendations:
                for r in c.recommendations[:1]:
                    story.append(Paragraph("  → " + r, ST["action"]))

    # Pareto
    if bi_report.pareto:
        story.append(Paragraph("Pareto Analysis", ST["sub_h"]))
        for p in bi_report.pareto[:2]:
            story.append(Paragraph("• " + p.interpretation, ST["body"]))

    # Key BI insights
    if bi_report.key_insights:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Key Business Insights", ST["sub_h"]))
        for ins in bi_report.key_insights[:5]:
            story.append(Paragraph("• " + ins, ST["insight"]))


def _ml_section(story, ST, theme, ml_report, CW):
    """ML Predictions section."""
    if ml_report is None:
        return
    _page_h(story, ST, theme, "ML Predictions")

    story.append(Paragraph(
        "Target: '{}' | Task: {} | Rows used: {:,} | Features: {}".format(
            ml_report.target_col, ml_report.task.title(),
            ml_report.n_rows_used, ml_report.n_features),
        ST["body"]
    ))
    story.append(Spacer(1, 2*mm))

    # Model comparison table
    valid = [m for m in ml_report.models if m.cv_score != -999]
    if valid:
        story.append(Paragraph("Model Comparison", ST["sub_h"]))
        rows = [[
            Paragraph("Model", ST["body_bold"]),
            Paragraph("CV Score", ST["body_bold"]),
            Paragraph("Test Score", ST["body_bold"]),
            Paragraph("Overfit", ST["body_bold"]),
            Paragraph("Best?", ST["body_bold"]),
        ]]
        for m in valid:
            rows.append([
                Paragraph(m.name, ST["body"]),
                Paragraph("{:.4f} ±{:.4f}".format(
                    m.cv_score, m.cv_std), ST["body"]),
                Paragraph("{:.4f}".format(m.test_score), ST["body"]),
                Paragraph(m.overfit_label, ST["body"]),
                Paragraph("YES" if m.is_best else "", ST["body"]),
            ])
        _table(story, theme, rows,
               [CW*0.30, CW*0.22, CW*0.18, CW*0.15, CW*0.15])

    # Feature importance
    if ml_report.feature_importance:
        story.append(Paragraph("Top Feature Importances", ST["sub_h"]))
        rows = [[
            Paragraph("Rank", ST["body_bold"]),
            Paragraph("Feature", ST["body_bold"]),
            Paragraph("Importance", ST["body_bold"]),
            Paragraph("Direction", ST["body_bold"]),
        ]]
        for fi in ml_report.feature_importance[:8]:
            rows.append([
                Paragraph(str(fi.rank), ST["body"]),
                Paragraph(fi.feature, ST["body"]),
                Paragraph("{:.1f}%".format(fi.importance*100), ST["body"]),
                Paragraph(fi.direction.title(), ST["body"]),
            ])
        _table(story, theme, rows, [CW*0.1, CW*0.35, CW*0.25, CW*0.30])

    # ML insights
    if ml_report.insights:
        story.append(Paragraph("ML Insights", ST["sub_h"]))
        for ins in ml_report.insights[:5]:
            story.append(Paragraph("• " + ins, ST["insight"]))


def _chart_page(story, ST, theme, img_bytes, title, narrative, num, CW):
    """One chart + narrative per page."""
    _page_h(story, ST, theme, "Chart {}: {}".format(num, title))

    if img_bytes:
        try:
            img = Image(io.BytesIO(img_bytes),
                        width=CW, height=CW*0.50)
            story.append(KeepTogether([img, Spacer(1, 3*mm)]))
        except Exception:
            pass

    if narrative:
        story.append(Paragraph("Analysis", ST["sub_h"]))
        _narrative_box(story, ST, theme, narrative)


def _recommendations(story, ST, theme, actions, CW):
    """Action plan page."""
    _page_h(story, ST, theme, "Recommendations & Action Plan")

    priority_labels = [
        ("IMMEDIATE", "negative"),
        ("SHORT TERM", "warning"),
        ("SHORT TERM", "warning"),
        ("MEDIUM TERM", "accent"),
        ("MEDIUM TERM", "accent"),
        ("LONG TERM", "positive"),
    ]

    for i, action in enumerate(actions[:9]):
        label, color_key = priority_labels[min(i, len(priority_labels)-1)]
        T = theme
        story.append(Paragraph(
            "[{}] {}".format(label, action),
            ST["action"]
        ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "All recommendations are based solely on the provided dataset. "
        "Please verify findings with domain experts before making business decisions.",
        ST["caption"]
    ))


def _appendix(story, ST, theme, config, df, CW):
    """Appendix with methodology."""
    _page_h(story, ST, theme, "Appendix")

    story.append(Paragraph("A. Methodology", ST["sub_h"]))
    story.append(Paragraph(
        "Data quality scoring uses a weighted composite formula: "
        "60% completeness, 30% deduplication, 10% column health. "
        "Outlier detection uses IQR (1.5x) and Modified Z-Score methods. "
        "Normality tested using Shapiro-Wilk, D'Agostino-Pearson, and "
        "Anderson-Darling tests (majority vote). "
        "Correlations use Pearson (normal data) or Spearman (non-normal). "
        "ML models evaluated using 5-fold cross-validation. "
        "AI narratives generated using Groq Llama 3.3 70B.",
        ST["body"]
    ))

    story.append(Paragraph("B. Quality Score Formula", ST["sub_h"]))
    rows = [
        [Paragraph("Component", ST["body_bold"]),
         Paragraph("Weight", ST["body_bold"]),
         Paragraph("Description", ST["body_bold"])],
        [Paragraph("Completeness", ST["body"]),
         Paragraph("60%", ST["body"]),
         Paragraph("Percentage of non-missing cells", ST["body"])],
        [Paragraph("Deduplication", ST["body"]),
         Paragraph("30%", ST["body"]),
         Paragraph("Percentage of unique rows", ST["body"])],
        [Paragraph("Column Health", ST["body"]),
         Paragraph("10%", ST["body"]),
         Paragraph("Average per-column quality score", ST["body"])],
    ]
    _table(story, theme, rows, [CW*0.25, CW*0.15, CW*0.60])

    story.append(Paragraph("C. Disclaimer", ST["sub_h"]))
    story.append(Paragraph(
        "This report was automatically generated by DataForge AI on {} "
        "for {}. All findings are based solely on the provided dataset "
        "and should be verified by a qualified data analyst before "
        "making business decisions.".format(
            datetime.now().strftime("%B %d, %Y"),
            config.get("client_name","Client")),
        ST["body"]
    ))


# ══════════════════════════════════════════════════════════
#  MAIN BUILD FUNCTION
# ══════════════════════════════════════════════════════════

def build_pdf(
    df: pd.DataFrame,
    config: dict,
    profile=None,
    cleaning_summary: dict = None,
    stats_report=None,
    bi_report=None,
    ml_report=None,
    chart_data: list = None,       # [(title, img_bytes, narrative)]
    executive_summary: str = "",
    findings: list = None,
    risks: list = None,
    opportunities: list = None,
    recommendations: list = None,
) -> bytes:
    """
    Build complete PDF report.
    All parameters are optional — gracefully skips missing sections.

    config dict keys:
        title, subtitle, client_name, confidential,
        theme_name (Corporate Light / Dark Tech / Executive Green)
    """
    findings      = findings or []
    risks         = risks or []
    opportunities = opportunities or []
    recommendations = recommendations or []
    chart_data    = chart_data or []

    theme_name = config.get("theme_name", "Corporate Light")
    theme      = THEMES.get(theme_name, THEMES["Corporate Light"])

    buf = io.BytesIO()
    W, H = A4
    M    = 18 * mm
    CW   = W - 2*M

    # ── Header / Footer callback ──────────────────────────
    def _hf(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(_rgb(theme["header_bg"]))
        canvas.rect(0, H - 12*mm, W, 12*mm, fill=1, stroke=0)
        canvas.setFillColor(_rgb(theme["header_text"]))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(M, H - 7.5*mm, "DataForge AI")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            W - M, H - 7.5*mm,
            config.get("title","Data Analysis Report")[:55]
        )
        canvas.setStrokeColor(_rgb(theme["accent"]))
        canvas.setLineWidth(0.4)
        canvas.line(M, 11*mm, W - M, 11*mm)
        canvas.setFillColor(_rgb(theme["text_muted"]))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            M, 7*mm,
            "Prepared for: " + config.get("client_name","Client")
        )
        if config.get("confidential", True):
            canvas.setFillColor(_rgb(theme["negative"]))
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawCentredString(W/2, 7*mm, "CONFIDENTIAL")
        canvas.setFillColor(_rgb(theme["text_muted"]))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(W - M, 7*mm, "Page " + str(doc.page))
        canvas.restoreState()

    frame = Frame(
        M, 14*mm, CW, H - 28*mm,
        leftPadding=0, rightPadding=0,
        topPadding=4*mm, bottomPadding=2*mm
    )
    tpl = PageTemplate(id="main", frames=[frame], onPage=_hf)
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        pageTemplates=[tpl],
        leftMargin=M, rightMargin=M,
        topMargin=14*mm, bottomMargin=14*mm,
    )

    ST    = _styles(theme, CW)
    story = []

    # ── TOC sections ──────────────────────────────────────
    page_num = 3
    toc_sections = []
    toc_sections.append((1, "Executive Summary", page_num)); page_num += 1
    toc_sections.append((2, "Dataset Overview", page_num)); page_num += 1
    if stats_report:
        toc_sections.append((3, "Statistical Analysis", page_num)); page_num += 1
    if bi_report:
        toc_sections.append((4, "Business Intelligence", page_num)); page_num += 1
    if ml_report:
        toc_sections.append((5, "ML Predictions", page_num)); page_num += 1
    for i, (title, _, _) in enumerate(chart_data, 1):
        toc_sections.append((
            len(toc_sections)+1,
            "Chart {}: {}".format(i, title[:35]),
            page_num
        )); page_num += 1
    toc_sections.append((len(toc_sections)+1, "Recommendations", page_num)); page_num += 1
    toc_sections.append((len(toc_sections)+1, "Appendix", page_num))

    # ── Build story ───────────────────────────────────────
    _cover(story, ST, theme, config, CW)
    story.append(PageBreak())

    _toc(story, ST, theme, toc_sections, CW)
    story.append(PageBreak())

    _exec_summary(story, ST, theme, executive_summary,
                  findings, risks, opportunities, CW)
    story.append(PageBreak())

    _dataset_overview(story, ST, theme, df, profile, cleaning_summary, CW)
    story.append(PageBreak())

    if stats_report:
        _statistical_analysis(story, ST, theme, stats_report, CW)
        story.append(PageBreak())

    if bi_report:
        _bi_section(story, ST, theme, bi_report, CW)
        story.append(PageBreak())

    if ml_report:
        _ml_section(story, ST, theme, ml_report, CW)
        story.append(PageBreak())

    for i, (title, img_bytes, narrative) in enumerate(chart_data, 1):
        _chart_page(story, ST, theme, img_bytes, title, narrative, i, CW)
        story.append(PageBreak())

    if recommendations:
        _recommendations(story, ST, theme, recommendations, CW)
        story.append(PageBreak())

    _appendix(story, ST, theme, config, df, CW)

    doc.build(story)
    buf.seek(0)
    return buf.read()
