import io
import pandas as pd
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from core.data_profiler import DatasetProfile
from core.report_engine import ReportConfig


# ── Colours ───────────────────────────────────────────────
BLUE    = colors.HexColor("#4f8ef7")
GREEN   = colors.HexColor("#22d3a5")
ORANGE  = colors.HexColor("#f7934f")
RED     = colors.HexColor("#f77070")
DARK    = colors.HexColor("#07080f")
DARKER  = colors.HexColor("#0e0f1a")
LIGHT   = colors.HexColor("#dde1f5")
MUTED   = colors.HexColor("#636a8a")
WHITE   = colors.white
PAGE_W, PAGE_H = A4


def build_pdf(
    df: pd.DataFrame,
    profile: DatasetProfile,
    config: ReportConfig,
    chart_images: list,
    chart_titles: list,
    chart_narratives: list,
    executive_summary: list,
    recommendations: dict,
) -> bytes:
    """
    Build complete PDF report.

    Args:
        df:                 Cleaned DataFrame
        profile:            DatasetProfile from profiler
        config:             ReportConfig (title, client name etc)
        chart_images:       List of PNG bytes for each chart
        chart_titles:       List of chart title strings
        chart_narratives:   List of AI narrative strings per chart
        executive_summary:  List of finding dicts from narrator
        recommendations:    Dict with immediate/short_term/long_term

    Returns:
        PDF as bytes
    """
    buf      = io.BytesIO()
    styles   = _build_styles()
    doc      = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15*mm,
        rightMargin=15*mm,
        topMargin=15*mm,
        bottomMargin=15*mm,
    )

    story = []

    # ── Pages ──────────────────────────────────────────────
    _cover_page(story, styles, config)
    story.append(PageBreak())

    _executive_summary_page(story, styles, config, executive_summary)
    story.append(PageBreak())

    _data_quality_page(story, styles, profile)
    story.append(PageBreak())

    _metrics_page(story, styles, df, profile)
    story.append(PageBreak())

    # One page per chart
    for i, (img_bytes, title, narrative) in enumerate(
        zip(chart_images, chart_titles, chart_narratives)
    ):
        _chart_page(story, styles, img_bytes, title, narrative, i + 1)
        story.append(PageBreak())

    _anomaly_page(story, styles, df, profile)
    story.append(PageBreak())

    _recommendations_page(story, styles, recommendations)
    story.append(PageBreak())

    _appendix_page(story, styles, df, profile, config)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── PAGE BUILDERS ─────────────────────────────────────────

def _cover_page(story, styles, config: ReportConfig):
    story.append(Spacer(1, 40*mm))

    # Title block
    story.append(Paragraph("🔬 DataForge AI", styles["brand"]))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=3, color=BLUE))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(config.title, styles["cover_title"]))
    story.append(Spacer(1, 4*mm))

    if config.client_name:
        story.append(Paragraph(f"Prepared for: {config.client_name}", styles["cover_sub"]))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y')}",
        styles["cover_date"]
    ))
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=MUTED))
    story.append(Spacer(1, 4*mm))

    if config.confidential:
        story.append(Paragraph("CONFIDENTIAL", styles["confidential"]))

    story.append(Spacer(1, 40*mm))
    story.append(Paragraph(f"Prepared by: {config.prepared_by}", styles["cover_prepared"]))


def _executive_summary_page(story, styles, config, findings: list):
    story.append(Paragraph("Executive Summary", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        f"This report presents a comprehensive analysis of the dataset "
        f"provided by {config.client_name}. The following key findings "
        f"were identified:",
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))

    for i, item in enumerate(findings):
        finding = item.get("finding", str(item))
        ftype   = item.get("type", "neutral")
        icon    = "✅" if ftype == "positive" else "⚠️" if ftype == "negative" else "📌"
        c       = GREEN if ftype == "positive" else ORANGE if ftype == "negative" else BLUE

        row = Table(
            [[
                Paragraph(f"{i+1}.", styles["finding_num"]),
                Paragraph(f"{icon}  {finding}", styles["finding_text"])
            ]],
            colWidths=[10*mm, 160*mm]
        )
        row.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), DARKER),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("ROUNDEDCORNERS", (0,0), (-1,-1), 4),
            ("LINEBELOW", (0,0), (-1,-1), 0.5, colors.HexColor("#1e2035")),
        ]))
        story.append(row)
        story.append(Spacer(1, 2*mm))


def _data_quality_page(story, styles, profile: DatasetProfile):
    story.append(Paragraph("Data Quality Report", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN))
    story.append(Spacer(1, 4*mm))

    # Score card
    score = profile.overall_quality_score
    sc    = GREEN if score >= 80 else ORANGE if score >= 60 else RED
    score_tbl = Table(
        [[
            Paragraph("Overall Quality Score", styles["kpi_label"]),
            Paragraph(f"{score}/100", styles["kpi_value"]),
            Paragraph(
                "Good" if score >= 80 else "Fair" if score >= 60 else "Poor",
                styles["kpi_sub"]
            )
        ]],
        colWidths=[70*mm, 60*mm, 40*mm]
    )
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), DARKER),
        ("LINEAFTER",    (0,0), (0,-1),  2, sc),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 4*mm))

    # Dataset stats
    stats_data = [
        ["Metric", "Value"],
        ["Total Rows",       f"{profile.rows:,}"],
        ["Total Columns",    str(profile.cols)],
        ["Missing Cells",    f"{profile.missing_cells:,} ({profile.missing_pct}%)"],
        ["Duplicate Rows",   f"{profile.duplicate_rows:,} ({profile.duplicate_pct}%)"],
        ["Numeric Columns",  str(len(profile.numeric_cols))],
        ["Category Columns", str(len(profile.categorical_cols))],
        ["DateTime Columns", str(len(profile.datetime_cols))],
    ]
    _styled_table(story, stats_data, col_widths=[90*mm, 80*mm])
    story.append(Spacer(1, 4*mm))

    # Column breakdown
    story.append(Paragraph("Column Analysis", styles["section_title"]))
    story.append(Spacer(1, 2*mm))

    col_data = [["Column", "Type", "Missing %", "Unique", "Outliers", "Score"]]
    for p in profile.column_profiles:
        col_data.append([
            p.name[:20],
            p.dtype[:10],
            f"{p.missing_pct}%",
            str(p.unique_count),
            str(p.outlier_count) if p.has_outliers else "—",
            f"{p.quality_score}/100",
        ])
    _styled_table(story, col_data,
        col_widths=[50*mm, 25*mm, 25*mm, 25*mm, 25*mm, 25*mm])

    # Recommendations
    if profile.recommendations:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("Recommendations", styles["section_title"]))
        story.append(Spacer(1, 2*mm))
        for rec in profile.recommendations[:6]:
            story.append(Paragraph(f"• {rec}", styles["bullet"]))
            story.append(Spacer(1, 1*mm))


def _metrics_page(story, styles, df: pd.DataFrame, profile: DatasetProfile):
    story.append(Paragraph("Key Metrics Overview", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 4*mm))

    num_cols = profile.numeric_cols[:6]
    if not num_cols:
        story.append(Paragraph("No numeric columns found.", styles["body"]))
        return

    # KPI grid — 3 per row
    kpi_rows = []
    row = []
    for i, col in enumerate(num_cols):
        s = df[col].dropna()
        cell = Table([[
            Paragraph(col[:15], styles["kpi_label"]),
        ],[
            Paragraph(f"{s.sum():,.0f}", styles["kpi_value"]),
        ],[
            Paragraph(f"avg {s.mean():,.1f}", styles["kpi_sub"]),
        ]], colWidths=[55*mm])
        cell.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), DARKER),
            ("LINEABOVE",     (0,0), (-1,0),  3, BLUE),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        row.append(cell)
        if len(row) == 3:
            kpi_rows.append(row)
            row = []
    if row:
        while len(row) < 3:
            row.append(Paragraph("", styles["body"]))
        kpi_rows.append(row)

    for r in kpi_rows:
        t = Table([r], colWidths=[59*mm, 59*mm, 59*mm])
        t.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
        story.append(t)
        story.append(Spacer(1, 3*mm))

    # Stats table
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Descriptive Statistics", styles["section_title"]))
    story.append(Spacer(1, 2*mm))

    desc    = df[num_cols].describe().round(2)
    headers = ["Statistic"] + [c[:12] for c in num_cols]
    rows_   = [headers]
    for idx in desc.index:
        rows_.append([idx] + [str(desc.loc[idx, c]) for c in num_cols])
    _styled_table(story, rows_)


def _chart_page(story, styles, img_bytes: bytes, title: str, narrative: str, num: int):
    story.append(Paragraph(f"Chart {num}: {title}", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 4*mm))

    # Embed chart image
    try:
        img_buf = io.BytesIO(img_bytes)
        img     = Image(img_buf, width=175*mm, height=90*mm)
        story.append(img)
    except Exception:
        story.append(Paragraph("[Chart could not be rendered]", styles["muted"]))

    story.append(Spacer(1, 4*mm))

    # AI narrative
    story.append(Paragraph("Analysis", styles["section_title"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(narrative, styles["body"]))


def _anomaly_page(story, styles, df: pd.DataFrame, profile: DatasetProfile):
    story.append(Paragraph("Anomaly Detection", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE))
    story.append(Spacer(1, 4*mm))

    num_cols = profile.numeric_cols
    if not num_cols:
        story.append(Paragraph("No numeric columns for anomaly detection.", styles["body"]))
        return

    found_any = False
    for col in num_cols[:4]:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        out    = df[(df[col] < lo) | (df[col] > hi)]

        if not out.empty:
            found_any = True
            story.append(Paragraph(f"⚠️  {col}", styles["section_title"]))
            story.append(Paragraph(
                f"{len(out):,} outliers detected ({len(out)/max(len(df),1)*100:.1f}% of rows). "
                f"Normal range: {lo:,.2f} – {hi:,.2f}. "
                f"Values outside this range may require investigation.",
                styles["body"]
            ))
            story.append(Spacer(1, 2*mm))

            # Show top 5 outlier rows
            show_cols = [col] + [c for c in profile.categorical_cols[:2]]
            show_cols = [c for c in show_cols if c in df.columns]
            tbl_data  = [show_cols] + out[show_cols].head(5).astype(str).values.tolist()
            _styled_table(story, tbl_data)
            story.append(Spacer(1, 3*mm))

    if not found_any:
        story.append(Paragraph(
            "✅  No significant outliers detected in numeric columns.",
            styles["body"]
        ))


def _recommendations_page(story, styles, recs: dict):
    story.append(Paragraph("Recommendations", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN))
    story.append(Spacer(1, 4*mm))

    sections = [
        ("🔴 Immediate Actions", "immediate",  RED),
        ("🟠 Short Term",        "short_term", ORANGE),
        ("🟢 Long Term",         "long_term",  GREEN),
    ]

    for title, key, colour in sections:
        items = recs.get(key, [])
        if not items:
            continue
        story.append(Paragraph(title, styles["section_title"]))
        story.append(Spacer(1, 1*mm))
        for item in items:
            story.append(Paragraph(f"□  {item}", styles["bullet"]))
            story.append(Spacer(1, 1*mm))
        story.append(Spacer(1, 3*mm))


def _appendix_page(story, styles, df, profile, config):
    story.append(Paragraph("Appendix", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=MUTED))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Methodology", styles["section_title"]))
    story.append(Paragraph(
        "Data quality scoring uses a weighted composite: "
        "60% completeness, 30% deduplication, 10% column health. "
        "Outlier detection uses the IQR method (1.5× interquartile range). "
        "AI narratives generated using Groq Llama 3.3 70B.",
        styles["body"]
    ))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Disclaimer", styles["section_title"]))
    story.append(Paragraph(
        f"This report was automatically generated by DataForge AI on "
        f"{datetime.now().strftime('%B %d, %Y')} for {config.client_name}. "
        f"All findings are based on the provided dataset and should be "
        f"verified by a qualified analyst before making business decisions.",
        styles["body"]
    ))


# ── HELPERS ───────────────────────────────────────────────

def _styled_table(story, data: list, col_widths=None):
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0),  9),
        ("TOPPADDING",   (0,0), (-1,0),  5),
        ("BOTTOMPADDING",(0,0), (-1,0),  5),
        # Body
        ("BACKGROUND",   (0,1), (-1,-1), DARKER),
        ("TEXTCOLOR",    (0,1), (-1,-1), LIGHT),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,1), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [DARKER, colors.HexColor("#111220")]),
        ("TOPPADDING",   (0,1), (-1,-1), 4),
        ("BOTTOMPADDING",(0,1), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#1e2035")),
    ]))
    story.append(tbl)


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "brand": s("brand",
            fontSize=22, textColor=BLUE,
            fontName="Helvetica-Bold", alignment=TA_CENTER),

        "cover_title": s("cover_title",
            fontSize=28, textColor=LIGHT,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
            spaceAfter=4),

        "cover_sub": s("cover_sub",
            fontSize=14, textColor=MUTED,
            alignment=TA_CENTER),

        "cover_date": s("cover_date",
            fontSize=11, textColor=MUTED,
            alignment=TA_CENTER),

        "cover_prepared": s("cover_prepared",
            fontSize=10, textColor=MUTED,
            alignment=TA_CENTER),

        "confidential": s("confidential",
            fontSize=10, textColor=ORANGE,
            fontName="Helvetica-Bold", alignment=TA_CENTER),

        "page_title": s("page_title",
            fontSize=20, textColor=LIGHT,
            fontName="Helvetica-Bold",
            spaceBefore=2, spaceAfter=4),

        "section_title": s("section_title",
            fontSize=13, textColor=BLUE,
            fontName="Helvetica-Bold",
            spaceBefore=4, spaceAfter=2),

        "body": s("body",
            fontSize=10, textColor=LIGHT,
            leading=16, spaceAfter=4),

        "bullet": s("bullet",
            fontSize=10, textColor=LIGHT,
            leftIndent=10, leading=15),

        "muted": s("muted",
            fontSize=9, textColor=MUTED),

        "finding_num": s("finding_num",
            fontSize=11, textColor=BLUE,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER),

        "finding_text": s("finding_text",
            fontSize=10, textColor=LIGHT,
            leading=15),

        "kpi_label": s("kpi_label",
            fontSize=9, textColor=MUTED,
            fontName="Helvetica-Bold"),

        "kpi_value": s("kpi_value",
            fontSize=18, textColor=LIGHT,
            fontName="Helvetica-Bold"),

        "kpi_sub": s("kpi_sub",
            fontSize=9, textColor=MUTED),
    }
