import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from core.data_profiler import DatasetProfile
from core.report_engine import ReportConfig, ReportTheme
import pandas as pd


def _rgb(t): return colors.Color(t[0]/255, t[1]/255, t[2]/255)


def build_pdf(
    df: pd.DataFrame,
    profile: DatasetProfile,
    config: ReportConfig,
    chart_data: list,
    chart_narratives: list,
    executive_summary: list,
    recommendations: dict,
) -> bytes:

    buf   = io.BytesIO()
    theme = config.theme
    W, H  = A4
    m     = theme.page_margin_mm * mm

    def _header_footer(canvas, doc):
        canvas.saveState()
        T = theme
        canvas.setFillColor(_rgb(T.header_bg))
        canvas.rect(0, H - 14*mm, W, 14*mm, fill=1, stroke=0)
        canvas.setFillColor(_rgb(T.header_text))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(m, H - 9*mm, "DataForge AI")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - m, H - 9*mm, config.title[:50])
        canvas.setStrokeColor(_rgb(T.accent))
        canvas.setLineWidth(0.5)
        canvas.line(m, 12*mm, W - m, 12*mm)
        canvas.setFillColor(_rgb(T.text_muted))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(m, 8*mm, f"Prepared for: {config.client_name}")
        if config.confidential:
            canvas.setFillColor(_rgb(T.negative_color))
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawCentredString(W/2, 8*mm, "CONFIDENTIAL")
        canvas.setFillColor(_rgb(T.text_muted))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(W - m, 8*mm, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(
        m, 16*mm, W - 2*m, H - 30*mm,
        leftPadding=0, rightPadding=0,
        topPadding=4*mm, bottomPadding=4*mm
    )
    template = PageTemplate(
        id="main", frames=[frame],
        onPage=_header_footer
    )
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        pageTemplates=[template],
        leftMargin=m, rightMargin=m,
        topMargin=18*mm, bottomMargin=18*mm,
    )

    styles = _build_styles(theme)
    story  = []

    _cover_page(story, styles, config, theme, W, H)
    story.append(PageBreak())
    _toc_page(story, styles, theme, chart_data)
    story.append(PageBreak())
    _executive_summary_page(story, styles, theme, executive_summary)
    story.append(PageBreak())
    _dataset_overview_page(story, styles, theme, df, profile)
    story.append(PageBreak())
    _data_quality_page(story, styles, theme, profile)
    story.append(PageBreak())
    _metrics_page(story, styles, theme, df, profile)
    story.append(PageBreak())

    for i, (title, img_bytes) in enumerate(chart_data):
        narrative = chart_narratives[i] if i < len(chart_narratives) else ""
        _chart_page(story, styles, theme, img_bytes, title, narrative, i+1)
        story.append(PageBreak())

    _anomaly_page(story, styles, theme, df, profile)
    story.append(PageBreak())
    _recommendations_page(story, styles, theme, recommendations)
    story.append(PageBreak())
    _appendix_page(story, styles, theme, config, profile)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── PAGE BUILDERS ─────────────────────────────────────────

def _cover_page(story, styles, config, theme, W, H):
    T = theme
    story.append(Spacer(1, 20*mm))
    story.append(HRFlowable(width="100%", thickness=4, color=_rgb(T.accent)))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(config.title, styles["cover_title"]))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="40%", thickness=1, color=_rgb(T.text_muted)))
    story.append(Spacer(1, 6*mm))

    info = Table([
        [Paragraph("Prepared for:", styles["cover_label"]),
         Paragraph(config.client_name, styles["cover_value"])],
        [Paragraph("Prepared by:", styles["cover_label"]),
         Paragraph(config.prepared_by, styles["cover_value"])],
        [Paragraph("Date:", styles["cover_label"]),
         Paragraph(datetime.now().strftime("%B %d, %Y"), styles["cover_value"])],
        [Paragraph("Theme:", styles["cover_label"]),
         Paragraph(config.theme_name, styles["cover_value"])],
    ], colWidths=[40*mm, 100*mm])
    info.setStyle(TableStyle([
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
    ]))
    story.append(info)
    story.append(Spacer(1, 10*mm))

    if config.confidential:
        conf_tbl = Table(
            [[Paragraph("CONFIDENTIAL DOCUMENT", styles["confidential"])]],
            colWidths=[140*mm]
        )
        conf_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _rgb(T.negative_color)),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(conf_tbl)

    story.append(Spacer(1, 30*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=_rgb(T.text_muted)))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "This report was automatically generated by DataForge AI.",
        styles["cover_footer"]
    ))


def _toc_page(story, styles, theme, chart_data):
    T = theme
    story.append(Paragraph("Table of Contents", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 6*mm))

    toc_items = [
        ("1.", "Executive Summary",   "3"),
        ("2.", "Dataset Overview",    "4"),
        ("3.", "Data Quality Report", "5"),
        ("4.", "Key Metrics",         "6"),
    ]
    pg = 7
    for i, (title, _) in enumerate(chart_data):
        toc_items.append((f"{4+i+1}.", f"Chart {i+1}: {title}", str(pg)))
        pg += 1
    toc_items += [
        (str(pg)+".",   "Anomaly Detection", str(pg)),
        (str(pg+1)+".", "Recommendations",   str(pg+1)),
        (str(pg+2)+".", "Appendix",          str(pg+2)),
    ]

    for num, title, page in toc_items:
        row = Table([[
            Paragraph(num,   styles["toc_num"]),
            Paragraph(title, styles["toc_item"]),
            Paragraph(page,  styles["toc_page"]),
        ]], colWidths=[12*mm, 140*mm, 18*mm])
        row.setStyle(TableStyle([
            ("LINEBELOW",     (0,0), (-1,-1), 0.3, _rgb(T.table_border)),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ]))
        story.append(row)


def _executive_summary_page(story, styles, theme, findings):
    T = theme
    story.append(Paragraph("Executive Summary", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "The following key findings were identified from the dataset analysis:",
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))

    for i, item in enumerate(findings):
        finding = item.get("finding", str(item)) if isinstance(item, dict) else str(item)
        ftype   = item.get("type", "neutral") if isinstance(item, dict) else "neutral"

        if ftype == "positive":   marker_c = T.positive_color; marker = "+"
        elif ftype == "negative": marker_c = T.negative_color; marker = "!"
        else:                     marker_c = T.accent;          marker = str(i+1)

        row = Table([[
            Table([[Paragraph(marker, styles["finding_marker"])]],
                colWidths=[8*mm],
                style=TableStyle([
                    ("BACKGROUND",    (0,0),(-1,-1), _rgb(marker_c)),
                    ("TOPPADDING",    (0,0),(-1,-1), 4),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                    ("LEFTPADDING",   (0,0),(-1,-1), 2),
                    ("RIGHTPADDING",  (0,0),(-1,-1), 2),
                ])),
            Paragraph(finding, styles["finding_text"]),
        ]], colWidths=[12*mm, 158*mm])
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _rgb(T.accent_light)),
            ("LINEBELOW",     (0,0), (-1,-1), 0.5, _rgb(T.table_border)),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(row)
        story.append(Spacer(1, 2*mm))


def _dataset_overview_page(story, styles, theme, df, profile):
    T = theme
    story.append(Paragraph("Dataset Overview", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))

    score_c = T.positive_color if profile.overall_quality_score >= 80 \
              else T.negative_color if profile.overall_quality_score < 60 \
              else T.neutral_color

    def kpi_cell(label, value, sub="", accent=None):
        ac = accent or T.accent
        return Table([[
            Paragraph(label, styles["kpi_label"]),
        ],[
            Paragraph(str(value), styles["kpi_value"]),
        ],[
            Paragraph(sub, styles["kpi_sub"]),
        ]], colWidths=[42*mm],
        style=TableStyle([
            ("LINEABOVE",     (0,0),(-1,0), 3, _rgb(ac)),
            ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ]))

    kpi_row = Table([[
        kpi_cell("TOTAL ROWS",    f"{profile.rows:,}", "records"),
        kpi_cell("COLUMNS",       str(profile.cols),   "features"),
        kpi_cell("MISSING",       f"{profile.missing_pct}%",
                 f"{profile.missing_cells:,} cells",
                 T.negative_color if profile.missing_pct > 10 else T.positive_color),
        kpi_cell("QUALITY SCORE", f"{profile.overall_quality_score}/100",
                 "Good" if profile.overall_quality_score >= 80 else "Fair",
                 score_c),
    ]], colWidths=[43*mm, 43*mm, 43*mm, 43*mm])
    kpi_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 2),
        ("RIGHTPADDING", (0,0),(-1,-1), 2),
    ]))
    story.append(kpi_row)
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("Column Type Breakdown", styles["section_title"]))
    story.append(Spacer(1, 2*mm))
    type_data = [["Type", "Count", "Columns"]]
    type_data.append(["Numeric", str(len(profile.numeric_cols)),
        ", ".join(profile.numeric_cols[:5]) + ("..." if len(profile.numeric_cols) > 5 else "")])
    type_data.append(["Categorical", str(len(profile.categorical_cols)),
        ", ".join(profile.categorical_cols[:5]) + ("..." if len(profile.categorical_cols) > 5 else "")])
    type_data.append(["DateTime", str(len(profile.datetime_cols)),
        ", ".join(profile.datetime_cols[:5]) + ("..." if len(profile.datetime_cols) > 5 else "")])
    _render_table(story, theme, type_data, col_widths=[35*mm, 25*mm, 110*mm])
    story.append(Spacer(1, 5*mm))

    if profile.numeric_cols:
        story.append(Paragraph("Descriptive Statistics", styles["section_title"]))
        story.append(Spacer(1, 2*mm))
        cols  = profile.numeric_cols[:6]
        desc  = df[cols].describe().round(2)
        hdr   = ["Stat"] + [c[:10] for c in cols]
        rows_ = [hdr]
        for idx in ["mean","std","min","25%","50%","75%","max"]:
            if idx in desc.index:
                rows_.append([idx] + [str(desc.loc[idx, c]) for c in cols])
        cw = [20*mm] + [max(170//len(cols), 20)*mm] * len(cols)
        _render_table(story, theme, rows_, col_widths=cw)


def _data_quality_page(story, styles, theme, profile):
    T = theme
    story.append(Paragraph("Data Quality Report", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))

    score   = profile.overall_quality_score
    score_c = T.positive_color if score >= 80 \
              else T.negative_color if score < 60 else T.neutral_color
    label   = "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"

    banner = Table([[
        Paragraph("Overall Quality Score", styles["section_title"]),
        Paragraph(f"{score}/100  -  {label}",
                  ParagraphStyle("sc", fontSize=14,
                    textColor=_rgb(score_c),
                    fontName="Helvetica-Bold",
                    alignment=TA_RIGHT)),
    ]], colWidths=[100*mm, 70*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0
