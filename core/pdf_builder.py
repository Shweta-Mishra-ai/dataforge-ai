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


def _rgb(t):
    return colors.Color(t[0]/255, t[1]/255, t[2]/255)


def build_pdf(
    df,
    profile,
    config,
    chart_data,
    chart_narratives,
    executive_summary,
    recommendations,
):
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
        canvas.drawString(m, 8*mm, "Prepared for: " + config.client_name)
        if config.confidential:
            canvas.setFillColor(_rgb(T.negative_color))
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawCentredString(W/2, 8*mm, "CONFIDENTIAL")
        canvas.setFillColor(_rgb(T.text_muted))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(W - m, 8*mm, "Page " + str(doc.page))
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
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))
    story.append(info)
    story.append(Spacer(1, 10*mm))

    if config.confidential:
        conf_tbl = Table(
            [[Paragraph("CONFIDENTIAL DOCUMENT", styles["confidential"])]],
            colWidths=[140*mm]
        )
        conf_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.negative_color)),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
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
        toc_items.append((str(4+i+1)+".", "Chart " + str(i+1) + ": " + title, str(pg)))
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
            ("LINEBELOW",     (0, 0), (-1, -1), 0.3, _rgb(T.table_border)),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
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

        if ftype == "positive":
            marker_c = T.positive_color
            marker   = "+"
        elif ftype == "negative":
            marker_c = T.negative_color
            marker   = "!"
        else:
            marker_c = T.accent
            marker   = str(i+1)

        inner = Table(
            [[Paragraph(marker, styles["finding_marker"])]],
            colWidths=[8*mm],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), _rgb(marker_c)),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 2),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
            ])
        )
        row = Table(
            [[inner, Paragraph(finding, styles["finding_text"])]],
            colWidths=[12*mm, 158*mm]
        )
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.accent_light)),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.5, _rgb(T.table_border)),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
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
        return Table(
            [[Paragraph(label, styles["kpi_label"])],
             [Paragraph(str(value), styles["kpi_value"])],
             [Paragraph(sub, styles["kpi_sub"])]],
            colWidths=[42*mm],
            style=TableStyle([
                ("LINEABOVE",     (0, 0), (-1, 0), 3, _rgb(ac)),
                ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.accent_light)),
                ("LEFTPADDING",   (0, 0), (-1, -1), 5),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )

    kpi_row = Table([[
        kpi_cell("TOTAL ROWS",    str(profile.rows) + " rows", "records"),
        kpi_cell("COLUMNS",       str(profile.cols),            "features"),
        kpi_cell("MISSING",       str(profile.missing_pct) + "%",
                 str(profile.missing_cells) + " cells",
                 T.negative_color if profile.missing_pct > 10 else T.positive_color),
        kpi_cell("QUALITY SCORE", str(profile.overall_quality_score) + "/100",
                 "Good" if profile.overall_quality_score >= 80 else "Fair",
                 score_c),
    ]], colWidths=[43*mm, 43*mm, 43*mm, 43*mm])
    kpi_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(kpi_row)
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("Column Type Breakdown", styles["section_title"]))
    story.append(Spacer(1, 2*mm))
    nc = ", ".join(profile.numeric_cols[:5]) + ("..." if len(profile.numeric_cols) > 5 else "")
    cc = ", ".join(profile.categorical_cols[:5]) + ("..." if len(profile.categorical_cols) > 5 else "")
    dc = ", ".join(profile.datetime_cols[:5]) + ("..." if len(profile.datetime_cols) > 5 else "")
    type_data = [
        ["Type",        "Count",                          "Columns"],
        ["Numeric",     str(len(profile.numeric_cols)),     nc],
        ["Categorical", str(len(profile.categorical_cols)), cc],
        ["DateTime",    str(len(profile.datetime_cols)),    dc],
    ]
    _render_table(story, theme, type_data, col_widths=[35*mm, 25*mm, 110*mm])
    story.append(Spacer(1, 5*mm))

    if profile.numeric_cols:
        story.append(Paragraph("Descriptive Statistics", styles["section_title"]))
        story.append(Spacer(1, 2*mm))
        cols  = profile.numeric_cols[:6]
        desc  = df[cols].describe().round(2)
        hdr   = ["Stat"] + [c[:10] for c in cols]
        rows_ = [hdr]
        for idx in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
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
        Paragraph(str(score) + "/100  -  " + label,
                  ParagraphStyle("sc", fontSize=14,
                                 textColor=_rgb(score_c),
                                 fontName="Helvetica-Bold",
                                 alignment=TA_RIGHT)),
    ]], colWidths=[100*mm, 70*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.accent_light)),
        ("LINEABOVE",     (0, 0), (-1, 0),  3, _rgb(score_c)),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("Column Analysis", styles["section_title"]))
    story.append(Spacer(1, 2*mm))
    rows_ = [["Column", "Type", "Missing %", "Unique", "Outliers", "Score"]]
    for p in profile.column_profiles:
        rows_.append([
            p.name[:18],
            p.dtype[:8],
            str(p.missing_pct) + "%",
            str(p.unique_count),
            str(p.outlier_count) if p.has_outliers else "-",
            str(p.quality_score) + "/100",
        ])
    _render_table(story, theme, rows_,
                  col_widths=[48*mm, 22*mm, 24*mm, 22*mm, 22*mm, 24*mm])

    if profile.recommendations:
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("Recommendations", styles["section_title"]))
        story.append(Spacer(1, 2*mm))
        for rec in profile.recommendations[:6]:
            story.append(Paragraph("  " + rec, styles["bullet"]))
            story.append(Spacer(1, 1*mm))


def _metrics_page(story, styles, theme, df, profile):
    T = theme
    story.append(Paragraph("Key Metrics Overview", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))

    num_cols = profile.numeric_cols[:6]
    if not num_cols:
        story.append(Paragraph("No numeric columns found.", styles["body"]))
        return

    cards = []
    for col in num_cols:
        s = df[col].dropna()
        cards.append((
            col,
            "{:,.0f}".format(s.sum()),
            "avg {:,.2f}  |  max {:,.2f}".format(s.mean(), s.max())
        ))

    for i in range(0, len(cards), 3):
        row_cards = cards[i:i+3]
        cells = []
        for label, val, sub in row_cards:
            cells.append(Table(
                [[Paragraph(label[:14], styles["kpi_label"])],
                 [Paragraph(val,        styles["kpi_value"])],
                 [Paragraph(sub,        styles["kpi_sub"])]],
                colWidths=[56*mm],
                style=TableStyle([
                    ("LINEABOVE",     (0, 0), (-1, 0), 3, _rgb(T.accent)),
                    ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.accent_light)),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            ))
        while len(cells) < 3:
            cells.append(Spacer(56*mm, 1))
        row_tbl = Table([cells], colWidths=[58*mm, 58*mm, 58*mm])
        row_tbl.setStyle(TableStyle([
            ("LEFTPADDING",  (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(row_tbl)
        story.append(Spacer(1, 3*mm))


def _chart_page(story, styles, theme, img_bytes, title, narrative, num):
    T = theme
    story.append(Paragraph("Chart " + str(num) + ": " + title, styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))

    try:
        buf = io.BytesIO(img_bytes)
        img = Image(buf, width=170*mm, height=90*mm)
        img.hAlign = "CENTER"
        story.append(img)
    except Exception:
        story.append(Paragraph("[Chart image unavailable]", styles["muted"]))

    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("Analysis", styles["section_title"]))
    story.append(HRFlowable(width="30%", thickness=0.5, color=_rgb(T.text_muted)))
    story.append(Spacer(1, 2*mm))

    box = Table(
        [[Paragraph(narrative or "Chart analysis not available.", styles["body"])]],
        colWidths=[170*mm]
    )
    box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.accent_light)),
        ("LINEBEFORE",    (0, 0), (0,  -1), 3, _rgb(T.accent)),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(box)


def _anomaly_page(story, styles, theme, df, profile):
    T = theme
    story.append(Paragraph("Anomaly Detection", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "Outlier detection uses the IQR method (1.5x interquartile range). "
        "Values outside the normal range are flagged for review.",
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))

    found_any = False
    for col in profile.numeric_cols[:5]:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        lo     = q1 - 1.5*iqr
        hi     = q3 + 1.5*iqr
        out    = df[(df[col] < lo) | (df[col] > hi)]
        if out.empty:
            continue

        found_any = True
        pct = len(out) / max(len(df), 1) * 100

        hdr = Table([[
            Paragraph(col, styles["section_title"]),
            Paragraph(
                "{:,} outliers  ({:.1f}%)".format(len(out), pct),
                ParagraphStyle("ah", fontSize=10,
                               textColor=_rgb(T.negative_color),
                               fontName="Helvetica-Bold",
                               alignment=TA_RIGHT)
            ),
        ]], colWidths=[100*mm, 70*mm])
        hdr.setStyle(TableStyle([
            ("LINEBELOW",     (0, 0), (-1, -1), 0.5, _rgb(T.table_border)),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
        ]))
        story.append(hdr)
        story.append(Paragraph(
            "Normal range: {:,.2f} to {:,.2f}. "
            "Values outside this range may require investigation.".format(lo, hi),
            styles["body_small"]
        ))
        story.append(Spacer(1, 2*mm))

        show_cols = [col] + profile.categorical_cols[:2]
        show_cols = [c for c in show_cols if c in df.columns]
        tbl_data  = [show_cols]
        for _, row in out[show_cols].head(5).iterrows():
            tbl_data.append([str(v) for v in row])
        _render_table(story, theme, tbl_data)
        story.append(Spacer(1, 4*mm))

    if not found_any:
        ok = Table(
            [[Paragraph("No significant outliers detected.", styles["body"])]],
            colWidths=[170*mm]
        )
        ok.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _rgb(T.accent_light)),
            ("LINEBEFORE",    (0, 0), (0,  -1), 3, _rgb(T.positive_color)),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(ok)


def _recommendations_page(story, styles, theme, recs):
    T = theme
    story.append(Paragraph("Recommendations", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_rgb(T.accent)))
    story.append(Spacer(1, 4*mm))

    sections = [
        ("Immediate Actions", "immediate",  T.negative_color),
        ("Short Term",        "short_term", T.neutral_color),
        ("Long Term",         "long_term",  T.positive_color),
    ]
    for sec_title, key, colour in sections:
        items = recs.get(key, [])
        if not items:
            continue
        story.append(Paragraph(sec_title, styles["section_title"]))
        story.append(Spacer(1, 1*mm))
        for item in items:
            row = Table([[
                Paragraph("[ ]", ParagraphStyle(
                    "cb", fontSize=9,
                    textColor=_rgb(colour),
                    fontName="Helvetica-Bold"
                )),
                Paragraph(item, styles["bullet"]),
            ]], colWidths=[8*mm, 162*mm])
            row.setStyle(TableStyle([
                ("LINEBELOW",     (0, 0), (-1, -1), 0.3, _rgb(T.table_border)),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(row)
            story.append(Spacer(1, 1*mm))
        story.append(Spacer(1, 4*mm))


def _appendix_page(story, styles, theme, config, profile):
    T = theme
    story.append(Paragraph("Appendix", styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_rgb(T.text_muted)))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("A. Methodology", styles["section_title"]))
    story.append(Paragraph(
        "Data quality scoring uses a weighted composite formula: "
        "60% completeness score, 30% deduplication score, and 10% "
        "column health score. Outlier detection uses the IQR method "
        "with a 1.5x multiplier. AI narratives generated using "
        "Groq Llama 3.3 70B via the DataForge AI platform.",
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("B. Quality Score Formula", styles["section_title"]))
    formula_data = [
        ["Component",     "Weight", "Description"],
        ["Completeness",  "60%",    "Percentage of non-missing cells"],
        ["Deduplication", "30%",    "Percentage of unique rows"],
        ["Column Health", "10%",    "Average per-column quality score"],
    ]
    _render_table(story, theme, formula_data,
                  col_widths=[50*mm, 25*mm, 95*mm])
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("C. Disclaimer", styles["section_title"]))
    story.append(Paragraph(
        "This report was automatically generated by DataForge AI on "
        + datetime.now().strftime("%B %d, %Y")
        + " for " + config.client_name + ". "
        "All findings are based solely on the provided dataset and should "
        "be verified by a qualified data analyst before making business decisions.",
        styles["body"]
    ))


def _render_table(story, theme, data, col_widths=None):
    T   = theme
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0),  (-1, 0),  _rgb(T.table_header_bg)),
        ("TEXTCOLOR",      (0, 0),  (-1, 0),  _rgb(T.table_header_text)),
        ("FONTNAME",       (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0),  (-1, 0),  8),
        ("TOPPADDING",     (0, 0),  (-1, 0),  5),
        ("BOTTOMPADDING",  (0, 0),  (-1, 0),  5),
        ("LEFTPADDING",    (0, 0),  (-1, 0),  6),
        ("FONTNAME",       (0, 1),  (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1),  (-1, -1), 8),
        ("TEXTCOLOR",      (0, 1),  (-1, -1), _rgb(T.text_dark)),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -1),
         [_rgb(T.table_row_main), _rgb(T.table_row_alt)]),
        ("TOPPADDING",     (0, 1),  (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 1),  (-1, -1), 4),
        ("LEFTPADDING",    (0, 0),  (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0),  (-1, -1), 6),
        ("GRID",           (0, 0),  (-1, -1), 0.3, _rgb(T.table_border)),
    ]))
    story.append(tbl)


def _build_styles(theme):
    T = theme

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": s("cover_title",
            fontSize=26, fontName="Helvetica-Bold",
            textColor=_rgb(T.text_dark),
            alignment=TA_LEFT, spaceAfter=4),
        "cover_label": s("cover_label",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=_rgb(T.text_muted)),
        "cover_value": s("cover_value",
            fontSize=11, fontName="Helvetica",
            textColor=_rgb(T.text_dark)),
        "cover_footer": s("cover_footer",
            fontSize=8, textColor=_rgb(T.text_muted),
            alignment=TA_CENTER),
        "confidential": s("confidential",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER),
        "page_title": s("page_title",
            fontSize=18, fontName="Helvetica-Bold",
            textColor=_rgb(T.text_dark),
            spaceBefore=2, spaceAfter=3),
        "section_title": s("section_title",
            fontSize=11, fontName="Helvetica-Bold",
            textColor=_rgb(T.accent),
            spaceBefore=4, spaceAfter=2),
        "body": s("body",
            fontSize=9, fontName="Helvetica",
            textColor=_rgb(T.text_dark),
            leading=15, alignment=TA_JUSTIFY),
        "body_small": s("body_small",
            fontSize=8, fontName="Helvetica",
            textColor=_rgb(T.text_muted), leading=13),
        "bullet": s("bullet",
            fontSize=9, fontName="Helvetica",
            textColor=_rgb(T.text_dark),
            leftIndent=4, leading=14),
        "muted": s("muted",
            fontSize=8, textColor=_rgb(T.text_muted)),
        "toc_num": s("toc_num",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=_rgb(T.accent)),
        "toc_item": s("toc_item",
            fontSize=9, textColor=_rgb(T.text_dark)),
        "toc_page": s("toc_page",
            fontSize=9, textColor=_rgb(T.text_muted),
            alignment=TA_RIGHT),
        "finding_marker": s("finding_marker",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER),
        "finding_text": s("finding_text",
            fontSize=9, textColor=_rgb(T.text_dark), leading=14),
        "kpi_label": s("kpi_label",
            fontSize=7, fontName="Helvetica-Bold",
            textColor=_rgb(T.text_muted)),
        "kpi_value": s("kpi_value",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=_rgb(T.text_dark)),
        "kpi_sub": s("kpi_sub",
            fontSize=7, textColor=_rgb(T.text_muted)),
    }
