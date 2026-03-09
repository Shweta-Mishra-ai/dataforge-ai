import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from core.data_profiler import DatasetProfile
from core.report_engine import ReportConfig, ReportTheme
import pandas as pd


def _rgb(t):
    return colors.Color(t[0]/255, t[1]/255, t[2]/255)


def build_pdf(
    df, profile, config,
    chart_data, chart_narratives,
    executive_summary, recommendations,
):
    buf   = io.BytesIO()
    theme = config.theme
    W, H  = A4
    M     = 18 * mm   # margin

    # ── Header / Footer ─────────────────────────────────
    def _hf(canvas, doc):
        T = theme
        canvas.saveState()
        # Header bar
        canvas.setFillColor(_rgb(T.header_bg))
        canvas.rect(0, H - 12*mm, W, 12*mm, fill=1, stroke=0)
        canvas.setFillColor(_rgb(T.header_text))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(M, H - 7.5*mm, "DataForge AI")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - M, H - 7.5*mm, config.title[:55])
        # Footer
        canvas.setStrokeColor(_rgb(T.accent))
        canvas.setLineWidth(0.4)
        canvas.line(M, 11*mm, W - M, 11*mm)
        canvas.setFillColor(_rgb(T.text_muted))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(M, 7*mm, "Prepared for: " + config.client_name)
        if config.confidential:
            canvas.setFillColor(_rgb(T.negative_color))
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawCentredString(W/2, 7*mm, "CONFIDENTIAL")
        canvas.setFillColor(_rgb(T.text_muted))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(W - M, 7*mm, "Page " + str(doc.page))
        canvas.restoreState()

    CW = W - 2*M  # content width = 174mm

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

    ST = _styles(theme, CW)
    story = []

    _cover(story, ST, config, theme, CW)
    story.append(PageBreak())
    _toc(story, ST, theme, chart_data, CW)
    story.append(PageBreak())
    _exec_summary(story, ST, theme, executive_summary, CW)
    story.append(PageBreak())
    _dataset_overview(story, ST, theme, df, profile, CW)
    story.append(PageBreak())
    _quality_page(story, ST, theme, profile, CW)
    story.append(PageBreak())
    _metrics_page(story, ST, theme, df, profile, CW)
    story.append(PageBreak())

    for i, (title, img_bytes) in enumerate(chart_data):
        narr = chart_narratives[i] if i < len(chart_narratives) else ""
        _chart_page(story, ST, theme, img_bytes, title, narr, i+1, CW)
        story.append(PageBreak())

    _anomaly_page(story, ST, theme, df, profile, CW)
    story.append(PageBreak())
    _recs_page(story, ST, theme, recommendations, CW)
    story.append(PageBreak())
    _appendix_page(story, ST, theme, config, CW)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  PAGE BUILDERS
# ══════════════════════════════════════════════════════════

def _cover(story, ST, config, theme, CW):
    T = theme
    story.append(Spacer(1, 14*mm))
    story.append(HRFlowable(width="100%", thickness=5,
                            color=_rgb(T.accent), spaceAfter=3*mm))
    story.append(Paragraph(config.title, ST["cover_title"]))
    story.append(Spacer(1, 2*mm))
    story.append(HRFlowable(width="30%", thickness=1,
                            color=_rgb(T.text_muted), spaceAfter=6*mm))

    info = [
        ["Prepared for",  config.client_name],
        ["Prepared by",   config.prepared_by],
        ["Date",          datetime.now().strftime("%B %d, %Y")],
        ["Report Theme",  config.theme_name],
    ]
    rows = [[Paragraph(k, ST["cover_label"]),
             Paragraph(v, ST["cover_value"])] for k, v in info]
    t = Table(rows, colWidths=[40*mm, CW - 40*mm])
    t.setStyle(TableStyle([
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, _rgb(T.table_border)),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 8*mm))

    if config.confidential:
        badge = Table(
            [[Paragraph("CONFIDENTIAL DOCUMENT", ST["conf_badge"])]],
            colWidths=[90*mm]
        )
        badge.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.negative_color)),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
            ("RIGHTPADDING",  (0,0),(-1,-1), 14),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ]))
        story.append(badge)
        story.append(Spacer(1, 4*mm))

    story.append(Spacer(1, 28*mm))
    story.append(HRFlowable(width="100%", thickness=0.4,
                            color=_rgb(T.text_muted)))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Automatically generated by DataForge AI",
        ST["cover_footer"]
    ))


def _toc(story, ST, theme, chart_data, CW):
    T = theme
    _page_h(story, ST, theme, "Table of Contents")
    story.append(Spacer(1, 3*mm))

    entries = [
        ("1", "Executive Summary",   "3"),
        ("2", "Dataset Overview",    "4"),
        ("3", "Data Quality Report", "5"),
        ("4", "Key Metrics",         "6"),
    ]
    pg = 7
    for i, (title, _) in enumerate(chart_data):
        entries.append((str(4+i+1), "Chart {}: {}".format(i+1, title), str(pg)))
        pg += 1
    entries += [
        (str(pg),   "Anomaly Detection", str(pg)),
        (str(pg+1), "Recommendations",   str(pg+1)),
        (str(pg+2), "Appendix",          str(pg+2)),
    ]
    for num, title, page in entries:
        r = Table([[
            Paragraph(num,   ST["toc_num"]),
            Paragraph(title, ST["toc_item"]),
            Paragraph(page,  ST["toc_page"]),
        ]], colWidths=[10*mm, CW-26*mm, 16*mm])
        r.setStyle(TableStyle([
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, _rgb(T.table_border)),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(r)


def _exec_summary(story, ST, theme, findings, CW):
    T = theme
    _page_h(story, ST, theme, "Executive Summary")
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Key findings identified from the dataset analysis:",
        ST["body"]
    ))
    story.append(Spacer(1, 4*mm))

    for i, item in enumerate(findings):
        finding = item.get("finding", str(item)) if isinstance(item, dict) else str(item)
        ftype   = item.get("type", "neutral") if isinstance(item, dict) else "neutral"
        if ftype == "positive":
            mc, mk = T.positive_color, "+"
        elif ftype == "negative":
            mc, mk = T.negative_color, "!"
        else:
            mc, mk = T.accent, str(i+1)

        badge = Table(
            [[Paragraph(mk, ST["find_badge"])]],
            colWidths=[7*mm],
            style=TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), _rgb(mc)),
                ("TOPPADDING",    (0,0),(-1,-1), 5),
                ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                ("LEFTPADDING",   (0,0),(-1,-1), 1),
                ("RIGHTPADDING",  (0,0),(-1,-1), 1),
            ])
        )
        row = Table(
            [[badge, Paragraph(finding, ST["find_text"])]],
            colWidths=[10*mm, CW - 10*mm]
        )
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
            ("LINEBELOW",     (0,0),(-1,-1), 0.4, _rgb(T.table_border)),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(KeepTogether(row))
        story.append(Spacer(1, 2*mm))


def _dataset_overview(story, ST, theme, df, profile, CW):
    T = theme
    _page_h(story, ST, theme, "Dataset Overview")
    story.append(Spacer(1, 4*mm))

    # ── 4 KPI cards — fixed spacing ──────────────────────
    score_c = T.positive_color if profile.overall_quality_score >= 80 \
              else (T.negative_color if profile.overall_quality_score < 60
                    else T.neutral_color)
    miss_c  = T.negative_color if profile.missing_pct > 10 else T.positive_color

    def _kpi_card(label, big, small, accent_c):
        """
        3-row card: label (small caps) / big number / sub text.
        Fixed internal padding — no double-spacing.
        """
        cw = (CW / 4) - 3*mm
        return Table([
            [Paragraph(label, ST["kpi_label"])],
            [Paragraph(big,   ST["kpi_big"])],
            [Paragraph(small, ST["kpi_small"])],
        ], colWidths=[cw], style=TableStyle([
            ("LINEABOVE",     (0,0),(-1,0), 3, _rgb(accent_c)),
            ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
            ("LEFTPADDING",   (0,0),(-1,-1), 7),
            ("RIGHTPADDING",  (0,0),(-1,-1), 5),
            # Row 0 — label
            ("TOPPADDING",    (0,0),(0,0), 7),
            ("BOTTOMPADDING", (0,0),(0,0), 2),
            # Row 1 — big number
            ("TOPPADDING",    (0,1),(0,1), 2),
            ("BOTTOMPADDING", (0,1),(0,1), 2),
            # Row 2 — sub text
            ("TOPPADDING",    (0,2),(0,2), 1),
            ("BOTTOMPADDING", (0,2),(0,2), 7),
        ]))

    kpi_row = Table([[
        _kpi_card("TOTAL ROWS",
                  "{:,}".format(profile.rows),
                  "records", T.accent),
        _kpi_card("COLUMNS",
                  str(profile.cols),
                  "features", T.accent),
        _kpi_card("MISSING",
                  "{}%".format(profile.missing_pct),
                  "{:,} cells".format(profile.missing_cells), miss_c),
        _kpi_card("QUALITY SCORE",
                  "{}/100".format(profile.overall_quality_score),
                  "Good" if profile.overall_quality_score >= 80 else "Fair",
                  score_c),
    ]], colWidths=[(CW/4)]*4)
    kpi_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 2),
        ("RIGHTPADDING", (0,0),(-1,-1), 2),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    story.append(kpi_row)
    story.append(Spacer(1, 6*mm))

    # ── Column type breakdown ─────────────────────────────
    _sec_h(story, ST, theme, "Column Type Breakdown")
    story.append(Spacer(1, 2*mm))
    nc = ", ".join(profile.numeric_cols[:6]) + ("..." if len(profile.numeric_cols) > 6 else "")
    cc = ", ".join(profile.categorical_cols) or "-"
    dc = ", ".join(profile.datetime_cols) or "-"
    _table(story, theme, [
        ["Type",        "Count",                            "Columns"],
        ["Numeric",     str(len(profile.numeric_cols)),     nc],
        ["Categorical", str(len(profile.categorical_cols)), cc],
        ["DateTime",    str(len(profile.datetime_cols)),    dc],
    ], [32*mm, 20*mm, CW - 52*mm])
    story.append(Spacer(1, 6*mm))

    # ── Descriptive stats — KeepTogether prevents orphan ─
    if profile.numeric_cols:
        cols  = profile.numeric_cols[:6]
        desc  = df[cols].describe().round(2)
        hdr   = ["Stat"] + [c[:9] for c in cols]
        rows_ = [hdr]
        for idx in ["mean","std","min","25%","50%","75%","max"]:
            if idx in desc.index:
                rows_.append([idx] + [str(desc.loc[idx, c]) for c in cols])
        cw = [18*mm] + [round((CW-18*mm)/len(cols), 1)*mm] * len(cols)

        block = []
        _sec_h(block, ST, theme, "Descriptive Statistics")
        block.append(Spacer(1, 2*mm))
        _table(block, theme, rows_, cw)
        story.append(KeepTogether(block))


def _quality_page(story, ST, theme, profile, CW):
    T = theme
    _page_h(story, ST, theme, "Data Quality Report")
    story.append(Spacer(1, 3*mm))

    # Score banner
    score   = profile.overall_quality_score
    score_c = T.positive_color if score >= 80 \
              else (T.negative_color if score < 60 else T.neutral_color)
    label   = "Good" if score >= 80 else ("Fair" if score >= 60 else "Poor")

    banner = Table([[
        Paragraph("Overall Quality Score", ST["sec_title"]),
        Paragraph("{}/100  —  {}".format(score, label),
                  ParagraphStyle("qs", fontSize=12,
                                 textColor=_rgb(score_c),
                                 fontName="Helvetica-Bold",
                                 alignment=TA_RIGHT)),
    ]], colWidths=[90*mm, CW - 90*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
        ("LINEABOVE",     (0,0),(-1,0),  3, _rgb(score_c)),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 5*mm))

    _sec_h(story, ST, theme, "Column Analysis")
    story.append(Spacer(1, 2*mm))
    rows_ = [["Column","Type","Missing %","Unique","Outliers","Score"]]
    for p in profile.column_profiles:
        rows_.append([
            p.name[:16], p.dtype[:8],
            "{}%".format(p.missing_pct),
            str(p.unique_count),
            str(p.outlier_count) if p.has_outliers else "-",
            "{}/100".format(p.quality_score),
        ])
    _table(story, theme, rows_,
           [46*mm, 22*mm, 24*mm, 22*mm, 22*mm, CW - 136*mm])

    if profile.recommendations:
        story.append(Spacer(1, 5*mm))
        _sec_h(story, ST, theme, "Recommendations")
        story.append(Spacer(1, 2*mm))
        for rec in profile.recommendations[:8]:
            r = Table([[
                Paragraph("-", ST["bullet_dash"]),
                Paragraph(rec, ST["bullet"]),
            ]], colWidths=[5*mm, CW - 5*mm])
            r.setStyle(TableStyle([
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 0),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            story.append(r)


def _metrics_page(story, ST, theme, df, profile, CW):
    T = theme
    _page_h(story, ST, theme, "Key Metrics Overview")
    story.append(Spacer(1, 4*mm))

    num_cols = profile.numeric_cols[:6]
    if not num_cols:
        story.append(Paragraph("No numeric columns found.", ST["body"]))
        return

    CARD_W = (CW - 4*mm) / 3

    cards = []
    for col in num_cols:
        s = df[col].dropna()
        cards.append((col, "{:,.0f}".format(s.sum()),
                      "avg {:,.2f}   |   max {:,.2f}".format(s.mean(), s.max())))

    for i in range(0, len(cards), 3):
        chunk = cards[i:i+3]
        cells = []
        for col, val, sub in chunk:
            cells.append(Table([
                [Paragraph(col[:16],  ST["kpi_label"])],
                [Paragraph(val,       ST["kpi_big"])],
                [Paragraph(sub,       ST["kpi_small"])],
            ], colWidths=[CARD_W], style=TableStyle([
                ("LINEABOVE",     (0,0),(-1,0), 3, _rgb(T.accent)),
                ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
                ("LEFTPADDING",   (0,0),(-1,-1), 7),
                ("RIGHTPADDING",  (0,0),(-1,-1), 5),
                ("TOPPADDING",    (0,0),(0,0), 7),
                ("BOTTOMPADDING", (0,0),(0,0), 2),
                ("TOPPADDING",    (0,1),(0,1), 2),
                ("BOTTOMPADDING", (0,1),(0,1), 2),
                ("TOPPADDING",    (0,2),(0,2), 1),
                ("BOTTOMPADDING", (0,2),(0,2), 7),
            ])))
        while len(cells) < 3:
            cells.append(Spacer(CARD_W, 1))

        row_t = Table([cells], colWidths=[CARD_W+2*mm]*3)
        row_t.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0),(-1,-1), 1),
            ("RIGHTPADDING", (0,0),(-1,-1), 1),
            ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ]))
        story.append(row_t)
        story.append(Spacer(1, 4*mm))


def _chart_page(story, ST, theme, img_bytes, title, narrative, num, CW):
    T = theme
    _page_h(story, ST, theme, "Chart {}: {}".format(num, title))

    blk = []
    try:
        img = Image(io.BytesIO(img_bytes), width=CW, height=CW * 0.50)
        img.hAlign = "CENTER"
        blk.append(img)
    except Exception:
        blk.append(Paragraph("[Chart image unavailable]", ST["muted"]))

    blk.append(Spacer(1, 4*mm))
    blk.append(Paragraph("Analysis", ST["sec_title"]))
    blk.append(Spacer(1, 2*mm))
    box = Table(
        [[Paragraph(narrative or "Chart analysis not available.", ST["body_j"])]],
        colWidths=[CW]
    )
    box.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
        ("LINEBEFORE",    (0,0),(0,-1),  3, _rgb(T.accent)),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ]))
    blk.append(box)
    story.append(KeepTogether(blk))


def _anomaly_page(story, ST, theme, df, profile, CW):
    T = theme
    _page_h(story, ST, theme, "Anomaly Detection")
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Outlier detection uses the IQR method (1.5x interquartile range). "
        "Values outside the normal range are flagged for review.",
        ST["body"]
    ))
    story.append(Spacer(1, 4*mm))

    found = False
    for col in profile.numeric_cols[:5]:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        out = df[(df[col] < lo) | (df[col] > hi)]
        if out.empty:
            continue
        found = True
        pct = len(out) / max(len(df), 1) * 100

        hdr = Table([[
            Paragraph(col, ST["sec_title"]),
            Paragraph("{:,} outliers  ({:.1f}%)".format(len(out), pct),
                      ParagraphStyle("ah", fontSize=10,
                                     textColor=_rgb(T.negative_color),
                                     fontName="Helvetica-Bold",
                                     alignment=TA_RIGHT)),
        ]], colWidths=[90*mm, CW - 90*mm])
        hdr.setStyle(TableStyle([
            ("LINEBELOW",     (0,0),(-1,-1), 0.5, _rgb(T.table_border)),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("VALIGN",        (0,0),(-1,-1), "BOTTOM"),
        ]))
        blk = [hdr,
               Spacer(1, 1*mm),
               Paragraph("Normal range: {:,.2f} to {:,.2f}".format(lo, hi), ST["muted"]),
               Spacer(1, 2*mm)]

        show = [col] + profile.categorical_cols[:2]
        show = [c for c in show if c in df.columns]
        td = [show]
        for _, row in out[show].head(5).iterrows():
            td.append([str(v) for v in row])
        _table(blk, theme, td)
        blk.append(Spacer(1, 5*mm))
        story.append(KeepTogether(blk))

    if not found:
        ok = Table(
            [[Paragraph("No significant outliers detected.", ST["body"])]],
            colWidths=[CW]
        )
        ok.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), _rgb(T.accent_light)),
            ("LINEBEFORE",    (0,0),(0,-1),  3, _rgb(T.positive_color)),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ]))
        story.append(ok)


def _recs_page(story, ST, theme, recs, CW):
    T = theme
    _page_h(story, ST, theme, "Recommendations")
    story.append(Spacer(1, 3*mm))

    sections = [
        ("Immediate Actions", "immediate",  T.negative_color),
        ("Short Term",        "short_term", T.neutral_color),
        ("Long Term",         "long_term",  T.positive_color),
    ]
    for sec, key, colour in sections:
        items = recs.get(key, [])
        if not items:
            continue
        blk = [Paragraph(sec, ST["sec_title"]), Spacer(1, 1*mm)]
        for item in items:
            r = Table([[
                Paragraph("[ ]", ParagraphStyle(
                    "cb", fontSize=9, textColor=_rgb(colour),
                    fontName="Helvetica-Bold")),
                Paragraph(item, ST["bullet"]),
            ]], colWidths=[8*mm, CW - 8*mm])
            r.setStyle(TableStyle([
                ("LINEBELOW",     (0,0),(-1,-1), 0.3, _rgb(T.table_border)),
                ("TOPPADDING",    (0,0),(-1,-1), 5),
                ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                ("LEFTPADDING",   (0,0),(-1,-1), 0),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            blk.append(r)
        blk.append(Spacer(1, 5*mm))
        story.append(KeepTogether(blk))


def _appendix_page(story, ST, theme, config, CW):
    _page_h(story, ST, theme, "Appendix")
    story.append(Spacer(1, 3*mm))

    _sec_h(story, ST, theme, "A. Methodology")
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        "Data quality scoring: 60% completeness, 30% deduplication, 10% column health. "
        "Outlier detection uses IQR x1.5 multiplier. "
        "AI narratives generated via Groq Llama 3.3 70B.",
        ST["body"]
    ))
    story.append(Spacer(1, 5*mm))

    _sec_h(story, ST, theme, "B. Quality Score Formula")
    story.append(Spacer(1, 2*mm))
    _table(story, theme, [
        ["Component",     "Weight", "Description"],
        ["Completeness",  "60%",    "Percentage of non-missing cells"],
        ["Deduplication", "30%",    "Percentage of unique rows"],
        ["Column Health", "10%",    "Average per-column quality score"],
    ], [40*mm, 22*mm, CW - 62*mm])
    story.append(Spacer(1, 5*mm))

    _sec_h(story, ST, theme, "C. Disclaimer")
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        "This report was automatically generated by DataForge AI on "
        + datetime.now().strftime("%B %d, %Y")
        + " for " + config.client_name + ". "
        "All findings are based solely on the provided dataset and should be "
        "verified by a qualified data analyst before making business decisions.",
        ST["body"]
    ))


# ══════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════

def _page_h(target, ST, theme, title):
    T = theme
    target.append(Paragraph(title, ST["page_title"]))
    target.append(Spacer(1, 2.5*mm))   # gap between title and underline
    target.append(HRFlowable(width="100%", thickness=2,
                              color=_rgb(T.accent), spaceAfter=0))
    target.append(Spacer(1, 3*mm))     # breathing room below line


def _sec_h(target, ST, theme, title):
    target.append(Paragraph(title, ST["sec_title"]))


def _table(target, theme, data, col_widths=None):
    T = theme
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), _rgb(T.table_header_bg)),
        ("TEXTCOLOR",      (0,0), (-1,0), _rgb(T.table_header_text)),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,0), 8),
        ("TOPPADDING",     (0,0), (-1,0), 6),
        ("BOTTOMPADDING",  (0,0), (-1,0), 6),
        ("LEFTPADDING",    (0,0), (-1,0), 7),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,1), (-1,-1), 8),
        ("TEXTCOLOR",      (0,1), (-1,-1), _rgb(T.text_dark)),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [_rgb(T.table_row_main), _rgb(T.table_row_alt)]),
        ("TOPPADDING",     (0,1), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,1), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 7),
        ("RIGHTPADDING",   (0,0), (-1,-1), 7),
        ("GRID",           (0,0), (-1,-1), 0.3, _rgb(T.table_border)),
    ]))
    target.append(t)


# ══════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════

def _styles(theme, CW):
    T = theme
    def s(n, **k):
        return ParagraphStyle(n, **k)

    return {
        # Cover
        "cover_title":  s("ct",  fontSize=26, fontName="Helvetica-Bold",
                          textColor=_rgb(T.text_dark), leading=30),
        "cover_label":  s("cl",  fontSize=9,  fontName="Helvetica-Bold",
                          textColor=_rgb(T.text_muted), leading=14),
        "cover_value":  s("cv",  fontSize=10, textColor=_rgb(T.text_dark), leading=14),
        "cover_footer": s("cf",  fontSize=8,  textColor=_rgb(T.text_muted),
                          alignment=TA_CENTER),
        "conf_badge":   s("cb2", fontSize=10, fontName="Helvetica-Bold",
                          textColor=colors.white, alignment=TA_CENTER),

        # Structure
        "page_title":   s("pt",  fontSize=20, fontName="Helvetica-Bold",
                          textColor=_rgb(T.text_dark), leading=24,
                          spaceBefore=0, spaceAfter=1),
        "sec_title":    s("st",  fontSize=10, fontName="Helvetica-Bold",
                          textColor=_rgb(T.accent), leading=14,
                          spaceBefore=1, spaceAfter=1),

        # Body
        "body":         s("b",   fontSize=9,  textColor=_rgb(T.text_dark),
                          leading=14),
        "body_j":       s("bj",  fontSize=9,  textColor=_rgb(T.text_dark),
                          leading=15, alignment=TA_JUSTIFY),
        "muted":        s("mu",  fontSize=8,  textColor=_rgb(T.text_muted), leading=12),

        # KPI cards — 3 distinct sizes
        "kpi_label":    s("kl",  fontSize=7,  fontName="Helvetica-Bold",
                          textColor=_rgb(T.text_muted),
                          leading=9,  spaceAfter=0, spaceBefore=0),
        "kpi_big":      s("kb",  fontSize=20, fontName="Helvetica-Bold",
                          textColor=_rgb(T.text_dark),
                          leading=24, spaceAfter=0, spaceBefore=0),
        "kpi_small":    s("ks",  fontSize=7,  textColor=_rgb(T.text_muted),
                          leading=9,  spaceAfter=0, spaceBefore=0),

        # TOC
        "toc_num":      s("tn",  fontSize=9,  fontName="Helvetica-Bold",
                          textColor=_rgb(T.accent), leading=14),
        "toc_item":     s("ti",  fontSize=9,  textColor=_rgb(T.text_dark), leading=14),
        "toc_page":     s("tp",  fontSize=9,  textColor=_rgb(T.text_muted),
                          alignment=TA_RIGHT, leading=14),

        # Findings
        "find_badge":   s("fb",  fontSize=10, fontName="Helvetica-Bold",
                          textColor=colors.white, alignment=TA_CENTER),
        "find_text":    s("ft",  fontSize=9,  textColor=_rgb(T.text_dark),
                          leading=14, leftIndent=4),

        # Bullets
        "bullet":       s("bu",  fontSize=9,  textColor=_rgb(T.text_dark),
                          leading=14),
        "bullet_dash":  s("bd",  fontSize=10, fontName="Helvetica-Bold",
                          textColor=_rgb(T.accent), leading=14),
    }
