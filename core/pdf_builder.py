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
    """
    Cover page using nested table — reliable, no canvas tricks.
    Header/footer suppressed by using a blank first page via NextPageTemplate.
    """
    from reportlab.platypus import NextPageTemplate
    T   = theme
    now = datetime.now().strftime("%B %d, %Y")
    title   = config.get("title", "Data Analysis Report")
    client  = config.get("client_name", "Client")

    # Word-wrap title manually
    title_lines = []
    words = title.split()
    line  = ""
    max_chars = 28  # approx chars per line at font 24
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            if line: title_lines.append(line)
            line = w
    if line: title_lines.append(line)
    title_text = "<br/>".join(title_lines)

    # Build content list for cover cell
    cover_content = [
        Spacer(1, 8*mm),
        Paragraph(title_text, ParagraphStyle(
            "cv_t", fontName="Helvetica-Bold", fontSize=24,
            textColor=_rgb(T["header_text"]), alignment=TA_CENTER,
            leading=28, spaceAfter=4*mm,
        )),
        Paragraph(config.get("subtitle",""), ParagraphStyle(
            "cv_s", fontName="Helvetica", fontSize=12,
            textColor=_rgb(T["accent2"]), alignment=TA_CENTER,
            spaceAfter=10*mm,
        )),
        HRFlowable(width="80%", thickness=1, color=_rgb(T["accent"]),
                   spaceAfter=8*mm),
    ]

    meta_style = ParagraphStyle(
        "cv_m", fontName="Helvetica", fontSize=10,
        textColor=_rgb(T["header_text"]), alignment=TA_CENTER,
        spaceAfter=3*mm,
    )
    meta_label_style = ParagraphStyle(
        "cv_ml", fontName="Helvetica", fontSize=7,
        textColor=_rgb(T["text_muted"]), alignment=TA_CENTER,
        spaceAfter=1*mm,
    )

    for label, value in [
        ("PREPARED FOR", client),
        ("PREPARED BY",  "DataForge AI"),
        ("DATE",         now),
        ("THEME",        config.get("theme_name","Corporate Light")),
    ]:
        cover_content.append(Paragraph(label, meta_label_style))
        cover_content.append(Paragraph(value, meta_style))
        cover_content.append(Spacer(1, 2*mm))

    if config.get("confidential", True):
        cover_content.append(Spacer(1, 6*mm))
        cover_content.append(Paragraph("CONFIDENTIAL DOCUMENT", ParagraphStyle(
            "cv_conf", fontName="Helvetica-Bold", fontSize=9,
            textColor=_rgb(T["negative"]), alignment=TA_CENTER,
        )))
        cover_content.append(Paragraph(
            "Automatically generated by DataForge AI",
            ParagraphStyle("cv_gen", fontName="Helvetica", fontSize=8,
                           textColor=_rgb(T["header_text"]), alignment=TA_CENTER)
        ))

    tbl = Table([[cover_content]], colWidths=[CW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _rgb(T["header_bg"])),
        ("TOPPADDING",    (0,0), (-1,-1), 16*mm),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8*mm),
        ("LEFTPADDING",   (0,0), (-1,-1), 12*mm),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12*mm),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
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



def _top_insights_page(story, ST, theme, insights, CW):
    """
    Top Insights — one clear card per insight.
    Format: Title | Problem | Cause | Evidence | Action | Impact
    """
    if not insights:
        return
    _page_h(story, ST, theme, "Top Insights — Key Findings at a Glance")
    story.append(Paragraph(
        "Each insight follows: PROBLEM → CAUSE → EVIDENCE → ACTION → IMPACT",
        ST["caption"]))
    story.append(Spacer(1, 3*mm))

    T = theme
    sev_colors = {
        "critical": T["negative"],
        "warning":  T["warning"],
        "positive": T["positive"],
        "info":     T["accent"],
    }

    for i, ins in enumerate(insights[:6], 1):
        color = sev_colors.get(ins.severity, T["accent"])

        # Severity badge + title
        badge_style = ParagraphStyle(
            "badge_{}".format(i),
            fontName="Helvetica-Bold", fontSize=8,
            textColor=_rgb(color), spaceAfter=1*mm,
        )
        title_style = ParagraphStyle(
            "ins_title_{}".format(i),
            fontName="Helvetica-Bold", fontSize=10,
            textColor=_rgb(T["text"]), spaceAfter=2*mm,
        )
        row_label = ParagraphStyle(
            "row_label_{}".format(i),
            fontName="Helvetica-Bold", fontSize=8,
            textColor=_rgb(color),
        )
        row_val = ParagraphStyle(
            "row_val_{}".format(i),
            fontName="Helvetica", fontSize=8,
            textColor=_rgb(T["text"]), leading=11,
        )

        card_rows = [
            [Paragraph("[{}]  {}. {}".format(
                ins.severity.upper(), i, ins.title), title_style)],
        ]
        for label, value in [
            ("PROBLEM",  ins.problem),
            ("CAUSE",    ins.cause),
            ("EVIDENCE", ins.evidence),
            ("ACTION",   ins.action),
            ("IMPACT",   ins.impact),
        ]:
            if value and value != "N/A":
                card_rows.append([Table(
                    [[Paragraph(label, row_label),
                      Paragraph(value, row_val)]],
                    colWidths=[CW*0.13, CW*0.87]
                )])

        card = Table([[row] for row in card_rows], colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _rgb(T["bg_card"])),
            ("LINEBEFORE",    (0,0), (0,-1),  3, _rgb(color)),
            ("BOX",           (0,0), (-1,-1), 0.5, _rgb(T["border"])),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(card)
        story.append(Spacer(1, 3*mm))


def _attrition_page(story, ST, theme, attrition, CW):
    """
    Deep attrition analysis page.
    Rate, drivers, segment breakdown, flight risk, cost.
    """
    if attrition is None:
        return
    _page_h(story, ST, theme, "Attrition Analysis")
    T = theme

    sev_color = {
        "critical": T["negative"],
        "high":     T["warning"],
        "warning":  T["warning"],
        "normal":   T["positive"],
        "low":      T["positive"],
    }.get(attrition.severity, T["accent"])

    # KPI row
    _kpi_row(story, ST, theme, [
        {"label": "ATTRITION RATE", "value": "{:.1f}%".format(attrition.rate),
         "sub": "{:,} of {:,} employees".format(attrition.n_left, attrition.n_total)},
        {"label": "SEVERITY",       "value": attrition.severity.upper(),
         "sub": "Benchmark: 10-15%"},
        {"label": "FLIGHT RISK",    "value": "{:,}".format(attrition.n_flight_risk),
         "sub": "{:.0f}% of remaining staff".format(attrition.flight_risk_pct)},
        {"label": "COST ESTIMATE",  "value": "HIGH" if attrition.n_left > 50 else "MED",
         "sub": "Replacement cost risk"},
    ], CW)

    story.append(Paragraph(attrition.interpretation, ST["body"]))
    story.append(Paragraph(attrition.cost_estimate, ST["risk"]))
    story.append(Spacer(1, 3*mm))

    # Top drivers table
    if attrition.top_drivers:
        story.append(Paragraph("Attrition Drivers (Statistically Significant)", ST["sub_h"]))
        rows = [[
            Paragraph("Factor",    ST["body_bold"]),
            Paragraph("Type",      ST["body_bold"]),
            Paragraph("Impact",    ST["body_bold"]),
            Paragraph("Detail",    ST["body_bold"]),
        ]]
        for d in attrition.top_drivers[:6]:
            rows.append([
                Paragraph(d["factor"], ST["body"]),
                Paragraph(d["type"].title(), ST["body"]),
                Paragraph("{:.0f}% diff".format(d["impact"]), ST["body"]),
                Paragraph(d["detail"][:70], ST["body"]),
            ])
        _table(story, theme, rows, [CW*0.22, CW*0.13, CW*0.15, CW*0.50])

    # Segment breakdown
    two_col_data = []
    if attrition.dept_attrition:
        story.append(Paragraph("Attrition by Department", ST["sub_h"]))
        sorted_dept = sorted(attrition.dept_attrition.items(),
                             key=lambda x: x[1], reverse=True)
        rows = [[Paragraph("Department", ST["body_bold"]),
                 Paragraph("Attrition Rate", ST["body_bold"]),
                 Paragraph("Status", ST["body_bold"])]]
        for dept, rate in sorted_dept:
            status = "CRITICAL" if rate > 25 else "HIGH" if rate > 18 else "OK"
            rows.append([
                Paragraph(str(dept), ST["body"]),
                Paragraph("{:.1f}%".format(rate), ST["body"]),
                Paragraph(status, ST["body"]),
            ])
        _table(story, theme, rows, [CW*0.45, CW*0.30, CW*0.25])

    if attrition.salary_attrition:
        story.append(Paragraph("Attrition by Salary Band", ST["sub_h"]))
        sorted_sal = sorted(attrition.salary_attrition.items(),
                            key=lambda x: x[1], reverse=True)
        rows = [[Paragraph("Salary Band", ST["body_bold"]),
                 Paragraph("Attrition Rate", ST["body_bold"])]]
        for sal, rate in sorted_sal:
            rows.append([Paragraph(str(sal), ST["body"]),
                         Paragraph("{:.1f}%".format(rate), ST["body"])])
        _table(story, theme, rows, [CW*0.5, CW*0.5])



def _build_insight_cards(story, ST, theme, findings, risks, opportunities, CW):
    """
    Structured insight cards — Problem → Cause → Action → Impact.
    Parsed from finding strings or raw insight objects.
    """
    _page_h(story, ST, theme, "Top Insights — Decision Summary")
    story.append(Paragraph(
        "Each card shows: WHAT is the problem, WHY it is happening, "
        "WHAT ACTION to take, and WHAT IMPACT to expect.",
        ST["caption"]
    ))
    story.append(Spacer(1, 3*mm))
    T = theme

    all_items = []
    for r in risks[:4]:
        all_items.append(("critical", r))
    for f in findings[:3]:
        all_items.append(("info", f))
    for o in opportunities[:2]:
        all_items.append(("positive", o))

    sev_colors = {
        "critical": T["negative"],
        "warning":  T["warning"],
        "positive": T["positive"],
        "info":     T["accent"],
    }

    for i, (sev, text) in enumerate(all_items[:7], 1):
        color = sev_colors.get(sev, T["accent"])

        # Parse WHAT/WHY/ACTION/IMPACT from text
        lines = []
        if "| " in text:
            # Structured format from story_engine
            parts = text.split(" | ")
            for part in parts:
                if ":" in part:
                    label, value = part.split(":", 1)
                    lines.append((label.strip(), value.strip()))
                else:
                    lines.append(("NOTE", part.strip()))
        else:
            lines.append(("FINDING", text))

        title_text = lines[0][1] if lines else text[:60]
        detail_lines = lines[1:] if len(lines) > 1 else []

        title_s = ParagraphStyle(
            "ct{}".format(i), fontName="Helvetica-Bold", fontSize=10,
            textColor=_rgb(T["text"]), spaceAfter=1*mm, leading=13,
        )
        lbl_s = ParagraphStyle(
            "cl{}".format(i), fontName="Helvetica-Bold", fontSize=7,
            textColor=_rgb(color),
        )
        val_s = ParagraphStyle(
            "cv{}".format(i), fontName="Helvetica", fontSize=8,
            textColor=_rgb(T["text"]), leading=11,
        )
        badge_s = ParagraphStyle(
            "cb{}".format(i), fontName="Helvetica-Bold", fontSize=7,
            textColor=_rgb(color), spaceAfter=1*mm,
        )

        inner = [
            [Paragraph("[{}] Finding {}".format(sev.upper(), i), badge_s)],
            [Paragraph(title_text[:100], title_s)],
        ]
        for label, value in detail_lines[:5]:
            inner.append([Table(
                [[Paragraph(label[:12], lbl_s),
                  Paragraph(value[:140], val_s)]],
                colWidths=[CW*0.14, CW*0.84],
            )])

        card = Table([[row] for row in inner], colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _rgb(T["bg_card"])),
            ("LINEBEFORE",    (0,0), (0,-1),  3, _rgb(color)),
            ("BOX",           (0,0), (-1,-1), 0.5, _rgb(T["border"])),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 7),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([card, Spacer(1, 3*mm)]))


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
             df.isna().sum().sum() / max(df.shape[0]*df.shape[1], 1) * 100),
         "sub": "{:,} cells missing".format(int(df.isna().sum().sum()))},
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

    # Descriptive stats — wrapped in KeepTogether to prevent overlap
    if num_cols:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("Descriptive Statistics", ST["sub_h"]))
        desc  = df[num_cols[:6]].describe().round(2)
        hdr   = [Paragraph("Stat", ST["body_bold"])] + [
            Paragraph(c[:10], ST["body_bold"]) for c in num_cols[:6]]
        rows  = [hdr]
        for stat in ["mean","std","min","25%","50%","75%","max"]:
            if stat in desc.index:
                row = [Paragraph(stat, ST["body"])] + [
                    Paragraph(str(desc.loc[stat, c]), ST["body"])
                    for c in num_cols[:6]]
                rows.append(row)
        cw_stat = CW / (len(num_cols[:6]) + 1)
        # KeepTogether prevents table from splitting across pages
        from reportlab.platypus import KeepTogether as KT
        stat_elems = []
        tbl_obj = Table(
            rows,
            colWidths=[cw_stat] + [cw_stat]*len(num_cols[:6]),
            repeatRows=1
        )
        T2 = theme
        tbl_obj.setStyle(TableStyle([
            ("FONTNAME",  (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME",  (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",  (0,0), (-1,-1), 8),
            ("TEXTCOLOR", (0,0), (-1,0), _rgb(T2["header_text"])),
            ("BACKGROUND",(0,0), (-1,0), _rgb(T2["header_bg"])),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [_rgb(T2["bg_card"]), _rgb(T2["bg_light"])]),
            ("GRID",      (0,0), (-1,-1), 0.3, _rgb(T2["border"])),
            ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",(0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("LEFTPADDING",(0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ]))
        stat_elems.append(tbl_obj)
        story.append(KT(stat_elems))
        story.append(Spacer(1, 2*mm))


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

    # ── TOC sections — sequential numbering ───────────────
    page_num  = 3
    sec_num   = 1
    toc_sections = []

    def _toc_add(title):
        nonlocal page_num, sec_num
        toc_sections.append((sec_num, title, page_num))
        sec_num  += 1
        page_num += 1

    _toc_add("Executive Summary")
    _toc_add("Top Insights — Decision Summary")
    _toc_add("Dataset Overview")
    if stats_report:
        _toc_add("Statistical Analysis")
    if bi_report:
        _toc_add("Business Intelligence")
    if ml_report:
        _toc_add("ML Predictions")
    for i, (title, _, _) in enumerate(chart_data, 1):
        _toc_add("Chart {}: {}".format(i, title[:35]))
    _toc_add("Recommendations")
    _toc_add("Appendix")

    # ── Build story ───────────────────────────────────────
    _cover(story, ST, theme, config, CW)
    story.append(PageBreak())

    _toc(story, ST, theme, toc_sections, CW)
    story.append(PageBreak())

    _exec_summary(story, ST, theme, executive_summary,
                  findings, risks, opportunities, CW)
    story.append(PageBreak())

    # Top Insights — structured cards
    _build_insight_cards(story, ST, theme, findings, risks, opportunities, CW)
    story.append(PageBreak())

    _dataset_overview(story, ST, theme, df, profile, cleaning_summary, CW)
    story.append(PageBreak())

    if stats_report:
        _statistical_analysis(story, ST, theme, stats_report, CW)
        story.append(PageBreak())

    # Attrition deep dive — from story report
    attrition_data = getattr(
        type('_', (object,), {})(), 'attrition', None)
    # Passed as extra kwarg if available
    if bi_report and hasattr(bi_report, 'attrition'):
        _attrition_page(story, ST, theme, bi_report.attrition, CW)
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
