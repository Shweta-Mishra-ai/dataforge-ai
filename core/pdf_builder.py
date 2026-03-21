"""
pdf_builder.py — Client-grade PDF report builder.
Color coordinated by domain: HR=Blue, Ecommerce=Orange, Sales=Green.
Senior analyst level — charts, insights, Problem→Cause→Action→Impact.
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
#  DOMAIN COLOR THEMES
# ══════════════════════════════════════════════════════════

THEMES = {
    # User-selectable themes
    "Corporate Light": {
        "header_bg":   "#1a4a8a", "header_text": "#ffffff",
        "accent":      "#2196F3", "accent2":     "#22d3a5",
        "text":        "#1e1e28", "text_muted":  "#646882",
        "bg_light":    "#f0f4ff", "bg_card":     "#f8faff",
        "border":      "#d0d8f0",
        "positive":    "#22d3a5", "negative":    "#f77070",
        "warning":     "#f7934f", "info":        "#2196F3",
        "critical_bg": "#fff0f0", "warning_bg":  "#fff8f0",
        "positive_bg": "#f0fff8", "info_bg":     "#f0f8ff",
    },
    "HR Blue": {
        "header_bg":   "#1a3a6b", "header_text": "#ffffff",
        "accent":      "#1976D2", "accent2":     "#42A5F5",
        "text":        "#1a2035", "text_muted":  "#5a6482",
        "bg_light":    "#e8f0fe", "bg_card":     "#f5f8ff",
        "border":      "#c5d3f0",
        "positive":    "#2e7d32", "negative":    "#c62828",
        "warning":     "#e65100", "info":        "#1565C0",
        "critical_bg": "#ffebee", "warning_bg":  "#fff3e0",
        "positive_bg": "#e8f5e9", "info_bg":     "#e3f2fd",
    },
    "Ecommerce Orange": {
        "header_bg":   "#bf360c", "header_text": "#ffffff",
        "accent":      "#F4511E", "accent2":     "#FF8A65",
        "text":        "#1a1a1a", "text_muted":  "#5a5a5a",
        "bg_light":    "#fbe9e7", "bg_card":     "#fff8f6",
        "border":      "#ffccbc",
        "positive":    "#2e7d32", "negative":    "#b71c1c",
        "warning":     "#e65100", "info":        "#1565C0",
        "critical_bg": "#ffebee", "warning_bg":  "#fff3e0",
        "positive_bg": "#e8f5e9", "info_bg":     "#e8f0fe",
    },
    "Sales Green": {
        "header_bg":   "#1b5e20", "header_text": "#ffffff",
        "accent":      "#2E7D32", "accent2":     "#66BB6A",
        "text":        "#1a2a1a", "text_muted":  "#4a6a4a",
        "bg_light":    "#e8f5e9", "bg_card":     "#f5fbf5",
        "border":      "#c8e6c9",
        "positive":    "#1b5e20", "negative":    "#b71c1c",
        "warning":     "#e65100", "info":        "#1565C0",
        "critical_bg": "#ffebee", "warning_bg":  "#fff3e0",
        "positive_bg": "#e8f5e9", "info_bg":     "#e8f0fe",
    },
    "Dark Tech": {
        "header_bg":   "#0d1117", "header_text": "#58a6ff",
        "accent":      "#58a6ff", "accent2":     "#3fb950",
        "text":        "#e6edf3", "text_muted":  "#8b949e",
        "bg_light":    "#161b22", "bg_card":     "#1c2128",
        "border":      "#30363d",
        "positive":    "#3fb950", "negative":    "#f85149",
        "warning":     "#d29922", "info":        "#58a6ff",
        "critical_bg": "#1c1010", "warning_bg":  "#1c1800",
        "positive_bg": "#0d1a0f", "info_bg":     "#0d1421",
    },
}

# Domain → auto theme mapping
DOMAIN_THEMES = {
    "hr":        "HR Blue",
    "ecommerce": "Ecommerce Orange",
    "sales":     "Sales Green",
    "finance":   "Corporate Light",
    "general":   "Corporate Light",
}


def _rgb(hex_str):
    h = hex_str.lstrip("#")
    return colors.Color(int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255)


def get_theme(theme_name: str, domain: str = "general") -> dict:
    """Get theme — auto-select by domain if not specified."""
    if theme_name in THEMES:
        return THEMES[theme_name]
    auto = DOMAIN_THEMES.get(domain, "Corporate Light")
    return THEMES.get(auto, THEMES["Corporate Light"])


# ══════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════

def _styles(T: dict, CW: float) -> dict:
    txt = _rgb(T["text"])
    mut = _rgb(T["text_muted"])
    acc = _rgb(T["accent"])
    return {
        "section_h": ParagraphStyle("sh", fontName="Helvetica-Bold",
            fontSize=15, textColor=acc, spaceAfter=2*mm, spaceBefore=3*mm),
        "sub_h":     ParagraphStyle("sub", fontName="Helvetica-Bold",
            fontSize=11, textColor=txt, spaceAfter=2*mm, spaceBefore=2*mm),
        "body":      ParagraphStyle("body", fontName="Helvetica",
            fontSize=9, textColor=txt, leading=14, spaceAfter=2*mm,
            alignment=TA_JUSTIFY),
        "body_bold": ParagraphStyle("bb", fontName="Helvetica-Bold",
            fontSize=9, textColor=txt, leading=14, spaceAfter=1*mm),
        "caption":   ParagraphStyle("cap", fontName="Helvetica",
            fontSize=8, textColor=mut, leading=11, spaceAfter=1*mm),
        "toc":       ParagraphStyle("toc", fontName="Helvetica",
            fontSize=10, textColor=txt, leading=16, spaceAfter=1*mm),
        "kpi_label": ParagraphStyle("kl", fontName="Helvetica",
            fontSize=7, textColor=mut, alignment=TA_CENTER),
        "kpi_value": ParagraphStyle("kv", fontName="Helvetica-Bold",
            fontSize=18, textColor=acc, alignment=TA_CENTER),
        "kpi_sub":   ParagraphStyle("ks", fontName="Helvetica",
            fontSize=7, textColor=mut, alignment=TA_CENTER),
        "risk":      ParagraphStyle("risk", fontName="Helvetica",
            fontSize=9, textColor=_rgb(T["negative"]),
            leading=13, leftIndent=4*mm, spaceAfter=1*mm),
        "positive":  ParagraphStyle("pos", fontName="Helvetica",
            fontSize=9, textColor=_rgb(T["positive"]),
            leading=13, leftIndent=4*mm, spaceAfter=1*mm),
        "action":    ParagraphStyle("act", fontName="Helvetica",
            fontSize=9, textColor=txt, leading=13,
            leftIndent=6*mm, spaceAfter=2*mm),
        "narrative": ParagraphStyle("narr", fontName="Helvetica",
            fontSize=9, textColor=txt, leading=14, spaceAfter=2*mm,
            alignment=TA_JUSTIFY, leftIndent=3*mm, rightIndent=3*mm),
        "insight_title": ParagraphStyle("it", fontName="Helvetica-Bold",
            fontSize=10, textColor=txt, spaceAfter=1*mm, leading=13),
        "badge":     ParagraphStyle("badge", fontName="Helvetica-Bold",
            fontSize=7, textColor=acc, spaceAfter=1*mm),
        "label":     ParagraphStyle("lbl", fontName="Helvetica-Bold",
            fontSize=7, textColor=acc),
        "value":     ParagraphStyle("val", fontName="Helvetica",
            fontSize=8, textColor=txt, leading=11),
        "cover_title": ParagraphStyle("ct", fontName="Helvetica-Bold",
            fontSize=22, textColor=_rgb(T["header_text"]),
            alignment=TA_CENTER, spaceAfter=4*mm, leading=26),
        "cover_sub": ParagraphStyle("cs", fontName="Helvetica",
            fontSize=12, textColor=_rgb(T["accent2"]),
            alignment=TA_CENTER, spaceAfter=3*mm),
        "cover_meta":ParagraphStyle("cm", fontName="Helvetica",
            fontSize=9, textColor=_rgb(T["header_text"]),
            alignment=TA_CENTER, spaceAfter=2*mm),
        "meta_label":ParagraphStyle("ml", fontName="Helvetica",
            fontSize=7, textColor=_rgb(T["text_muted"]),
            alignment=TA_CENTER, spaceAfter=0*mm),
    }


# ══════════════════════════════════════════════════════════
#  LAYOUT HELPERS
# ══════════════════════════════════════════════════════════

def _page_h(story, ST, T, title, subtitle=""):
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(title, ST["section_h"]))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=_rgb(T["accent"]),
                             spaceAfter=2*mm, spaceBefore=0))
    if subtitle:
        story.append(Paragraph(subtitle, ST["caption"]))
        story.append(Spacer(1, 1*mm))


def _divider(story, T):
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=_rgb(T["border"]),
                             spaceAfter=2*mm, spaceBefore=2*mm))


def _table(story, T, data, col_widths=None):
    if not data or len(data) < 1: return
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 8),
        ("TEXTCOLOR",      (0,0), (-1,0),  _rgb(T["header_text"])),
        ("BACKGROUND",     (0,0), (-1,0),  _rgb(T["header_bg"])),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [_rgb(T["bg_card"]), _rgb(T["bg_light"])]),
        ("GRID",           (0,0), (-1,-1), 0.3, _rgb(T["border"])),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 3),
        ("LEFTPADDING",    (0,0), (-1,-1), 5),
        ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ("WORDWRAP",       (0,0), (-1,-1), True),
    ]))
    story.append(t)
    story.append(Spacer(1, 2*mm))


def _kpi_row(story, ST, T, kpis, CW):
    n   = len(kpis)
    cw  = CW / n
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
    t = Table([row], colWidths=[cw]*n)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,-1), _rgb(T["bg_card"])),
        ("BOX",            (0,0), (-1,-1), 0.8, _rgb(T["accent"])),
        ("LINEAFTER",      (0,0), (-2,-1), 0.5, _rgb(T["border"])),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))


def _narrative_box(story, ST, T, text):
    if not text: return
    box = Table([[Paragraph(text, ST["narrative"])]], colWidths=["100%"])
    box.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,-1), _rgb(T["bg_light"])),
        ("LINEBEFORE",     (0,0), (0,-1),  4, _rgb(T["accent"])),
        ("TOPPADDING",     (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 8),
        ("LEFTPADDING",    (0,0), (-1,-1), 10),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(box)
    story.append(Spacer(1, 2*mm))


def _insight_card(story, ST, T, ins, CW, num=None):
    """
    Color-coded insight card.
    CRITICAL=red, WARNING=orange, POSITIVE=green, INFO=blue
    """
    sev_colors = {
        "critical": T["negative"],
        "warning":  T["warning"],
        "positive": T["positive"],
        "info":     T["info"],
    }
    sev_bgs = {
        "critical": T["critical_bg"],
        "warning":  T["warning_bg"],
        "positive": T["positive_bg"],
        "info":     T["info_bg"],
    }
    color  = sev_colors.get(ins.severity, T["info"])
    bg     = sev_bgs.get(ins.severity, T["bg_card"])

    badge_s = ParagraphStyle("badge_ins", fontName="Helvetica-Bold",
        fontSize=7, textColor=_rgb(color), spaceAfter=1*mm)
    title_s = ParagraphStyle("title_ins", fontName="Helvetica-Bold",
        fontSize=10, textColor=_rgb(T["text"]), spaceAfter=1*mm, leading=13)
    lbl_s   = ParagraphStyle("lbl_ins", fontName="Helvetica-Bold",
        fontSize=7, textColor=_rgb(color))
    val_s   = ParagraphStyle("val_ins", fontName="Helvetica",
        fontSize=8, textColor=_rgb(T["text"]), leading=11)

    num_str = "{}. ".format(num) if num else ""
    inner   = [
        [Paragraph("[{}]  {}{}".format(ins.severity.upper(), num_str, ins.title), badge_s)],
    ]

    for label, value in [
        ("PROBLEM",  ins.problem),
        ("CAUSE",    ins.cause),
        ("EVIDENCE", ins.evidence),
        ("ACTION",   ins.action),
        ("IMPACT",   ins.impact),
    ]:
        if value and value.strip() and value != "N/A":
            inner.append([Table(
                [[Paragraph(label, lbl_s), Paragraph(value[:200], val_s)]],
                colWidths=[CW*0.12, CW*0.88]
            )])

    card = Table([[row] for row in inner], colWidths=[CW])
    card.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,-1), _rgb(bg)),
        ("LINEBEFORE",     (0,0), (0,-1),  4, _rgb(color)),
        ("BOX",            (0,0), (-1,-1), 0.5, _rgb(T["border"])),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story.append(KeepTogether([card, Spacer(1, 3*mm)]))


# ══════════════════════════════════════════════════════════
#  COVER PAGE
# ══════════════════════════════════════════════════════════

def _cover(story, ST, T, config, CW):
    now   = datetime.now().strftime("%B %d, %Y")
    title = config.get("title","Data Analysis Report")
    domain= config.get("domain","general")

    # Domain label
    domain_labels = {
        "hr":        "HR & People Analytics",
        "ecommerce": "E-Commerce Analytics",
        "sales":     "Sales Performance Analytics",
        "finance":   "Financial Analytics",
        "general":   "Business Analytics",
    }
    domain_label = domain_labels.get(domain, "Business Analytics")

    # Word wrap title
    words = title.split()
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        if len(test) <= 32:
            line = test
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    title_text = "<br/>".join(lines)

    content = [
        Spacer(1, 6*mm),
        Paragraph(domain_label.upper(), ParagraphStyle(
            "dl", fontName="Helvetica-Bold", fontSize=8,
            textColor=_rgb(T["accent2"]), alignment=TA_CENTER,
            spaceAfter=4*mm, charSpace=2)),
        Paragraph(title_text, ST["cover_title"]),
        Paragraph(config.get("subtitle","Powered by DataForge AI"), ST["cover_sub"]),
        Spacer(1, 6*mm),
        Table([[
            Paragraph("PREPARED FOR", ST["meta_label"]),
            Paragraph("PREPARED BY",  ST["meta_label"]),
            Paragraph("DATE",         ST["meta_label"]),
        ],[
            Paragraph(config.get("client_name","Client"), ST["cover_meta"]),
            Paragraph("DataForge AI",                     ST["cover_meta"]),
            Paragraph(now,                                ST["cover_meta"]),
        ]], colWidths=[CW/3]*3),
        Spacer(1, 6*mm),
    ]

    if config.get("confidential", True):
        content.extend([
            Paragraph("CONFIDENTIAL DOCUMENT", ParagraphStyle(
                "conf", fontName="Helvetica-Bold", fontSize=9,
                textColor=_rgb(T["negative"]), alignment=TA_CENTER)),
            Paragraph("Automatically generated by DataForge AI",
                      ST["cover_meta"]),
        ])

    tbl = Table([content], colWidths=[CW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,-1), _rgb(T["header_bg"])),
        ("TOPPADDING",     (0,0), (-1,-1), 14*mm),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 8*mm),
        ("LEFTPADDING",    (0,0), (-1,-1), 10*mm),
        ("RIGHTPADDING",   (0,0), (-1,-1), 10*mm),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
    ]))
    story.append(tbl)


# ══════════════════════════════════════════════════════════
#  TOC
# ══════════════════════════════════════════════════════════

def _toc(story, ST, T, sections, CW):
    _page_h(story, ST, T, "Table of Contents")
    for num, title, page in sections:
        dots = "." * max(1, 58 - len(str(num)+title))
        story.append(Paragraph(
            "{}. {}  {}  {}".format(num, title, dots, page),
            ST["toc"]))
    story.append(Spacer(1, 3*mm))


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════

def _exec_summary(story, ST, T, summary, findings, risks, opps, CW):
    _page_h(story, ST, T, "Executive Summary",
            "Key findings and strategic priorities")

    if summary:
        _narrative_box(story, ST, T, summary)

    if findings:
        story.append(Paragraph("Key Findings", ST["sub_h"]))
        for f in findings[:5]:
            story.append(Paragraph("+ " + f, ST["body"]))
        story.append(Spacer(1, 2*mm))

    if risks:
        story.append(Paragraph("Business Risks", ST["sub_h"]))
        for r in risks[:4]:
            story.append(Paragraph("! " + r, ST["risk"]))
        story.append(Spacer(1, 2*mm))

    if opps:
        story.append(Paragraph("Opportunities", ST["sub_h"]))
        for o in opps[:3]:
            story.append(Paragraph("* " + o, ST["positive"]))


# ══════════════════════════════════════════════════════════
#  TOP INSIGHTS CARDS
# ══════════════════════════════════════════════════════════

def _top_insights(story, ST, T, insights, CW):
    _page_h(story, ST, T, "Top Insights",
            "Each finding: Problem → Cause → Evidence → Action → Impact")

    if not insights:
        story.append(Paragraph("No structured insights available.", ST["body"]))
        return

    for i, ins in enumerate(insights[:6], 1):
        _insight_card(story, ST, T, ins, CW, num=i)


# ══════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════

def _dataset_overview(story, ST, T, df, profile, cleaning_summary, CW):
    _page_h(story, ST, T, "Dataset Overview",
            "Data quality assessment and column breakdown")

    miss_pct = df.isna().sum().sum() / max(df.shape[0]*df.shape[1],1) * 100
    qual_score = getattr(profile, "overall_quality_score", "N/A")
    grade = getattr(profile, "data_quality_grade", "")

    _kpi_row(story, ST, T, [
        {"label":"TOTAL ROWS",   "value":"{:,}".format(len(df)),
         "sub":"records"},
        {"label":"COLUMNS",      "value":str(len(df.columns)),
         "sub":"features"},
        {"label":"MISSING DATA", "value":"{:.1f}%".format(miss_pct),
         "sub":"{:,} cells".format(int(df.isna().sum().sum()))},
        {"label":"QUALITY SCORE","value":"{}".format(qual_score),
         "sub":"Grade {}".format(grade) if grade else "out of 100"},
    ], CW)

    # Column breakdown
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

    story.append(Paragraph("Column Type Breakdown", ST["sub_h"]))
    _table(story, T, [
        [Paragraph("Type",    ST["body_bold"]),
         Paragraph("Count",   ST["body_bold"]),
         Paragraph("Columns (sample)", ST["body_bold"])],
        [Paragraph("Numeric",    ST["body"]),
         Paragraph(str(len(num_cols)), ST["body"]),
         Paragraph(", ".join(num_cols[:6]), ST["body"])],
        [Paragraph("Categorical",ST["body"]),
         Paragraph(str(len(cat_cols)), ST["body"]),
         Paragraph(", ".join(cat_cols[:6]), ST["body"])],
        [Paragraph("DateTime",   ST["body"]),
         Paragraph(str(len(dt_cols)), ST["body"]),
         Paragraph(", ".join(dt_cols[:4]) or "None", ST["body"])],
    ], [CW*0.2, CW*0.12, CW*0.68])

    # Recommendations from profile
    recs = getattr(profile, "recommendations", [])
    if recs:
        story.append(Paragraph("Data Quality Recommendations", ST["sub_h"]))
        for rec in recs[:5]:
            sev = "risk" if rec.startswith("CRITICAL") else "body"
            story.append(Paragraph("• " + rec, ST[sev]))
        story.append(Spacer(1, 2*mm))

    # Descriptive stats
    if num_cols:
        story.append(Paragraph("Descriptive Statistics", ST["sub_h"]))
        story.append(Spacer(1, 1*mm))
        cols_show = num_cols[:5]
        desc  = df[cols_show].describe().round(2)
        hdr   = [Paragraph("Stat", ST["body_bold"])] + [
            Paragraph(c[:10], ST["body_bold"]) for c in cols_show]
        rows  = [hdr]
        for stat in ["mean","std","min","25%","50%","75%","max"]:
            if stat in desc.index:
                row = [Paragraph(stat, ST["body"])] + [
                    Paragraph(str(desc.loc[stat,c]), ST["body"])
                    for c in cols_show]
                rows.append(row)
        cw_s  = CW / (len(cols_show)+1)
        tbl_s = Table(rows, colWidths=[cw_s]*(len(cols_show)+1), repeatRows=1)
        tbl_s.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("TEXTCOLOR",     (0,0),(-1,0),  _rgb(T["header_text"])),
            ("BACKGROUND",    (0,0),(-1,0),  _rgb(T["header_bg"])),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_rgb(T["bg_card"]),_rgb(T["bg_light"])]),
            ("GRID",          (0,0),(-1,-1), 0.3, _rgb(T["border"])),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ]))
        story.append(KeepTogether([tbl_s]))
        story.append(Spacer(1, 2*mm))


# ══════════════════════════════════════════════════════════
#  ATTRITION PAGE (HR only)
# ══════════════════════════════════════════════════════════

def _attrition_page(story, ST, T, attrition, CW):
    if attrition is None: return
    _page_h(story, ST, T, "Attrition Deep Dive",
            "Employee turnover analysis — drivers, segments, cost")

    sev_col = {
        "critical":"negative","high":"warning",
        "warning":"warning","normal":"positive"
    }.get(attrition.severity, "info")

    _kpi_row(story, ST, T, [
        {"label":"ATTRITION RATE","value":"{:.1f}%".format(attrition.rate),
         "sub":"{:,} of {:,} left".format(attrition.n_left,attrition.n_total)},
        {"label":"SEVERITY",      "value":attrition.severity.upper(),
         "sub":"Benchmark: 10-15%"},
        {"label":"FLIGHT RISK",   "value":"{:,}".format(attrition.n_flight_risk),
         "sub":"{:.0f}% of remaining".format(attrition.flight_risk_pct)},
        {"label":"COST RISK",     "value":"HIGH" if attrition.n_left>50 else "MED",
         "sub":"Replacement cost"},
    ], CW)

    story.append(Paragraph(attrition.interpretation, ST["body"]))
    story.append(Paragraph(attrition.cost_estimate, ST["risk"]))
    story.append(Spacer(1, 3*mm))

    # Top drivers
    if attrition.top_drivers:
        story.append(Paragraph("Attrition Drivers", ST["sub_h"]))
        rows = [[
            Paragraph("Factor",  ST["body_bold"]),
            Paragraph("Type",    ST["body_bold"]),
            Paragraph("Impact",  ST["body_bold"]),
            Paragraph("Finding", ST["body_bold"]),
        ]]
        for d in attrition.top_drivers[:6]:
            rows.append([
                Paragraph(d["factor"][:20],            ST["body"]),
                Paragraph(d["type"].title(),           ST["body"]),
                Paragraph("{:.0f}% diff".format(d["impact"]), ST["body"]),
                Paragraph(d["detail"][:70],            ST["body"]),
            ])
        _table(story, T, rows, [CW*0.22,CW*0.13,CW*0.15,CW*0.50])

    # Department breakdown
    if attrition.dept_attrition:
        story.append(Paragraph("Attrition by Department", ST["sub_h"]))
        sorted_d = sorted(attrition.dept_attrition.items(), key=lambda x:x[1], reverse=True)
        rows = [[Paragraph("Department",ST["body_bold"]),
                 Paragraph("Rate",      ST["body_bold"]),
                 Paragraph("Status",    ST["body_bold"])]]
        for dept, rate in sorted_d:
            status = "CRITICAL" if rate>25 else "HIGH" if rate>18 else "NORMAL"
            rows.append([Paragraph(str(dept), ST["body"]),
                         Paragraph("{:.1f}%".format(rate), ST["body"]),
                         Paragraph(status, ST["body"])])
        _table(story, T, rows, [CW*0.5,CW*0.25,CW*0.25])


# ══════════════════════════════════════════════════════════
#  CHART PAGE
# ══════════════════════════════════════════════════════════

def _chart_page(story, ST, T, img_bytes, title, narrative, num, CW):
    _page_h(story, ST, T, "Chart {}: {}".format(num, title))

    if img_bytes:
        try:
            img = Image(io.BytesIO(img_bytes), width=CW, height=CW*0.48)
            story.append(KeepTogether([img, Spacer(1, 3*mm)]))
        except Exception:
            pass

    if narrative:
        story.append(Paragraph("Analysis", ST["sub_h"]))
        _narrative_box(story, ST, T, narrative)


# ══════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════

def _stats_section(story, ST, T, stats_report, CW):
    if stats_report is None: return
    _page_h(story, ST, T, "Statistical Analysis",
            "Distribution, normality, correlations")

    # Column stats
    story.append(Paragraph("Distribution Summary", ST["sub_h"]))
    col_stats = getattr(stats_report, "column_stats", {})
    for col, cs in list(col_stats.items())[:8]:
        if getattr(cs,"mean",None) is None: continue
        normal  = "Normal" if getattr(cs,"is_normal",False) else "Non-normal"
        skew_l  = getattr(cs,"skew_label","") or ""
        out_ct  = getattr(cs,"outlier_count_iqr",0)
        story.append(Paragraph(
            "• '{}': {} | {} | Outliers: {}".format(col, normal, skew_l, out_ct),
            ST["body"]))

    # Correlations
    corrs = getattr(stats_report,"correlations",[])
    sig   = [c for c in corrs if getattr(c,"is_significant",False) and abs(getattr(c,"pearson_r",0))>=0.3]
    if sig:
        story.append(Spacer(1,2*mm))
        story.append(Paragraph("Significant Correlations", ST["sub_h"]))
        rows = [[Paragraph(h,ST["body_bold"]) for h in
                 ["Column A","Column B","r","p-value","Strength"]]]
        for c in sig[:6]:
            rows.append([Paragraph(c.col_a,ST["body"]),
                         Paragraph(c.col_b,ST["body"]),
                         Paragraph(str(c.pearson_r),ST["body"]),
                         Paragraph(str(c.p_value),ST["body"]),
                         Paragraph(c.strength.title(),ST["body"])])
        _table(story, T, rows, [CW*0.22,CW*0.22,CW*0.14,CW*0.14,CW*0.28])


# ══════════════════════════════════════════════════════════
#  BUSINESS INTELLIGENCE
# ══════════════════════════════════════════════════════════

def _bi_section(story, ST, T, bi_report, CW):
    if bi_report is None: return
    _page_h(story, ST, T, "Business Intelligence",
            "Benchmarking, cohort analysis, segment performance")

    if getattr(bi_report,"executive_brief",""):
        _narrative_box(story, ST, T, bi_report.executive_brief)

    # Benchmarks
    bms = getattr(bi_report,"benchmarks",[])
    if bms:
        story.append(Paragraph("Benchmarking Summary", ST["sub_h"]))
        rows = [[Paragraph(h,ST["body_bold"]) for h in
                 ["Metric","Mean","Median","Top 10%","Bottom 10%","Variation"]]]
        for bm in bms[:4]:
            rows.append([
                Paragraph(bm.column,                     ST["body"]),
                Paragraph(str(bm.mean),                  ST["body"]),
                Paragraph(str(bm.median),                ST["body"]),
                Paragraph(str(bm.top_10_pct),            ST["body"]),
                Paragraph(str(bm.bottom_10_pct),         ST["body"]),
                Paragraph(bm.benchmark_label.split("—")[0].strip()[:15], ST["body"]),
            ])
        _table(story, T, rows,
               [CW*0.22,CW*0.12,CW*0.12,CW*0.12,CW*0.12,CW*0.30])

    # Cohort findings
    sig_c = [c for c in getattr(bi_report,"cohorts",[]) if c.is_significant]
    if sig_c:
        story.append(Paragraph("Significant Cohort Differences", ST["sub_h"]))
        for c in sig_c[:2]:
            story.append(Paragraph("• " + c.interpretation, ST["body"]))

    # Key insights
    ki = getattr(bi_report,"key_insights",[])
    if ki:
        story.append(Spacer(1,2*mm))
        story.append(Paragraph("Key Business Insights", ST["sub_h"]))
        for ins in ki[:4]:
            story.append(Paragraph("• " + ins, ST["body"]))


# ══════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

def _recommendations(story, ST, T, actions, CW):
    _page_h(story, ST, T, "Recommendations & Action Plan",
            "Prioritized by urgency — act on CRITICAL items first")

    priority_colors = {
        "CRITICAL":   T["negative"],
        "SHORT TERM": T["warning"],
        "LONG TERM":  T["accent"],
    }

    for action in actions[:9]:
        # Determine priority
        if "CRITICAL" in action:
            priority = "CRITICAL"
            bg_key   = "critical_bg"
        elif "SHORT TERM" in action:
            priority = "SHORT TERM"
            bg_key   = "warning_bg"
        else:
            priority = "LONG TERM"
            bg_key   = "info_bg"

        color = priority_colors.get(priority, T["accent"])
        text  = action.replace("[{}] ".format(priority),"").strip()

        card = Table([[
            Table([[Paragraph(priority, ParagraphStyle(
                "p", fontName="Helvetica-Bold", fontSize=7,
                textColor=_rgb(color), alignment=TA_CENTER))
            ]], colWidths=[CW*0.14]),
            Paragraph(text, ST["body"]),
        ]], colWidths=[CW*0.14, CW*0.86])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _rgb(T[bg_key])),
            ("LINEBEFORE",    (0,0), (0,-1),  3, _rgb(color)),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([card, Spacer(1,2*mm)]))

    story.append(Spacer(1,3*mm))
    story.append(Paragraph(
        "All recommendations based solely on provided dataset. "
        "Verify findings with domain experts before implementation.",
        ST["caption"]))


# ══════════════════════════════════════════════════════════
#  APPENDIX
# ══════════════════════════════════════════════════════════

def _appendix(story, ST, T, config, df, CW):
    _page_h(story, ST, T, "Appendix", "Methodology and data quality details")

    story.append(Paragraph("A. Methodology", ST["sub_h"]))
    story.append(Paragraph(
        "Data quality scoring: 60% completeness, 30% deduplication, 10% column health. "
        "Outlier detection: IQR (1.5x) and Modified Z-Score. "
        "Normality: Shapiro-Wilk (n≤5000) and D'Agostino-Pearson tests. "
        "Correlations: Pearson (normal) / Spearman (non-normal). "
        "Domain detection: keyword matching across HR, Ecommerce, Sales, Finance. "
        "Attrition drivers: Mann-Whitney U (numeric) and Chi-Square (categorical). "
        "AI narratives: Groq Llama 3.3 70B with pre-computed statistics.",
        ST["body"]))

    story.append(Paragraph("B. Quality Score Formula", ST["sub_h"]))
    _table(story, T, [
        [Paragraph(h,ST["body_bold"]) for h in ["Component","Weight","Description"]],
        [Paragraph("Completeness",  ST["body"]), Paragraph("60%",ST["body"]),
         Paragraph("% of non-missing cells",    ST["body"])],
        [Paragraph("Deduplication", ST["body"]), Paragraph("30%",ST["body"]),
         Paragraph("% of unique rows",          ST["body"])],
        [Paragraph("Column Health", ST["body"]), Paragraph("10%",ST["body"]),
         Paragraph("Avg per-column quality",    ST["body"])],
    ], [CW*0.25,CW*0.15,CW*0.60])

    story.append(Paragraph("C. Disclaimer", ST["sub_h"]))
    story.append(Paragraph(
        "Report generated by DataForge AI on {} for {}. "
        "Findings based solely on provided dataset. "
        "Verify with qualified data analyst before business decisions.".format(
            datetime.now().strftime("%B %d, %Y"),
            config.get("client_name","Client")),
        ST["body"]))


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
    chart_data: list = None,
    executive_summary: str = "",
    findings: list = None,
    risks: list = None,
    opportunities: list = None,
    recommendations: list = None,
    # NEW: structured insights from story engine
    top_insights: list = None,
    attrition=None,
    domain: str = "general",
) -> bytes:
    """
    Build complete client-grade PDF report.
    Color theme auto-selected by domain.
    """
    findings       = findings or []
    risks          = risks or []
    opportunities  = opportunities or []
    recommendations= recommendations or []
    chart_data     = chart_data or []
    top_insights   = top_insights or []

    # Auto-select theme by domain
    theme_name = config.get("theme_name","")
    if theme_name not in THEMES:
        auto = DOMAIN_THEMES.get(domain, "Corporate Light")
        theme_name = auto
    T = THEMES[theme_name]

    config["domain"]     = domain
    config["theme_name"] = theme_name

    buf  = io.BytesIO()
    W, H = A4
    M    = 18 * mm
    CW   = W - 2*M

    # Header / Footer
    def _hf(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(_rgb(T["header_bg"]))
        canvas.rect(0, H-12*mm, W, 12*mm, fill=1, stroke=0)
        canvas.setFillColor(_rgb(T["header_text"]))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(M, H-7.5*mm, "DataForge AI")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W-M, H-7.5*mm,
            config.get("title","Report")[:55])
        canvas.setStrokeColor(_rgb(T["accent"]))
        canvas.setLineWidth(0.4)
        canvas.line(M, 11*mm, W-M, 11*mm)
        canvas.setFillColor(_rgb(T["text_muted"]))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(M, 7*mm,
            "Prepared for: " + config.get("client_name","Client"))
        if config.get("confidential", True):
            canvas.setFillColor(_rgb(T["negative"]))
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawCentredString(W/2, 7*mm, "CONFIDENTIAL")
        canvas.setFillColor(_rgb(T["text_muted"]))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(W-M, 7*mm, "Page " + str(doc.page))
        canvas.restoreState()

    frame = Frame(M, 14*mm, CW, H-28*mm,
                  leftPadding=0, rightPadding=0,
                  topPadding=4*mm, bottomPadding=2*mm)
    tpl = PageTemplate(id="main", frames=[frame], onPage=_hf)
    doc = BaseDocTemplate(buf, pagesize=A4, pageTemplates=[tpl],
                          leftMargin=M, rightMargin=M,
                          topMargin=14*mm, bottomMargin=14*mm)

    ST    = _styles(T, CW)
    story = []

    # ── TOC ───────────────────────────────────────────────
    pn, sn = 3, 1
    toc = []
    def _add(title):
        nonlocal pn, sn
        toc.append((sn, title, pn))
        sn += 1; pn += 1

    _add("Executive Summary")
    _add("Top Insights")
    if attrition:
        _add("Attrition Deep Dive")
    _add("Dataset Overview")
    if stats_report:
        _add("Statistical Analysis")
    if bi_report:
        _add("Business Intelligence")
    for i, (t,_,_) in enumerate(chart_data,1):
        _add("Chart {}: {}".format(i, t[:30]))
    _add("Recommendations")
    _add("Appendix")

    # ── Build ─────────────────────────────────────────────
    _cover(story, ST, T, config, CW)
    story.append(PageBreak())

    _toc(story, ST, T, toc, CW)
    story.append(PageBreak())

    _exec_summary(story, ST, T, executive_summary,
                  findings, risks, opportunities, CW)
    story.append(PageBreak())

    _top_insights(story, ST, T, top_insights, CW)
    story.append(PageBreak())

    if attrition:
        _attrition_page(story, ST, T, attrition, CW)
        story.append(PageBreak())

    _dataset_overview(story, ST, T, df, profile, cleaning_summary, CW)
    story.append(PageBreak())

    if stats_report:
        _stats_section(story, ST, T, stats_report, CW)
        story.append(PageBreak())

    if bi_report:
        _bi_section(story, ST, T, bi_report, CW)
        story.append(PageBreak())

    for i, (title, img_bytes, narrative) in enumerate(chart_data, 1):
        _chart_page(story, ST, T, img_bytes, title, narrative, i, CW)
        story.append(PageBreak())

    _recommendations(story, ST, T, recommendations, CW)
    story.append(PageBreak())

    _appendix(story, ST, T, config, df, CW)

    doc.build(story)
    buf.seek(0)
    return buf.read()
