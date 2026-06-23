"""
core/pdf/primitives.py — Reusable ReportLab building blocks.
All functions append to a story list — no side effects.
"""
"""
core/pdf_builder.py — DataForge AI
Senior Analyst Edition v4 — DROP-IN REPLACEMENT

UPGRADED from basic to:
  ✅ Premium dark-navy cover with domain badge + KPI strip
  ✅ Running header / footer on every page
  ✅ Insight cards: Problem → Cause → Evidence → Action → Impact
  ✅ Internal-only benchmark context (no external sources hardcoded)
  ✅ Correlation warning box (r ≠ causation)
  ✅ Scenario modelling table
  ✅ Risk register with severity colouring
  ✅ Deduplication transparency note
  ✅ All chart analyses properly domain-aware

SAME public API — zero changes needed in 8_Reports.py:
  build_pdf(df, config, profile, cleaning_summary,
            stats_report, bi_report, ml_report,
            chart_data, executive_summary,
            findings, risks, opportunities, recommendations,
            top_insights, attrition, domain) → bytes
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.pdfgen import canvas as CV
import logging
logger = logging.getLogger(__name__)
from core.pdf.theme import _c, _styles, _ReportCanvas

W, H = A4
CW_DEFAULT = W - 36 * mm   # content width (18mm each side)


# ══════════════════════════════════════════════════════════
#  DOMAIN COLOUR THEMES  (matches your existing THEMES keys)
# ══════════════════════════════════════════════════════════

THEMES = {
    "Corporate Light": {
        "cover_bg":    "#0A1628", "cover_accent": "#1B4FD8",
        "header_bg":   "#0A1628", "header_text":  "#FFFFFF",
        "accent":      "#1B4FD8", "accent2":      "#60A5FA",
        "text":        "#1F2937", "text_muted":   "#6B7280",
        "bg_light":    "#EFF6FF", "bg_card":      "#F8FAFF",
        "border":      "#E5E7EB",
        "positive":    "#10B981", "negative":     "#EF4444",
        "warning":     "#F59E0B", "info":         "#3B82F6",
        "critical_bg": "#FEE2E2", "warning_bg":   "#FEF3C7",
        "positive_bg": "#D1FAE5", "info_bg":      "#DBEAFE",
        "domain_label":"BUSINESS ANALYTICS",
        "domain_badge":"#1B4FD8",
    },
    "HR Blue": {
        "cover_bg":    "#0A1F4E", "cover_accent": "#1976D2",
        "header_bg":   "#0A1F4E", "header_text":  "#FFFFFF",
        "accent":      "#1976D2", "accent2":      "#90CAF9",
        "text":        "#1A2035", "text_muted":   "#5A6482",
        "bg_light":    "#E8F0FE", "bg_card":      "#F5F8FF",
        "border":      "#C5D3F0",
        "positive":    "#2E7D32", "negative":     "#C62828",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E3F2FD",
        "domain_label":"HR & PEOPLE ANALYTICS",
        "domain_badge":"#1976D2",
    },
    "Ecommerce Orange": {
        "cover_bg":    "#3E1500", "cover_accent": "#F4511E",
        "header_bg":   "#BF360C", "header_text":  "#FFFFFF",
        "accent":      "#F4511E", "accent2":      "#FFAB91",
        "text":        "#1A1A1A", "text_muted":   "#5A5A5A",
        "bg_light":    "#FBE9E7", "bg_card":      "#FFF8F6",
        "border":      "#FFCCBC",
        "positive":    "#2E7D32", "negative":     "#B71C1C",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E8F0FE",
        "domain_label":"E-COMMERCE ANALYTICS",
        "domain_badge":"#F4511E",
    },
    "Sales Green": {
        "cover_bg":    "#0A2710", "cover_accent": "#2E7D32",
        "header_bg":   "#1B5E20", "header_text":  "#FFFFFF",
        "accent":      "#2E7D32", "accent2":      "#A5D6A7",
        "text":        "#1A2A1A", "text_muted":   "#4A6A4A",
        "bg_light":    "#E8F5E9", "bg_card":      "#F5FBF5",
        "border":      "#C8E6C9",
        "positive":    "#1B5E20", "negative":     "#B71C1C",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E8F0FE",
        "domain_label":"SALES PERFORMANCE ANALYTICS",
        "domain_badge":"#2E7D32",
    },
    "Dark Tech": {
        "cover_bg":    "#0D1117", "cover_accent": "#58A6FF",
        "header_bg":   "#0D1117", "header_text":  "#E6EDF3",
        "accent":      "#58A6FF", "accent2":      "#3FB950",
        "text":        "#E6EDF3", "text_muted":   "#8B949E",
        "bg_light":    "#161B22", "bg_card":      "#1C2128",
        "border":      "#30363D",
        "positive":    "#3FB950", "negative":     "#F85149",
        "warning":     "#D29922", "info":         "#58A6FF",
        "critical_bg": "#1C1010", "warning_bg":   "#1C1800",
        "positive_bg": "#0D1A0F", "info_bg":      "#0D1421",
        "domain_label":"TECHNICAL ANALYTICS",
        "domain_badge":"#58A6FF",
    },
}

# Auto-select theme by domain
DOMAIN_THEMES = {
    "hr":        "HR Blue",
    "ecommerce": "Ecommerce Orange",
    "sales":     "Sales Green",
    "finance":   "Corporate Light",
    "general":   "Corporate Light",
}

# Internal comparison guidance only — no hardcoded external benchmarks
# Values in "General Guidance" column are labelled as indicative
HR_BENCHMARKS = []  # Not used — see _benchmark_section for dataset-computed values


# ══════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════



def _sec(story: list, s: dict, T: dict, title: str, sub: str = ""):
    story.append(Spacer(1, 3*mm))
    story.append(HRFlowable(width="100%", thickness=3,
                             color=_c(T["accent"]), spaceAfter=3))
    story.append(Paragraph(title, s["h2"]))
    if sub:
        story.append(Paragraph(sub, s["sm"]))


def _kpi_row(story: list, s: dict, T: dict, kpis: list, CW: float):
    cols = min(4, len(kpis))
    cw   = CW / cols
    vals = [Paragraph(
                "<b>{}</b>".format(k.get("value", "")),
                ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=18,
                               textColor=HexColor(k.get("color", T["accent"])),
                               alignment=TA_CENTER))
            for k in kpis[:cols]]
    lbls = [Paragraph(
                k.get("label", ""),
                ParagraphStyle("kl", fontName="Helvetica-Bold", fontSize=7.5,
                               textColor=_c(T["text"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    subs = [Paragraph(
                k.get("sub", ""),
                ParagraphStyle("ks", fontName="Helvetica", fontSize=7,
                               textColor=_c(T["text_muted"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    t = Table([vals, lbls, subs], colWidths=[cw]*cols)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))


def _narrative_box(story: list, s: dict, T: dict, text: str):
    if not text: return
    t = Table([[Paragraph(text, s["body"])]], colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
        ("LINEBEFORE",    (0,0), (0,-1),  4, _c(T["accent"])),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 2*mm))


def _gtable(story: list, T: dict, headers: list,
            rows_data: list, col_widths: list,
            severity_col: int = -1):
    """Generic styled table."""
    hrow = [Paragraph(h, ParagraphStyle(
                "th", fontName="Helvetica-Bold", fontSize=8,
                textColor=HexColor("#FFFFFF"), alignment=TA_CENTER))
            for h in headers]
    body = []
    for row in rows_data:
        body.append([Paragraph(str(c), ParagraphStyle(
                "td", fontName="Helvetica", fontSize=8,
                textColor=_c(T["text"]), leading=12))
                     for c in row])
    tbl = Table([hrow] + body, colWidths=col_widths)
    sty = [
        ("BACKGROUND",    (0,0), (-1,0),  _c(T["header_bg"])),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#FFFFFF"), _c(T["bg_light"])]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]
    if severity_col >= 0:
        _sev = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B",
                "MEDIUM":   "#06B6D4", "LOW":  "#10B981",
                "IMMEDIATE":"#EF4444", "SHORT TERM": "#F59E0B",
                "LONG TERM":"#3B82F6", "WARNING": "#3B82F6",
                "NONE":     "#10B981"}
        for i, row in enumerate(rows_data, 1):
            val = str(row[severity_col]).upper()
            for k, col in _sev.items():
                if k in val:
                    sty += [
                        ("BACKGROUND", (severity_col, i), (severity_col, i), HexColor(col)),
                        ("TEXTCOLOR",  (severity_col, i), (severity_col, i), HexColor("#FFFFFF")),
                        ("FONTNAME",   (severity_col, i), (severity_col, i), "Helvetica-Bold"),
                        ("FONTSIZE",   (severity_col, i), (severity_col, i), 7),
                    ]
                    break
    tbl.setStyle(TableStyle(sty))
    story.append(tbl)
    story.append(Spacer(1, 2*mm))


def _insight_card(story: list, s: dict, T: dict, ins, CW: float, num=None):
    """
    Works with both:
      - Dataclass objects (has .severity, .title, .problem …)
      - Plain dicts (keys: severity, title, problem …)
    """
    def _get(obj, key, default=""):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    sev    = _get(ins, "severity", "info").lower()
    sev_c  = {"critical": T["negative"], "high": T["warning"],
              "warning":  T["info"],     "info": T["text_muted"]}
    sev_bg = {"critical": T["critical_bg"], "high": T["warning_bg"],
              "warning":  T["info_bg"],     "info": T["bg_card"]}
    col = sev_c.get(sev, T["accent"])
    _bg = sev_bg.get(sev, T["bg_card"])  # noqa: F841

    bs = ParagraphStyle("bi_badge", fontName="Helvetica-Bold", fontSize=7.5,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    ts = ParagraphStyle("bi_title", fontName="Helvetica-Bold", fontSize=9.5,
                        textColor=_c(T["text"]))
    rl = ParagraphStyle("bi_lbl",   fontName="Helvetica-Bold", fontSize=8,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    rv = ParagraphStyle("bi_val",   fontName="Helvetica",      fontSize=8.5,
                        textColor=_c(T["text"]),  leading=12.5)

    num_str = "{}. ".format(num) if num else ""
    hdr = Table([[
        Paragraph(_get(ins, "severity", "INFO").upper(), bs),
        Paragraph("{}{}".format(num_str, _get(ins, "title", "")), ts),
    ]], colWidths=[20*mm, CW - 20*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), HexColor(col)),
        ("BACKGROUND", (1,0), (1,0), _c(T["bg_light"])),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",      (0,0), (0,0),  "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("BOX",        (0,0), (-1,-1), 0.5, _c(T["border"])),
    ]))

    lw, vw = 26*mm, CW - 26*mm
    rows = [
        [Paragraph(k, rl), Paragraph(_get(ins, fk, ""), rv)]
        for k, fk in [("PROBLEM", "problem"), ("CAUSE",  "cause"),
                      ("EVIDENCE","evidence"),("ACTION", "action"),
                      ("IMPACT",  "impact")]
        if _get(ins, fk, "").strip()
    ]
    if not rows:
        rows = [[Paragraph("DETAIL", rl),
                 Paragraph(str(ins)[:200], rv)]]

    body = Table(rows, colWidths=[lw, vw])
    body.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), _c(T["header_bg"])),
        ("ROWBACKGROUNDS",(1,0),(1,-1),
         [HexColor("#FFFFFF"), _c(T["bg_light"]),
          HexColor("#FFFFFF"), _c(T["bg_light"]), HexColor("#FFFFFF")]),
        ("VALIGN",  (0,0), (-1,-1), "TOP"),
        ("ALIGN",   (0,0), (0,-1),  "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("BOX",       (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID", (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]))
    story.append(KeepTogether([hdr, body, Spacer(1, 4*mm)]))


# ══════════════════════════════════════════════════════════
#  TOC PAGE
# ══════════════════════════════════════════════════════════

def _toc(story, s, T, entries, CW):
    _sec(story, s, T, "Table of Contents")
    for num, title in entries:
        row = Table([[
            Paragraph(str(num), ParagraphStyle(
                "tn", fontName="Helvetica-Bold", fontSize=10,
                textColor=_c(T["accent"]), alignment=TA_CENTER)),
            Paragraph(title, s["toc"]),
        ]], colWidths=[9*mm, CW - 9*mm])
        row.setStyle(TableStyle([
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",   (0,0), (-1,-1), 0.3, _c(T["border"])),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(row)


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════

