"""
core/pdf/builder.py — Public entry point: build_pdf().
Orchestrates all sections into a final merged PDF bytes object.
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


from core.pdf.theme import _build_cover, _ReportCanvas, _styles, _c  # noqa: F811 — _styles/_c not in shared block
from core.pdf.primitives import _sec, _kpi_row, _narrative_box, _gtable, _insight_card, _toc
from core.pdf.sections import (
    _exec_summary, _top_insights, _dq_note, _benchmark_section,
    _attrition_page, _dataset_overview, _stats_section,
    _bi_section, _chart_page, _recommendations,
)
from core.pdf.domain_sections import _finance_page, _appendix


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
    top_insights: list = None,
    attrition=None,
    domain: str = "general",
) -> bytes:
    """
    Build complete senior-analyst PDF report.
    Identical public API to original pdf_builder.py — drop-in replacement.
    """
    from pypdf import PdfWriter, PdfReader

    findings        = findings       or []
    risks           = risks          or []
    opportunities   = opportunities  or []
    recommendations = recommendations or []
    chart_data      = chart_data     or []
    top_insights    = top_insights   or []

    # ── Theme selection ───────────────────────────────────
    theme_name = config.get("theme_name", "")
    if theme_name not in THEMES:
        auto_key  = DOMAIN_THEMES.get(domain, "Corporate Light")
        theme_name = auto_key
    T = dict(THEMES[theme_name])  # copy — we mutate domain_label below

    # Override domain label/badge based on detected domain (not just theme)
    _DOMAIN_LABELS = {
        "hr":        ("HR & PEOPLE ANALYTICS",    "#1976D2"),
        "ecommerce": ("E-COMMERCE ANALYTICS",     "#F4511E"),
        "sales":     ("SALES PERFORMANCE",        "#2E7D32"),
        "finance":   ("FINANCE & ACCOUNTING",     "#5C35CC"),
    }
    if domain in _DOMAIN_LABELS:
        T["domain_label"], T["domain_badge"] = _DOMAIN_LABELS[domain]

    config["domain"]     = domain
    config["theme_name"] = theme_name

    report_title = config.get("title", "Data Analysis Report")
    client_name  = config.get("client_name", "Client")
    report_date  = datetime.now().strftime("%B %d, %Y")

    # ── KPI preview for cover ─────────────────────────────
    n_rows    = len(df)
    miss_pct  = df.isna().mean().mean() * 100
    n_charts  = len(chart_data)
    qual      = getattr(profile, "overall_quality_score", "—")

    kpis_cover = [
        {"label": "RECORDS",      "value": "{:,}".format(n_rows),
         "sub":   "Clean dataset", "color": T["accent"]},
        {"label": "QUALITY",      "value": str(qual),
         "sub":   "/ 100",         "color": T["positive"]},
        {"label": "CHARTS",       "value": str(n_charts),
         "sub":   "Incl. in report","color": T["accent"]},
        {"label": "MISSING DATA", "value": "{:.1f}%".format(miss_pct),
         "sub":   "0% = perfect",  "color": T["positive"] if miss_pct < 1 else T["warning"]},
    ]

    # ── Cover page ────────────────────────────────────────
    cover_bytes = _build_cover(T, config, kpis_cover)

    # ── Content pages ─────────────────────────────────────
    content_buf = io.BytesIO()
    M   = 18 * mm
    CW  = W - 2 * M

    def canvas_maker(fn, **kw):
        c = _ReportCanvas(fn, T=T,
                             report_title=report_title,
                             client_name=client_name,
                             report_date=report_date, **kw)
        c._domain = domain
        return c

    doc = BaseDocTemplate(
        content_buf, pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=30*mm, bottomMargin=17*mm,
    )
    frame = Frame(M, 17*mm, CW, H - 47*mm,
                  leftPadding=0, rightPadding=0,
                  topPadding=0,  bottomPadding=0)
    tpl   = PageTemplate(id="main", frames=[frame], onPage=lambda c,d: None)
    doc.addPageTemplates([tpl])

    s     = _styles(T)
    story = []

    # ── Build TOC entries ─────────────────────────────────
    sec_num = 1
    toc     = []
    def _add_toc(title):
        nonlocal sec_num
        toc.append((sec_num, title))
        sec_num += 1

    _add_toc("Executive Summary")
    _add_toc("Data Quality & Transparency Note")
    if domain in ("hr", "ecommerce", "sales"):
        _add_toc("Performance Context")
    if domain == "finance":
        _add_toc("Finance Analysis — P&L · Margin · Budget")
    _add_toc("Top Insights — Decision Summary")
    if attrition:
        _add_toc("Attrition Deep Dive")
    _add_toc("Dataset Overview & Descriptive Statistics")
    if stats_report:
        _add_toc("Statistical Analysis")
    if bi_report:
        _add_toc("Business Intelligence")
    for i, (t, _, _) in enumerate(chart_data, 1):
        _add_toc("Chart {}: {}".format(i, t[:28]))
    _add_toc("Recommendations & Action Plan")
    _add_toc("Appendix — Methodology & Sources")

    # ── Assemble story ────────────────────────────────────
    _toc(story, s, T, toc, CW)
    story.append(PageBreak())

    _exec_summary(story, s, T, executive_summary,
                  findings, risks, opportunities, CW)
    story.append(PageBreak())

    _dq_note(story, s, T, df, profile, CW)
    story.append(PageBreak())

    if domain in ("hr", "ecommerce", "sales"):
        _benchmark_section(story, s, T, domain, CW, df=df)
        story.append(PageBreak())
    if domain == "finance":
        _finance_page(story, s, T, df, config, CW, profile)
        story.append(PageBreak())

    _top_insights(story, s, T, top_insights, CW)
    story.append(PageBreak())

    if attrition:
        _attrition_page(story, s, T, attrition, CW, config=config)
        story.append(PageBreak())

    _dataset_overview(story, s, T, df, profile, CW)
    story.append(PageBreak())

    if stats_report:
        _stats_section(story, s, T, stats_report, CW)
        story.append(PageBreak())

    if bi_report:
        _bi_section(story, s, T, bi_report, CW)
        story.append(PageBreak())

    for i, (title, img_bytes, narrative) in enumerate(chart_data, 1):
        _chart_page(story, s, T, img_bytes, title, narrative, i, CW)
        story.append(PageBreak())

    _recommendations(story, s, T, recommendations, CW)
    story.append(PageBreak())

    _appendix(story, s, T, config, CW, domain=domain, df=df, profile=profile)

    # ── Build PDF ─────────────────────────────────────────
    doc.build(story, canvasmaker=canvas_maker)
    content_buf.seek(0)

    # ── Merge cover + content ─────────────────────────────
    writer = PdfWriter()
    for pg in PdfReader(io.BytesIO(cover_bytes)).pages:
        writer.add_page(pg)
    for pg in PdfReader(content_buf).pages:
        writer.add_page(pg)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()
