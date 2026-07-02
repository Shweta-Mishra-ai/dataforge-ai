"""
core/pdf/theme.py — PDF theme: colors, styles, canvas, cover page.
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

def _c(hex_str: str) -> HexColor:
    return HexColor(hex_str)


# ══════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════

def _styles(T: dict) -> dict:
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "h1":    ps("h1",   fontName="Helvetica-Bold", fontSize=17,
                    textColor=_c(T["accent"]),     spaceAfter=4),
        "h2":    ps("h2",   fontName="Helvetica-Bold", fontSize=13,
                    textColor=_c(T["text"]),       spaceBefore=8, spaceAfter=3),
        "h3":    ps("h3",   fontName="Helvetica-Bold", fontSize=10,
                    textColor=_c(T["accent"]),     spaceBefore=6, spaceAfter=3),
        "body":  ps("body", fontName="Helvetica",      fontSize=9,
                    textColor=_c(T["text"]),       leading=14,  spaceAfter=3,
                    alignment=TA_JUSTIFY),
        "sm":    ps("sm",   fontName="Helvetica",      fontSize=7.5,
                    textColor=_c(T["text_muted"]), leading=11,  spaceAfter=2),
        "bl":    ps("bl",   fontName="Helvetica",      fontSize=9,
                    textColor=_c(T["text"]),       leading=13,  spaceAfter=3,
                    leftIndent=10, firstLineIndent=-10),
        "toc":   ps("toc",  fontName="Helvetica",      fontSize=10,
                    textColor=_c(T["text"]),       leading=16,  spaceAfter=3),
        "wh":    ps("wh",   fontName="Helvetica",      fontSize=9,
                    textColor=HexColor("#FFFFFF"),  leading=13),
        "wbh":   ps("wbh",  fontName="Helvetica-Bold", fontSize=10,
                    textColor=HexColor("#FFFFFF")),
        "note":  ps("note", fontName="Helvetica-Oblique", fontSize=7.5,
                    textColor=_c(T["text_muted"]), spaceAfter=3),
        "warn":  ps("warn", fontName="Helvetica",      fontSize=8.5,
                    textColor=_c(T["text"]),       leading=13,
                    backColor=_c(T["warning_bg"])),
        # Insight card row styles
        "rl":    ps("rl",   fontName="Helvetica-Bold", fontSize=8,
                    textColor=HexColor("#FFFFFF"),  alignment=TA_CENTER),
        "rv":    ps("rv",   fontName="Helvetica",      fontSize=8.5,
                    textColor=_c(T["text"]),       leading=12.5),
    }


# ══════════════════════════════════════════════════════════
#  RUNNING HEADER / FOOTER  (PageCanvas)
# ══════════════════════════════════════════════════════════

class _ReportCanvas(CV.Canvas):
    """Draws premium header + footer on every content page."""

    def __init__(self, fn, T, report_title="", client_name="", report_date="",
                 agency_name="DataForge AI", **kw):
        super().__init__(fn, **kw)
        self._sp          = []
        self.T            = T
        self.report_title = report_title[:60]
        self.client_name  = client_name
        self.report_date  = report_date
        # White-label: freelancers/agencies can override the masthead brand
        self.agency_name  = (agency_name or "DataForge AI")[:40]

    def showPage(self):
        self._sp.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        tot = len(self._sp)
        # Cover page = page 1 → content pages start at page 2 in the merged PDF
        page_offset = getattr(self, "_page_offset", 1)
        for i, st in enumerate(self._sp):
            self.__dict__.update(st)
            self._draw(tot, page_number=i + 1 + page_offset, total_pages=tot + page_offset)
            super().showPage()
        super().save()

    def _draw(self, tot, page_number=None, total_pages=None):
        T = self.T
        # ── Header ────────────────────────────────────────
        self.setFillColor(_c(T["header_bg"]))
        self.rect(0, H - 24*mm, W, 24*mm, fill=1, stroke=0)
        self.setFillColor(_c(T["accent"]))
        self.rect(0, H - 25.5*mm, W, 1.5*mm, fill=1, stroke=0)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont("Helvetica-Bold", 10)
        self.drawString(18*mm, H - 14*mm, self.agency_name)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(HexColor(T["accent2"]))
        self.drawString(18*mm, H - 20*mm, self.report_title)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont("Helvetica", 7)
        self.drawRightString(W - 18*mm, H - 14*mm, self.report_date)
        self.drawRightString(W - 18*mm, H - 20*mm,
                             "CONFIDENTIAL — " + self.client_name)
        # ── Footer ────────────────────────────────────────
        self.setFillColor(_c(T["header_bg"]))
        self.rect(0, 0, W, 12*mm, fill=1, stroke=0)
        self.setFillColor(_c(T["accent"]))
        self.rect(0, 12*mm, W, 1.2*mm, fill=1, stroke=0)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont("Helvetica", 6.5)
        self.drawString(18*mm, 4.5*mm,
            _domain_footer_text(getattr(self, "_domain", "general")))
        # Use explicit page_number / total_pages if provided (correct for merged cover+content)
        pn  = page_number   if page_number   is not None else self._pageNumber
        tot_display = total_pages if total_pages is not None else tot
        self.drawRightString(W - 18*mm, 4.5*mm,
            "Page {} of {}".format(pn, tot_display))

def _domain_footer_text(domain: str) -> str:
    """FIX-054: Domain-specific footer — no HR benchmarks in sales reports."""
    _map = {
        "hr":        "All metrics computed from submitted dataset only — no external benchmarks embedded",
        "ecommerce": "Benchmarks: Amazon · Shopify · Klaviyo — verify with e-commerce specialists before action",
        "sales":     "Benchmarks: Salesforce · Gartner · HubSpot — verify with sales leadership before action",
        "finance":   "Benchmarks: PwC · McKinsey · Deloitte — verify with qualified finance professionals before action",
        "general":   "Findings based on provided dataset — verify with domain experts before action",
    }
    return _map.get(domain, _map["general"])


# ══════════════════════════════════════════════════════════
#  COVER PAGE  (drawn on separate canvas, merged via pypdf)
# ══════════════════════════════════════════════════════════

def _build_cover(T: dict, config: dict, kpis_preview: list) -> bytes:
    buf = io.BytesIO()
    cv  = CV.Canvas(buf, pagesize=A4)
    title       = config.get("title", "Data Analysis Report")
    client_name = config.get("client_name", "Client")
    report_date = datetime.now().strftime("%B %d, %Y")
    domain_lbl  = T.get("domain_label", "BUSINESS ANALYTICS")
    # White-label: agency_name overrides "DataForge AI" on cover; tagline is configurable
    agency_name = (config.get("agency_name") or "DataForge AI")[:40]
    agency_tagline = config.get("agency_tagline", "Advanced Analytics Platform")[:60]

    # BG
    cv.setFillColor(_c(T["cover_bg"]))
    cv.rect(0, 0, W, H, fill=1, stroke=0)
    # Top stripe
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(0, H - 5*mm, W, 5*mm, fill=1, stroke=0)
    # Right panel
    cv.setFillColor(HexColor("#0D1F3C"))
    cv.rect(W - 17*mm, 0, 17*mm, H, fill=1, stroke=0)
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(W - 17*mm, 0, 1.8*mm, H, fill=1, stroke=0)
    # Decorative circles
    cv.setFillColor(HexColor("#112240"))
    cv.circle(W * 0.73, H * 0.53, 190, fill=1, stroke=0)
    cv.setFillColor(HexColor("#0D1A35"))
    cv.circle(W * 0.73, H * 0.53, 135, fill=1, stroke=0)

    # Brand
    cv.setFillColor(HexColor("#FFFFFF"))
    cv.setFont("Helvetica-Bold", 15)
    cv.drawString(20*mm, H - 32*mm, agency_name)
    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont("Helvetica", 9.5)
    cv.drawString(20*mm, H - 40*mm, agency_tagline)
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(20*mm, H - 44*mm, 55*mm, 1.2*mm, fill=1, stroke=0)

    # ── Client / Company Logo (top-right of cover) ────────
    logo_bytes = config.get("logo_bytes", None)
    logo_path  = config.get("logo_path", "")
    _logo_drawn = False

    if logo_bytes:
        try:
            import tempfile
            logo_ext = config.get("logo_ext", "png")
            _tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + logo_ext)
            _tmp.write(logo_bytes)
            _tmp.flush()
            _tmp.close()
            cv.drawImage(
                _tmp.name,
                W - 68*mm, H - 45*mm,
                width=48*mm, height=20*mm,
                preserveAspectRatio=True,
                mask="auto",
            )
            _logo_drawn = True
            try:
                os.unlink(_tmp.name)
            except Exception:
                logger.warning("PDF theme unexpected failure", exc_info=True)
        except Exception:
            logger.warning("PDF theme unexpected failure", exc_info=True)

    if not _logo_drawn and logo_path and os.path.exists(logo_path):
        try:
            cv.drawImage(
                logo_path,
                W - 68*mm, H - 45*mm,
                width=48*mm, height=20*mm,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            logger.warning("PDF logo embed failed — continuing without logo", exc_info=True)

    # Domain badge
    cv.setFillColor(_c(T["domain_badge"]))
    cv.roundRect(20*mm, H - 60*mm, 85*mm, 11*mm, 3, fill=1, stroke=0)
    cv.setFillColor(HexColor("#FFFFFF"))
    cv.setFont("Helvetica-Bold", 8)
    cv.drawString(25*mm, H - 56*mm, "◆  " + domain_lbl)

    # Title (word wrap at ~28 chars)
    words, lines, line = title.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if len(test) <= 28:
            line = test
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)

    cv.setFillColor(HexColor("#FFFFFF"))
    y_title = H / 2 + 36*mm
    for ln in lines:
        cv.setFont("Helvetica-Bold", 30 if len(ln) <= 20 else 24)
        cv.drawString(20*mm, y_title, ln)
        y_title -= 11*mm

    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont("Helvetica", 10)
    cv.drawString(20*mm, H / 2 + 12*mm,
                  config.get("subtitle", f"Powered by {agency_name}"))
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(20*mm, H / 2 + 6*mm, W - 37*mm, 1.5*mm, fill=1, stroke=0)

    # KPI strip (up to 4)
    kpis = kpis_preview[:4]
    bw   = (W - 37*mm) / max(len(kpis), 1)
    for i, kpi in enumerate(kpis):
        x = 20*mm + i * bw
        cv.setFillColor(HexColor("#1A3A5C"))
        cv.roundRect(x + 1.5, H / 2 - 14*mm, bw - 3, 18*mm, 3, fill=1, stroke=0)
        cv.setFillColor(HexColor(kpi.get("color", T["accent2"])))
        cv.setFont("Helvetica-Bold", 16)
        cv.drawCentredString(x + bw / 2, H / 2 - 2*mm,
                             str(kpi.get("value", ""))[:9])
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont("Helvetica-Bold", 7)
        cv.drawCentredString(x + bw / 2, H / 2 - 8*mm,
                             str(kpi.get("label", ""))[:18])
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont("Helvetica", 6.5)
        cv.drawCentredString(x + bw / 2, H / 2 - 13*mm,
                             str(kpi.get("sub", ""))[:20])

    # Bottom meta
    cv.setFillColor(HexColor("#0D1F3C"))
    cv.rect(0, 0, W - 17*mm, 30*mm, fill=1, stroke=0)
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(0, 30*mm, W - 17*mm, 1.2*mm, fill=1, stroke=0)
    meta = [("PREPARED FOR", client_name),
            ("DATE", report_date),
            ("CLASSIFICATION", "CONFIDENTIAL")]
    mw = (W - 37*mm) / 3
    for i, (k, v) in enumerate(meta):
        x = 20*mm + i * mw
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont("Helvetica", 6.5)
        cv.drawString(x, 21*mm, k)
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont("Helvetica-Bold", 8)
        cv.drawString(x, 13*mm, v[:22])

    # FIX-020: AI branding removed — report is signed by analyst, not tool
    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont("Helvetica", 6.5)
    cv.drawRightString(W - 21*mm, 5*mm,
                       f"{agency_name} — {agency_tagline}")
    cv.save()
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  REUSABLE COMPONENT HELPERS
# ══════════════════════════════════════════════════════════

