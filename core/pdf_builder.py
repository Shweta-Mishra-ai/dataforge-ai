"""
core/pdf_builder.py — SHIM (backwards compatibility only).
All new code should import from core.pdf.builder directly.
"""
from core.pdf.builder import build_pdf  # noqa: F401
from core.pdf.theme import THEMES  # noqa: F401
from core.pdf.domain_sections import _appendix  # noqa: F401

__all__ = ["build_pdf", "THEMES", "_appendix"]
