"""
core/domain_dashboards.py — SHIM (backwards compatibility only).
Split into core/dashboards/ — see that package for details.
"""
import logging
logger = logging.getLogger(__name__)

from core.dashboards.base import get_domain_kpis, get_domain_charts  # noqa: F401

__all__ = ["get_domain_kpis", "get_domain_charts"]
