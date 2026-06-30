"""
core/insight_engine.py — DEPRECATED SHIM.

All insight generation has been consolidated into:
    core/insights_builder.py  → build_top_insights() → List[Insight]
    core/engines/             → domain engines (hr, finance, sales, ecommerce, general)

This file is kept only so that any external references don't hard-crash.
It re-exports the canonical functions from their new locations.
"""
import logging
logger = logging.getLogger(__name__)
from core.insights_builder import build_top_insights  # noqa: F401
from core.story_engine import Insight                  # noqa: F401

def generate_insights(df, domain="general"):
    """
    DEPRECATED — use build_top_insights() instead.
    This wrapper exists only for backward compatibility.
    Returns List[Dict] format (legacy) from the new Insight objects.
    """
    logger.warning(
        "generate_insights() is deprecated. "
        "Use core.insights_builder.build_top_insights() directly."
    )
    from core.story_engine import generate_story
    try:
        story = generate_story(df)
        insights = build_top_insights(df=df, domain=domain, story_obj=story)
        return [
            {
                "title":        ins.title,
                "body":         f"{ins.problem} {ins.cause}",
                "type":         ins.severity,
                "icon":         {"critical":"🔴","warning":"🟠","info":"🔵","positive":"🟢"}.get(ins.severity,"⚪"),
                "metric":       ins.category,
                "value":        None,
                "benchmark":    "N/A",
                "columns_used": [],
            }
            for ins in insights
        ]
    except Exception:
        logger.warning("generate_insights fallback failed", exc_info=True)
        return []
