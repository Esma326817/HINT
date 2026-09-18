"""Rendering and stage options shared by online and offline workflows."""

from __future__ import annotations

from typing import Any

SEMANTIC_INTENT_HIGHLIGHTING = "highlighting"
SEMANTIC_INTENT_ATTENTION = "attention"
SEMANTIC_INTENT_BOTH = "both"
SEMANTIC_INTENT_INJECTION_MODES = {
    SEMANTIC_INTENT_HIGHLIGHTING,
    SEMANTIC_INTENT_ATTENTION,
    SEMANTIC_INTENT_BOTH,
}


def semantic_intent_injection(config: dict[str, Any]) -> str:
    """Read the injection mode, defaulting to visual highlighting."""
    return str(
        config.get("task", {}).get(
            "semantic_intent_injection", SEMANTIC_INTENT_HIGHLIGHTING
        )
    ).strip().lower()


def injects_highlighting(mode: str) -> bool:
    """Whether the selected mode renders visual overlays."""
    return mode in (SEMANTIC_INTENT_HIGHLIGHTING, SEMANTIC_INTENT_BOTH)


def injects_attention(mode: str) -> bool:
    """Whether the selected mode exports attention guidance."""
    return mode in (SEMANTIC_INTENT_ATTENTION, SEMANTIC_INTENT_BOTH)


def skipped_stages(config: dict[str, Any]) -> set[str]:
    """Read the stages excluded from rendering and attention guidance."""
    return {
        str(stage).strip()
        for stage in config.get("stage_aware", {}).get("skip_stages", [])
    }
