"""Semantic perception: task intent, target grounding, and visual tracking.

Public layout:

- ``task_manager``: VLM runtime + target-phrase resolution
- ``semantic_grounder``: GroundingDINO / Qwen bbox selection
- ``tracking``: SAM2 / bbox-mask propagation
"""

from __future__ import annotations
