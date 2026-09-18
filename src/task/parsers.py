"""Reusable parsers for recognition and episode-plan output.

A recognition prompt is only half of a task definition: the other half is what
counts as a valid answer. The parsers here cover the output shapes that recur
across tasks, so a spec can name one in
``configs/tasks/<name>.yaml`` (``recognition.landmark_parser`` /
``recognition.crop_parser``) instead of shipping a regex in Python.

Episode plans are different: ``task.base.resolve_episode_task_context`` owns the
shared priority (payload → prompt → YAML defaults). A task may pass
``parse_episode_plan`` into ``define_task`` when its plan fields need validation
or a synthesized instruction (e.g. peg-in-hole color/shape).
"""

from __future__ import annotations

import re
from typing import Any, Callable


def passthrough(text: Any) -> str:
    """Collapse whitespace and keep the answer as written."""
    return " ".join(str(text).strip().split())


def last_word(text: Any) -> str:
    """Last lowercase English word, for answers that trail a sentence."""
    words = re.findall(r"[a-z]+", str(text).lower())
    if not words:
        raise ValueError(f"failed to extract landmark word from Qwen output: {text!r}")
    return words[-1]


def single_letter(text: Any) -> str:
    """One letter out of a short answer; raise when the answer is ambiguous."""
    lowered = str(text).lower().strip()
    exact_letter = re.fullmatch(r"[^a-z]*([a-z])[^a-z]*", lowered)
    if exact_letter:
        return exact_letter.group(1)

    standalone_letters = re.findall(r"\b([a-z])\b", lowered)
    if standalone_letters:
        return standalone_letters[-1]

    labeled_answer = re.search(r"(?:letter|answer|result)\s*(?:is|:)\s*([a-z])\b", lowered)
    if labeled_answer:
        return labeled_answer.group(1)

    if not re.findall(r"[a-z]", lowered):
        raise ValueError(f"failed to extract letter from Qwen output: {text!r}")
    raise ValueError(f"ambiguous letter output from Qwen: {text!r}")


def normalized_label(text: Any) -> str:
    """Lowercase alphanumeric words, for open-set object names."""
    from task.base import normalize_label

    return normalize_label(text)


PARSERS: dict[str, Callable[[Any], str]] = {
    "passthrough": passthrough,
    "last_word": last_word,
    "single_letter": single_letter,
    "normalized_label": normalized_label,
}

#: Crop parser name handled by ``TaskSpec`` itself: it needs the classify spec.
LABEL_CATEGORY_JSON = "label_category_json"


def get_parser(name: str) -> Callable[[Any], str]:
    try:
        return PARSERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(PARSERS))
        raise KeyError(f"unknown parser {name!r}; known: {known}") from exc
