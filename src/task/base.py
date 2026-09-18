"""Shared types and helpers for task prompts, grounding, and detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from pattern.runtime.types import (
    DEXTEROUS_CONTACT_STAGE,
    FREE_MOVE_STAGE,
    PRE_CONTACT_STAGE,
    TRANSPORT_CONTACT_STAGE,
)

if TYPE_CHECKING:
    from PIL import Image

    from perception.semantic_grounder.dino import DinoClient
    from pattern.runtime.types import StageDecision
    from common.task_types import TaskState


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskPrompts:
    """VLM prompt templates for one manipulation task."""

    name: str
    landmark_recognition: str
    """Identify the target card or object at task start (global crop)."""
    crop_recognition: str | None = None
    """Optional per-crop recognition prompt (e.g. letter on a block)."""
    grounding_generation: str = ""
    """Prompt used when ``vlm.generate_prompt`` asks Qwen for a grounding phrase."""


COMMON_GROUNDING_GENERATION = """You choose the visual target phrase for the current robot step.

Read the task instruction, current action pattern, focus, active target, and default
phrase. Return the thing the robot should look at now: the object it is going for
while the hand is empty, and the destination or contact site once the object is in
hand. If the instruction has multiple actions, follow the active subtask instead of
restarting from the first action.

Output one short noun phrase only.
"""


# The four action patterns split by what the hand is holding. A subtask walks them
# in order, and leaving the contact group starts the next subtask.
ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "free": (FREE_MOVE_STAGE, PRE_CONTACT_STAGE),  # empty hand: look at the object
    "contact": (DEXTEROUS_CONTACT_STAGE, TRANSPORT_CONTACT_STAGE),  # holding it: look at the destination
}


def action_pattern_group(stage_name: str) -> str | None:
    """Which hand state ``stage_name`` belongs to (``free`` / ``contact``)."""
    for group, patterns in ACTION_PATTERNS.items():
        if stage_name in patterns:
            return group
    return None


# Global-only spatial cues. Wrist close-ups cannot see left/middle/right.
SPATIAL_MODIFIERS = frozenset(
    {
        "left",
        "leftmost",
        "middle",
        "center",
        "centre",
        "central",
        "right",
        "rightmost",
    }
)


def without_spatial_modifiers(phrase: str) -> str:
    """Drop left/middle/right so a wrist view can reuse the global object phrase."""
    kept = [
        token
        for token in str(phrase or "").split()
        if token.lower().strip(".,;:") not in SPATIAL_MODIFIERS
    ]
    return " ".join(kept)


class TaskHandler(Protocol):
    """Per-task prompts plus parsing and stage grounding rules.

    Prefer a YAML spec under ``configs/tasks/<name>.yaml``. Optional control-flow
    hooks live in ``src/task/task_hooks/<name>.py`` and are attached by ``task.registry``.
    Required after ``define_task``: prompts, parse_*, build_initial_state,
    resolve_rule_grounding_prompt. Optional: build_scene_layout, parse_crop_result,
    refine_scene, resolve_global_grounding_box, refresh_movables,
    seed_global_before_grounding, capture_tracker_parent_bbox,
    resolve_segment_render_mode, resolve_task_context / parse_episode_plan,
    on_subtask_advance.
    """

    prompts: TaskPrompts

    def parse_landmark_output(self, text: str) -> str: ...

    def parse_crop_output(self, text: str) -> str: ...

    def build_initial_state(
        self,
        *,
        image: Image.Image,
        dino_client: DinoClient,
        config: dict[str, Any],
        recognition_padding: int,
        task_context: Any | None = None,
    ) -> TaskState: ...

    def resolve_rule_grounding_prompt(
        self,
        *,
        decision: StageDecision,
        task_state: TaskState | None,
        task_instruction: str | None,
        base_prompt: str,
        config: dict[str, Any] | None = None,
    ) -> str: ...

    def resolve_task_context(
        self,
        *,
        config: dict[str, Any] | None = None,
        task_context: Any | None = None,
        episode_prompt: str | None = None,
    ) -> Any | None: ...


def grounding_generation_prompt(task_context: str = "") -> str:
    task_context = task_context.strip()
    if not task_context:
        return COMMON_GROUNDING_GENERATION
    return f"{COMMON_GROUNDING_GENERATION}\nTask context:\n{task_context}\n"


def task_config(config: dict[str, Any], task_name: str) -> dict[str, Any]:
    return config.get("task", {}).get(task_name, {})


def plan_from_prompt_enabled(
    config: dict[str, Any] | None,
    *,
    task_name: str | None = None,
) -> bool:
    """Whether free-text episode prompts may supply the per-episode task plan.

    Reads ``task.plan_from_prompt``, then optional ``task.<name>.plan_from_prompt``.
    Structured ``TaskContext.payload`` (dataset jsonl fields) is always honored
    by task handlers regardless of this flag.
    """
    task_cfg = (config or {}).get("task", {})
    if "plan_from_prompt" in task_cfg:
        return bool(task_cfg.get("plan_from_prompt"))
    if task_name:
        nested = task_config(config or {}, task_name)
        if "plan_from_prompt" in nested:
            return bool(nested.get("plan_from_prompt"))
    return False


# Keys under ``task`` / ``task.<name>`` that are runtime settings, not plan payload.
_TASK_PLAN_META_KEYS = frozenset(
    {
        "name",
        "plan_from_prompt",
        "semantic_intent_injection",
        "model_input_resolution",
        "output_resolution",
        "instruction",
        "episode_prompt",
        "reset_detector",
    }
)


def nested_task_plan_defaults(config: dict[str, Any] | None, task_name: str) -> dict[str, Any]:
    """Non-meta fields under ``task.<name>`` used as default episode plan payload."""
    cfg = task_config(config or {}, task_name)
    return {
        key: value
        for key, value in cfg.items()
        if key not in _TASK_PLAN_META_KEYS and not str(key).startswith("_")
    }


def resolve_episode_task_context(
    config: dict[str, Any] | None,
    *,
    task_name: str,
    task_context: Any | None = None,
    episode_prompt: str | None = None,
    parse_plan: Any | None = None,
) -> Any | None:
    """Resolve episode ``TaskContext`` from client input and YAML defaults.

    Priority:
    1. Structured ``task_context.payload`` (always), if ``parse_plan`` accepts it
       or no parser is registered.
    2. Free-text instruction / ``episode_prompt`` when ``plan_from_prompt`` is on
       and the text yields a complete plan (or no parser is registered).
    3. Non-meta ``task.<name>`` YAML fields as payload defaults.

    ``parse_plan`` is optional and task-specific: ``callable(raw) -> TaskContext | None``.
    Return ``None`` when ``raw`` is incomplete so the resolver can fall through.
    Without a parser, client context / prompt is kept as-is and YAML defaults are
    attached as payload when present.
    """
    from common.task_types import TaskContext

    use_prompt = plan_from_prompt_enabled(config, task_name=task_name)
    nested = task_config(config or {}, task_name)
    yaml_instruction = str(nested.get("instruction") or nested.get("episode_prompt") or "").strip()
    defaults = nested_task_plan_defaults(config, task_name)

    def _via_parser(raw: Any, *, source: str) -> Any | None:
        if parse_plan is None:
            return None
        parsed = parse_plan(raw)
        if parsed is None:
            return None
        instruction = str(getattr(parsed, "instruction", "") or "").strip()
        payload = dict(getattr(parsed, "payload", None) or {})
        return TaskContext(instruction=instruction, payload=payload, source=source)

    if task_context is not None:
        payload = dict(getattr(task_context, "payload", None) or {})
        instruction = str(getattr(task_context, "instruction", "") or "").strip()
        source = str(getattr(task_context, "source", None) or "api")
        if payload:
            parsed = _via_parser(payload, source=source)
            if parsed is not None:
                return parsed
            if parse_plan is None:
                return task_context
        if use_prompt and instruction:
            parsed = _via_parser(instruction, source=source)
            if parsed is not None:
                return parsed
            if parse_plan is None:
                return task_context

    prompt_text = str(episode_prompt or "").strip()
    if use_prompt and prompt_text:
        parsed = _via_parser(prompt_text, source="api")
        if parsed is not None:
            return parsed
        if parse_plan is None:
            return TaskContext(instruction=prompt_text, payload=dict(defaults), source="api")

    if defaults:
        parsed = _via_parser(defaults, source="config")
        if parsed is not None:
            return parsed
        instruction = yaml_instruction or prompt_text
        if instruction or defaults:
            return TaskContext(
                instruction=instruction,
                payload=dict(defaults),
                source="config",
            )

    return task_context


def with_instruction_hint(base_prompt: str, task_instruction: str | None) -> str:
    if task_instruction and "{instruction}" in base_prompt:
        return base_prompt.format(instruction=task_instruction)
    return base_prompt


def normalize_words(text: Any) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def normalize_label(text: Any) -> str:
    """Normalize a short object/label token from VLM output."""
    normalized = normalize_words(text).strip()
    if not normalized:
        raise ValueError(f"failed to extract label from Qwen output: {text!r}")
    return normalized


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text, or {}."""
    import json
    import re

    cleaned = str(text or "").strip()
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


def parse_label_category_json(text: str) -> dict[str, str]:
    """Parse crop JSON ``{label, category}`` with plain-text label fallback."""
    payload = parse_json_object(text)
    if payload:
        label = normalize_label(payload.get("label") or payload.get("name") or "")
        raw_category = str(payload.get("category") or payload.get("type") or "").strip()
        category = normalize_words(raw_category) if raw_category else ""
        return {"label": label, "category": category}
    return {"label": normalize_label(text), "category": ""}


def without_leading_the(phrase: str) -> str:
    return str(phrase).removeprefix("the ").strip()


def single_letter_identity(value: Any) -> str | None:
    """Return ``value`` only when it is already one English letter.

    Detector phrases such as ``letter blocks`` must not be sliced to ``l``.
    """
    text = str(value or "").strip().lower()
    return text if len(text) == 1 and text.isalpha() else None


def movable_detector_labels(scene_objects: Sequence[Any] | None) -> set[str]:
    """Class names from the spec (phrase / DINO prompt / key), not instance IDs."""
    labels: set[str] = set()
    for obj in scene_objects or ():
        if getattr(obj, "role", None) != "movable":
            continue
        for raw in (
            getattr(obj, "phrase", None),
            getattr(obj, "dino_prompt", None),
            getattr(obj, "key", None),
            getattr(obj, "label", None),
        ):
            text = without_leading_the(str(raw or "")).strip().lower()
            if not text:
                continue
            labels.add(text)
            labels.add(text.replace("_", " "))
            labels.add(" ".join(text.replace(".", " ").split()))
    return {item for item in labels if item}


def has_crop_identity(obj: Any, *, generic_labels: set[str] | None = None) -> bool:
    """True when the box already has a crop-recognized name, not a detector phrase."""
    if single_letter_identity(getattr(obj, "letter", None)):
        return True
    label = str(getattr(obj, "label", None) or "").strip().lower()
    if not label:
        return False
    if label in (generic_labels or set()):
        return False
    return True


def bbox_area(bbox: list[float] | tuple[float, ...]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_center_x(bbox: list[float] | tuple[float, ...]) -> float:
    return (float(bbox[0]) + float(bbox[2])) * 0.5


def detection_area(det: dict[str, Any]) -> float:
    return bbox_area(det["bbox_xyxy"])


def detection_center_x(det: dict[str, Any]) -> float:
    return bbox_center_x(det["bbox_xyxy"])


def resolve_reset_detector(config: dict[str, Any] | None = None) -> str:
    """Return the configured POST /reset object detector."""
    return str(config.get("grounding", {}).get("reset_detector", "dino")).lower()


def _grounding_defaults(config: dict[str, Any] | None) -> dict[str, float]:
    grounding = config.get("grounding", {})
    return {
        "box_threshold": float(grounding.get("box_threshold", 0.22)),
        "text_threshold": float(grounding.get("text_threshold", 0.18)),
        "max_box_area_ratio": float(grounding.get("max_box_area_ratio", 0.5)),
    }


def _multi_grounding_prompt(phrase: str, *, output_mode: str = "json") -> str:
    target = without_leading_the(phrase)
    base = (
        f"Locate every individual visible object matching this category: '{target}'. "
        "Return one tight bounding box for each separate physical object. "
        "Never return one box around a group or collection of objects. Do not box "
        "unrelated containers, holders, the table, or background. "
    )
    mode = str(output_mode or "json").lower()
    if mode == "flat_array":
        return (
            base
            + "Output only a compact nested array "
            "[[x1,y1,x2,y2], ...] with coordinates normalized to 0-1000. "
            "If no individual object matches, output []."
        )
    if mode != "json":
        raise ValueError(f"unsupported Qwen multi grounding output_mode: {mode!r}")
    return (
        base
        + "Output only a compact JSON array of bounding boxes in the form "
        '[{"bbox_2d":[x1,y1,x2,y2]}, ...] with coordinates normalized to 0-1000. '
        "If no individual object matches, output []."
    )


def _qwen_boxes(
    image: Any,
    phrase: str,
    *,
    multi: bool = False,
    max_new_tokens: int | None = None,
    output_mode: str = "json",
) -> list[tuple[float, float, float, float]]:
    try:
        if multi:
            from perception.task_manager.qwen_prompt import run_vlm_prompt

            result = run_vlm_prompt(
                image,
                _multi_grounding_prompt(phrase, output_mode=output_mode),
                output="boxes",
                max_new_tokens=max_new_tokens or 512,
            )
            if not result.boxes:
                _logger.warning(
                    "Qwen reset multi grounding returned no parseable boxes "
                    "for phrase=%r raw=%r",
                    phrase,
                    result.raw_text,
                )
            return result.boxes
        box = qwen_box(image, phrase)
        return [box] if box is not None else []
    except Exception:
        _logger.exception("Qwen reset grounding failed for phrase=%r", phrase)
        return []


def _detection_dict(bbox: list[float], *, confidence: float | None = None, det_id: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": det_id, "bbox_xyxy": bbox}
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


def reset_ground_bbox(
    *,
    image: Any,
    phrase: str,
    config: dict[str, Any] | None = None,
    dino_client: Any = None,
    dino_prompt: str | None = None,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
    max_box_area_ratio: float | None = None,
) -> tuple[list[float] | None, dict[str, Any] | None]:
    """Reset-time detection via Qwen grounding or GroundingDINO.

    ``phrase`` is the natural-language target (e.g. ``the green rectangular block``).
    When ``reset_detector`` is ``dino``, ``dino_prompt`` may supply DINO synonym
    prompts; thresholds fall back to ``grounding.*`` defaults.
    """

    phrase = (phrase or "").strip()
    if not phrase:
        return None, None

    reset_detector = resolve_reset_detector(config)
    if reset_detector == "qwen":
        box = qwen_box(image, phrase)
        return (list(box) if box is not None else None), None

    if dino_client is None:
        raise ValueError("dino_client is required when reset_detector is dino")

    defaults = _grounding_defaults(config)
    return detect_bbox(
        dino_client=dino_client,
        image=image,
        prompt=dino_prompt or without_leading_the(phrase),
        box_threshold=float(box_threshold if box_threshold is not None else defaults["box_threshold"]),
        text_threshold=float(text_threshold if text_threshold is not None else defaults["text_threshold"]),
        max_box_area_ratio=float(
            max_box_area_ratio if max_box_area_ratio is not None else defaults["max_box_area_ratio"]
        ),
    )


def reset_ground_boxes(
    *,
    image: Any,
    phrase: str,
    config: dict[str, Any] | None = None,
    dino_client: Any = None,
    dino_prompt: str | None = None,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
    max_box_area_ratio: float | None = None,
    keep_top_k: int | None = None,
    target_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Reset-time multi-instance detection via Qwen grounding or GroundingDINO."""

    phrase = (phrase or "").strip()
    if not phrase:
        return []

    reset_detector = resolve_reset_detector(config)
    if reset_detector == "qwen":
        qwen_cfg = config.get("grounding", {}).get("qwen", {})
        output_mode = str(qwen_cfg.get("output_mode", "json"))
        detections = [
            _detection_dict(list(box), det_id=idx)
            for idx, box in enumerate(
                _qwen_boxes(
                    image,
                    phrase,
                    multi=True,
                    max_new_tokens=int(qwen_cfg.get("reset_multi_max_new_tokens", 512)),
                    output_mode=output_mode,
                )
            )
        ]
        unfiltered_count = len(detections)
        if max_box_area_ratio is not None:
            width, height = image.size
            max_area = width * height * float(max_box_area_ratio)
            detections = [det for det in detections if detection_area(det) <= max_area]
        if unfiltered_count and not detections:
            _logger.warning(
                "all %d Qwen reset detections for phrase=%r were rejected by "
                "max_box_area_ratio=%s",
                unfiltered_count,
                phrase,
                max_box_area_ratio,
            )
        return detections

    if dino_client is None:
        raise ValueError("dino_client is required when reset_detector is dino")

    defaults = _grounding_defaults(config)
    detections = dino_client.detect_objects(
        image,
        dino_prompt or without_leading_the(phrase),
        box_threshold=float(box_threshold if box_threshold is not None else defaults["box_threshold"]),
        text_threshold=float(text_threshold if text_threshold is not None else defaults["text_threshold"]),
        max_box_area_ratio=float(
            max_box_area_ratio if max_box_area_ratio is not None else defaults["max_box_area_ratio"]
        ),
        keep_top_k=keep_top_k,
    )
    return list(detections)


def detect_one(
    dino_client: Any,
    image: Any,
    prompt: str,
    *,
    box_threshold: float,
    text_threshold: float,
    max_box_area_ratio: float,
) -> dict[str, Any] | None:
    detections = dino_client.detect_objects(
        image,
        prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        max_box_area_ratio=max_box_area_ratio,
        keep_top_k=1,
    )
    return detections[0] if detections else None


def qwen_box(image: Any, prompt: str) -> tuple[float, float, float, float] | None:
    try:
        from perception.semantic_grounder.qwen import qwen_ground_phrase

        return qwen_ground_phrase(image, prompt)
    except Exception:
        return None


def bbox_from_detection_or_qwen(detection: dict[str, Any] | None, image: Any, prompt: str):
    if detection is not None and detection.get("bbox_xyxy") is not None:
        return list(detection["bbox_xyxy"])
    box = qwen_box(image, prompt)
    return list(box) if box is not None else None


def detect_bbox(
    *,
    dino_client: Any,
    image: Any,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    max_box_area_ratio: float,
) -> tuple[list[float] | None, dict[str, Any] | None]:
    det = detect_one(
        dino_client,
        image,
        prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        max_box_area_ratio=max_box_area_ratio,
    )
    return bbox_from_detection_or_qwen(det, image, prompt), det


def resolve_pattern_target_phrase(
    *,
    stage_name: str,
    free_phrase: str,
    contact_phrase: str,
    base_prompt: str,
) -> str:
    """Phrase for the active subtask under the current action pattern."""
    group = action_pattern_group(stage_name)
    if group == "free" and free_phrase:
        return free_phrase
    if group == "contact" and contact_phrase:
        return contact_phrase
    return base_prompt
