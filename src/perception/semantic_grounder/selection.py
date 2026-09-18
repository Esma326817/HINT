"""Shared target-grounding selection for offline rendering and online inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from common.vision.crop import crop_xyxy
from pattern.runtime.types import (
    DEXTEROUS_CONTACT_STAGE,
    FREE_MOVE_STAGE,
    GLOBAL_CAMERA,
    LEFT_WRIST_CAMERA,
    PRE_CONTACT_STAGE,
    RIGHT_WRIST_CAMERA,
    TRANSPORT_CONTACT_STAGE,
)

WRIST_VERIFY_STAGES = (PRE_CONTACT_STAGE, DEXTEROUS_CONTACT_STAGE)
WRIST_CAMERAS = (LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA)


def free_move_grounding_delay_frames(config: dict[str, Any]) -> int:
    """Configured frames to wait for the arm to clear the global view."""
    stage_aware = config.get("stage_aware", {}) or {}
    return max(0, int(stage_aware.get("free_move_grounding_delay_frames", 0)))


def seed_transport_from_reset(config: dict[str, Any] | None) -> bool | None:
    """Whether transport should reuse the reset-time placement bbox.

    Returns ``None`` when unset so task hooks remain the default. Explicit
    ``stage_aware.seed_transport_from_reset: true|false`` overrides the seed
    hook and, when ``true``, also forces ``area_placement`` render mode.
    """
    stage_aware = (config or {}).get("stage_aware", {}) or {}
    if "seed_transport_from_reset" not in stage_aware:
        return None
    return bool(stage_aware.get("seed_transport_from_reset"))


def should_defer_free_move_grounding(
    *,
    config: dict[str, Any],
    stage_name: str | None,
    frames_in_stage: int,
) -> bool:
    """Whether grounding should wait at this position in a free-move run."""
    return (
        stage_name == FREE_MOVE_STAGE
        and frames_in_stage < free_move_grounding_delay_frames(config)
    )


@dataclass(frozen=True)
class GroundingSelection:
    bbox_xyxy: tuple[float, float, float, float] | None
    source: str
    detection: dict[str, Any] | None = None
    matched_label: str | None = None
    matched_category: str | None = None
    verification_mode: str | None = None


def _known_target_bbox(task_state: Any, stage_name: str | None):
    if task_state is None:
        return None
    if stage_name == TRANSPORT_CONTACT_STAGE:
        from task.operations import resolve_target_placement

        target = resolve_target_placement(task_state)
    else:
        from task.operations import resolve_target_block

        target = resolve_target_block(task_state)
    if target is None or not getattr(target, "bbox_xyxy", None):
        return None
    box = tuple(float(value) for value in target.bbox_xyxy)
    return box if len(box) == 4 else None


def _board_region_bbox(task_state: Any):
    placements = getattr(task_state, "placements", None) or []
    boxes = [
        tuple(float(value) for value in placement.bbox_xyxy)
        for placement in placements
        if getattr(placement, "bbox_xyxy", None) is not None
        and len(placement.bbox_xyxy) == 4
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _task_prefers_known_seed(
    task_state: Any,
    config: dict[str, Any],
    stage_name: str | None,
) -> bool:
    if task_state is None or not stage_name:
        return False
    if stage_name == TRANSPORT_CONTACT_STAGE:
        configured = seed_transport_from_reset(config)
        if configured is not None:
            return configured
    from task import get_task_handler

    handler = get_task_handler(getattr(task_state, "task_name", None), config)
    hook = getattr(handler, "seed_global_before_grounding", None)
    return bool(hook(task_state, stage_name)) if callable(hook) else False


def transport_area_placement_enabled(
    *,
    config: dict[str, Any] | None,
    task_state: Any,
    camera: str,
    stage_name: str | None,
) -> bool:
    """Whether global transport should paint the reset placement as a fixed area."""
    if camera != GLOBAL_CAMERA or stage_name != TRANSPORT_CONTACT_STAGE:
        return False
    configured = seed_transport_from_reset(config)
    if configured is not None:
        return configured
    if task_state is None:
        return False
    from task import get_task_handler

    handler = get_task_handler(getattr(task_state, "task_name", None), config)
    hook = getattr(handler, "resolve_segment_render_mode", None)
    if not callable(hook):
        return False
    mode = str(
        hook(task_state, camera=camera, stage_name=stage_name or "", prompt="") or "sam2"
    ).lower()
    return mode == "area_placement"


def _task_specific_bbox(
    *,
    image: Image.Image,
    dino_client: Any,
    config: dict[str, Any],
    task_state: Any,
    stage_name: str | None,
    max_new_tokens: int,
):
    if task_state is None or not stage_name:
        return None
    from task import get_task_handler

    handler = get_task_handler(getattr(task_state, "task_name", None), config)
    hook = getattr(handler, "resolve_global_grounding_box", None)
    if not callable(hook):
        return None
    raw = hook(
        image=image,
        dino_client=dino_client,
        config=config,
        task_state=task_state,
        stage_name=stage_name,
        max_new_tokens=max_new_tokens,
    )
    if raw is None:
        return None
    box = tuple(float(value) for value in raw)
    return box if len(box) == 4 else None


def _exact_label_match(wanted: str, observed: str) -> bool:
    from task.base import normalize_label

    try:
        return normalize_label(wanted) == normalize_label(observed)
    except ValueError:
        return False


def _category_fallback_for_target(task_state: Any):
    """Return the task-declared finite-category fallback and active category."""
    if task_state is None:
        return None, ""
    from task.operations import current_target_category
    from task.spec import load_task_spec

    task_name = str(getattr(task_state, "task_name", None) or "")
    if not task_name:
        return None, ""
    fallback = load_task_spec(task_name).category_fallback
    return fallback, current_target_category(task_state)


def _match_details(match: Any) -> tuple[str | None, str | None, str | None]:
    """Adapt rich robust matches and legacy string verifier callbacks."""
    if match is None:
        return None, None, None
    if isinstance(match, str):
        return match, None, "label"
    return (
        str(getattr(match, "observed_label", "") or "") or None,
        str(getattr(match, "observed_category", "") or "") or None,
        str(getattr(match, "mode", "") or "") or None,
    )


def build_wrist_verify_fn(
    *,
    task_state: Any,
    config: dict[str, Any] | None = None,
) -> Callable[[Image.Image, str], Any | None]:
    """Recognize a candidate so grounding can rank rather than reject boxes."""
    from perception.semantic_grounder.robust import (
        recognize_strict_then_category,
        recognize_target_candidate,
    )
    from perception.semantic_grounder.recognizer import recognize_crop_fields

    task_name = getattr(task_state, "task_name", None)
    fallback, expected_category = _category_fallback_for_target(task_state)
    allowed_categories = fallback.classify.labels if fallback is not None else ()
    recognize_candidate = (
        recognize_target_candidate if fallback is not None else recognize_strict_then_category
    )

    def verify(crop: Image.Image, verify_target: str) -> Any | None:
        return recognize_candidate(
            crop,
            recognize_fields=lambda image: recognize_crop_fields(
                image, task_name=task_name, config=config
            ),
            expected_label=verify_target,
            expected_category=expected_category,
            allowed_categories=allowed_categories,
        )

    return verify


def _verify_box_label(
    *,
    image: Image.Image,
    box: tuple[float, float, float, float],
    verify_target: str,
    task_state: Any,
    config: dict[str, Any],
    recognition_padding: int,
    verify_fn: Callable[[Image.Image, str], Any | None] | None = None,
) -> Any | None:
    """Crop-recognize a box; return observed label when the task accepts it."""
    if verify_fn is None:
        verify_fn = build_wrist_verify_fn(task_state=task_state, config=config)
    try:
        crop = crop_xyxy(image, list(box), padding=recognition_padding)
        return verify_fn(crop, verify_target)
    except Exception:  # noqa: BLE001
        return None


def resolve_grounding_selection(
    *,
    camera: str,
    image: Image.Image,
    prompt: str,
    task_state: Any,
    stage_name: str | None,
    config: dict[str, Any],
    dino_client: Any = None,
    verify_target: str | None = None,
    recognition_padding: int = 4,
    allow_vlm: bool = True,
    qwen_ground_fn: Callable[..., Any] | None = None,
) -> GroundingSelection:
    """Choose one grounding bbox identically for offline and online execution."""

    grounding_cfg = config.get("grounding", {})
    grounder = str(grounding_cfg.get("grounder", "dino")).lower()
    qwen_cfg = grounding_cfg.get("qwen", {})
    max_new_tokens = int(qwen_cfg.get("max_new_tokens", 64))
    qwen_output_mode = str(qwen_cfg.get("output_mode", "json"))

    def qwen_call_kwargs() -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "exclude_pad": float(qwen_cfg.get("board_pad", 6.0)),
        }
        # Keep injected legacy/test callbacks compatible unless the mode is
        # explicitly enabled in configuration.
        if qwen_output_mode != "json":
            kwargs["output_mode"] = qwen_output_mode
        return kwargs

    if camera == GLOBAL_CAMERA:
        if _task_prefers_known_seed(task_state, config, stage_name):
            box = _known_target_bbox(task_state, stage_name)
            if box is not None:
                return GroundingSelection(box, "known_task_seed")

        # An injected grounding function is an explicit end-to-end override
        # used by tests and alternate runtimes; do not bypass it through a
        # task hook that imports the global Qwen singleton.
        if allow_vlm and qwen_ground_fn is None:
            box = _task_specific_bbox(
                image=image,
                dino_client=dino_client,
                config=config,
                task_state=task_state,
                stage_name=stage_name,
                max_new_tokens=max_new_tokens,
            )
            if box is not None:
                return GroundingSelection(box, "task_specific")

        if allow_vlm and grounder == "qwen" and prompt:
            if stage_name == FREE_MOVE_STAGE and qwen_ground_fn is None:
                from perception.semantic_grounder.letter_instances import maybe_ground_letter_free_move

                letter_selection = maybe_ground_letter_free_move(
                    image=image,
                    task_state=task_state,
                    stage_name=stage_name,
                    config=config,
                )
                if letter_selection is not None:
                    return letter_selection
            if qwen_ground_fn is None:
                from perception.semantic_grounder.qwen import qwen_ground_phrase

                qwen_ground_fn = qwen_ground_phrase
            exclude = _board_region_bbox(task_state) if stage_name == FREE_MOVE_STAGE else None
            raw = qwen_ground_fn(
                image,
                prompt,
                exclude_bbox=exclude,
                **qwen_call_kwargs(),
            )
            if raw is not None:
                box = tuple(float(value) for value in raw)
                if len(box) == 4:
                    return GroundingSelection(box, "qwen")

        box = _known_target_bbox(task_state, stage_name)
        return GroundingSelection(box, "known_fallback" if box is not None else "missing")

    if grounder == "qwen" and not allow_vlm:
        return GroundingSelection(None, "qwen_disabled")

    qwen_failed = False
    should_verify = should_verify_wrist_grounding(
        grounding_cfg=grounding_cfg,
        camera=camera,
        stage_name=stage_name,
        verify_target=verify_target,
    )
    verify_fn = (
        build_wrist_verify_fn(task_state=task_state, config=config) if should_verify else None
    )
    category_fallback, expected_category = _category_fallback_for_target(task_state)
    if not should_verify:
        category_fallback, expected_category = None, ""
    allowed_categories = (
        category_fallback.classify.labels if category_fallback is not None else ()
    )
    category_prompt_template = (
        category_fallback.grounding_prompt
        if category_fallback is not None
        else "{category} object"
    )

    if grounder == "qwen" and prompt:
        from perception.semantic_grounder.robust import detect_with_category_fallback

        if qwen_ground_fn is None:
            from perception.semantic_grounder.qwen import qwen_ground_phrase

            qwen_ground_fn = qwen_ground_phrase
        saw_qwen_box = False

        def try_qwen(attempt_prompt: str):
            nonlocal saw_qwen_box
            raw = qwen_ground_fn(
                image,
                attempt_prompt,
                exclude_bbox=None,
                **qwen_call_kwargs(),
            )
            if raw is None:
                return None
            box = tuple(float(value) for value in raw)
            if len(box) != 4:
                return None
            saw_qwen_box = True
            if should_verify and verify_target:
                matched = _verify_box_label(
                    image=image,
                    box=box,
                    verify_target=verify_target,
                    task_state=task_state,
                    config=config,
                    recognition_padding=recognition_padding,
                    verify_fn=verify_fn,
                )
                if matched is None and category_fallback is not None:
                    # Classification tasks rank existing boxes best-effort. A
                    # sole Qwen box remains preferable to re-grounding a broad
                    # category and potentially selecting a nearby peer object.
                    return box, None
                return (box, matched) if matched is not None else None
            return box, verify_target

        qwen_result = detect_with_category_fallback(
            strict_prompt=prompt,
            detect_fn=try_qwen,
            expected_category=expected_category,
            allowed_categories=allowed_categories,
            category_prompt_template=category_prompt_template,
        )
        if qwen_result is not None:
            box, matched = qwen_result.value
            matched_label, matched_category, verification_mode = _match_details(matched)
            source = "qwen_category_fallback" if qwen_result.mode == "category" else "qwen"
            return GroundingSelection(
                box,
                source,
                matched_label=matched_label,
                matched_category=matched_category,
                verification_mode=verification_mode,
            )
        qwen_failed = True
        if not bool(grounding_cfg.get("qwen_dino_fallback", False)) or dino_client is None:
            return GroundingSelection(
                None, "qwen_verify_mismatch" if saw_qwen_box else "qwen_missing"
            )

    if dino_client is None or not prompt:
        return GroundingSelection(None, "qwen_missing" if qwen_failed else "missing_detector")

    from perception.semantic_grounder.robust import detect_with_category_fallback

    def try_dino(attempt_prompt: str):
        selection_cfg = grounding_cfg
        if category_fallback is not None and not grounding_cfg.get("verify_wrist_fallback"):
            selection_cfg = {**grounding_cfg, "verify_wrist_fallback": True}
        detection, matched = select_prompt_detection(
            dino_client,
            image,
            attempt_prompt,
            selection_cfg,
            verify_target=verify_target if should_verify else None,
            stage_name=stage_name,
            letter_padding=recognition_padding,
            verify_fn=verify_fn,
        )
        return (detection, matched) if detection is not None else None

    dino_result = detect_with_category_fallback(
        strict_prompt=prompt,
        detect_fn=try_dino,
        expected_category=expected_category,
        allowed_categories=allowed_categories,
        category_prompt_template=category_prompt_template,
    )
    if dino_result is None:
        return GroundingSelection(None, "dino_missing" if not qwen_failed else "qwen_dino_missing")
    detection, matched = dino_result.value
    raw_box = tuple(float(value) for value in detection.get("bbox_xyxy", ()))
    if len(raw_box) != 4:
        return GroundingSelection(None, "invalid_detection", detection=detection)
    source = "dino_after_qwen" if qwen_failed else "dino"
    if dino_result.mode == "category":
        source += "_category_fallback"
    matched_label, matched_category, verification_mode = _match_details(matched)
    return GroundingSelection(
        raw_box,
        source,
        detection=detection,
        matched_label=matched_label,
        matched_category=matched_category,
        verification_mode=verification_mode,
    )


def should_verify_wrist_grounding(
    *,
    grounding_cfg: dict[str, Any],
    camera: str,
    stage_name: str | None,
    verify_target: str | None,
) -> bool:
    return (
        camera in WRIST_CAMERAS
        and bool(verify_target)
        and stage_name in WRIST_VERIFY_STAGES
        and bool(grounding_cfg.get("verify_wrist_target", True))
    )


def select_prompt_detection(
    dino_client,
    image: Image.Image,
    prompt: str,
    grounding_cfg: dict[str, Any],
    *,
    verify_target: str | None = None,
    verify_target_letter: str | None = None,
    stage_name: str | None = None,
    letter_padding: int = 4,
    verify_fn: Callable[[Image.Image, str], Any | None] | None = None,
    recognize_letter_fn: Callable[[Image.Image], str] | None = None,
) -> tuple[dict[str, Any] | None, Any | None]:
    """Return the chosen DINO detection and optional verified label."""

    target = verify_target if verify_target is not None else verify_target_letter
    should_verify = (
        bool(grounding_cfg.get("verify_wrist_target", True))
        and bool(target)
        and stage_name in WRIST_VERIFY_STAGES
        and (verify_fn is not None or recognize_letter_fn is not None)
    )
    keep_top_k = int(grounding_cfg.get("keep_top_k", 1))
    if should_verify:
        keep_top_k = max(keep_top_k, int(grounding_cfg.get("verify_top_k", 4)))

    detections = dino_client.detect_objects(
        image,
        prompt,
        box_threshold=float(grounding_cfg.get("box_threshold", 0.25)),
        text_threshold=float(grounding_cfg.get("text_threshold", 0.2)),
        max_box_area_ratio=float(grounding_cfg.get("max_box_area_ratio", 0.5)),
        keep_top_k=max(1, keep_top_k),
    )
    if not detections:
        return None, None
    if not should_verify:
        return detections[0], None

    recognized: list[tuple[int, dict[str, Any], Any]] = []
    for det in detections:
        bbox = tuple(float(value) for value in det.get("bbox_xyxy", ()))
        if len(bbox) != 4:
            continue
        try:
            crop = crop_xyxy(image, list(bbox), padding=letter_padding)
            if verify_fn is not None:
                matched = verify_fn(crop, str(target))
            else:
                letter = recognize_letter_fn(crop)  # type: ignore[misc]
                matched = letter if _exact_label_match(str(target), letter) else None
        except Exception:  # noqa: BLE001
            continue
        if matched is not None:
            mode = str(getattr(matched, "mode", "label") or "label")
            priority = {"label": 2, "category": 1}.get(mode, 0)
            recognized.append((priority, det, matched))
    if recognized:
        # Detections arrive in descending confidence order. max() is stable for
        # ties, so semantic preference wins and detector confidence breaks ties.
        _priority, detection, matched = max(recognized, key=lambda item: item[0])
        return detection, matched
    if bool(grounding_cfg.get("verify_wrist_fallback", False)):
        return detections[0], None
    return None, None
