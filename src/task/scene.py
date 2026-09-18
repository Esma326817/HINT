"""Pre-task scene understanding: declare entities, detect, recognize, plan TaskState.

Used once at ``/reset`` so the rest of the pipeline can schedule the subtasks of a
long-horizon task from a stable ``TaskState`` without re-parsing the whole table
every frame.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from task.base import (
    has_crop_identity,
    movable_detector_labels,
    reset_ground_bbox,
    reset_ground_boxes,
    single_letter_identity,
    task_config,
    without_leading_the,
)
from task.hooks import TaskModule
from task.types import (
    BuiltSubtasks,
    SceneAccumulator,
    SceneLayout,
    SceneObject,
    SubtaskSource,
)

_logger = logging.getLogger(__name__)


def _bbox_center_xy(bbox_xyxy: Any) -> tuple[float, float, int, int]:
    from common.vision.bbox import bbox_to_int

    x1, y1, x2, y2 = bbox_to_int(bbox_xyxy)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), x1, y1)


_ASSIGN_ORDER_KEYS: dict[str, Callable[[float, float, int, int], tuple[float, float, int, int]]] = {
    "left_to_right": lambda cx, cy, x1, y1: (cx, cy, x1, y1),
    "right_to_left": lambda cx, cy, x1, y1: (-cx, cy, -x1, y1),
    "top_to_bottom": lambda cx, cy, x1, y1: (cy, cx, y1, x1),
    "bottom_to_top": lambda cx, cy, x1, y1: (-cy, cx, -y1, x1),
}


def _assign_order_key(order: str):
    """Sort key for a detection dict under a named spatial ``assign_order``."""
    signed = _ASSIGN_ORDER_KEYS[order]

    def key(det: dict[str, Any]) -> tuple[float, float, int, int]:
        return signed(*_bbox_center_xy(det["bbox_xyxy"]))

    return key


def _order_left_to_right(blocks: Sequence[Any]) -> list[str]:
    """Left→right by bbox, matching how the demonstrations were collected."""
    from common.vision.bbox import bbox_to_int

    def key(block: Any) -> tuple[float, int, int]:
        x1, y1, x2, _y2 = bbox_to_int(block.bbox_xyxy)
        return (0.5 * (x1 + x2), x1, y1)

    return [
        str(block.label)
        for block in sorted(
            (block for block in blocks if block.label and block.bbox_xyxy),
            key=key,
        )
    ]


def _order_as_detected(blocks: Sequence[Any]) -> list[str]:
    return [str(block.label) for block in blocks if block.label]


# Which order the detected objects are worked through. This encodes a collection
# bias of the demonstrations, not task semantics, so it is a framework policy the
# runtime config can override via ``task.<name>.subtask_order``.
SUBTASK_ORDERINGS: dict[str, Callable[[Sequence[Any]], list[str]]] = {
    "left_to_right": _order_left_to_right,
    "detected": _order_as_detected,
}


def define_task(
    *,
    spec: Any,
    prompts: Any = None,
    build_scene_layout: Callable[..., SceneLayout] | None = None,
    parse_landmark_output: Callable[..., str] | None = None,
    parse_crop_output: Callable[..., str] | None = None,
    parse_crop_result: Callable[..., dict[str, str]] | None = None,
    build_initial_state: Callable[..., Any] | None = None,
    resolve_rule_grounding_prompt: Callable[..., str] | None = None,
    refine_scene: Callable[..., Any] | None = None,
    resolve_global_grounding_box: Callable[..., Any] | None = None,
    refresh_movables: Callable[..., Any] | None = None,
    seed_global_before_grounding: Callable[..., bool] | None = None,
    capture_tracker_parent_bbox: Callable[..., Any] | None = None,
    resolve_segment_render_mode: Callable[..., str] | None = None,
    parse_episode_plan: Callable[..., Any] | None = None,
    resolve_task_context: Callable[..., Any] | None = None,
    on_subtask_advance: Callable[..., Any] | None = None,
) -> TaskModule:
    """Bind a YAML spec (and optional plugin hooks) into a runtime handler.

    A task whose YAML spec is complete only passes ``spec``: prompts, the scene
    layout, and the declared output parsers all come from it. Plugin ``HOOKS``
    override the parts that need real code.
    """

    prompts = prompts or spec.prompts()
    build_scene_layout = build_scene_layout or spec.scene_layout
    parse_landmark_output = parse_landmark_output or spec.landmark_output_parser()
    parse_crop_output = parse_crop_output or spec.crop_output_parser()
    parse_crop_result = parse_crop_result or spec.crop_result_parser()
    if seed_global_before_grounding is None and spec.known_bbox_seed_rules:
        seed_global_before_grounding = spec.prefers_known_bbox_seed
    if resolve_segment_render_mode is None and spec.segment_render_rules:
        resolve_segment_render_mode = spec.resolve_segment_render_mode
    custom_refine = refine_scene
    if spec.read_from is not None or spec.precise_ground is not None:
        def refine_scene(
            task_state: Any,
            *,
            image: Any,
            dino_client: Any,
            config: dict[str, Any],
            recognition_padding: int,
            task_context: Any = None,
        ) -> Any:
            from task.behaviors import apply_precise_ground, apply_read_from

            if spec.read_from is not None:
                task_state = apply_read_from(
                    task_state, spec=spec, image=image, config=config
                )
            if spec.precise_ground is not None:
                task_state = apply_precise_ground(task_state, spec=spec)
            if custom_refine is None:
                return task_state
            return custom_refine(
                task_state,
                image=image,
                dino_client=dino_client,
                config=config,
                recognition_padding=recognition_padding,
                task_context=task_context,
            )
    if refresh_movables is None and spec.exclude_inside:
        from task.behaviors import refresh_movables_from_spec

        def refresh_movables(
            *,
            image: Any,
            dino_client: Any,
            config: dict[str, Any],
            task_state: Any,
            recognition_padding: int,
        ) -> Any:
            return refresh_movables_from_spec(
                spec=spec,
                image=image,
                dino_client=dino_client,
                config=config,
                task_state=task_state,
                recognition_padding=recognition_padding,
            )

    handler_box: dict[str, TaskModule] = {}

    def _default_parse_landmark(text: str) -> str:
        del text
        return prompts.name

    def _default_parse_crop(text: str) -> str:
        if parse_crop_result is not None:
            return parse_crop_result(text)["label"]
        from task.base import normalize_label

        return normalize_label(text)

    def _default_build_initial_state(
        *,
        image: Any,
        dino_client: Any,
        config: dict[str, Any],
        recognition_padding: int,
        task_context: Any | None = None,
    ) -> Any:
        return understand_scene(
            image=image,
            dino_client=dino_client,
            config=config,
            recognition_padding=recognition_padding,
            handler=handler_box["handler"],
            task_context=task_context,
        )

    def _default_resolve_rule_grounding_prompt(
        *,
        decision: Any,
        task_state: Any,
        task_instruction: str | None,
        base_prompt: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        from common.task_types import TaskContext
        from task.base import (
            resolve_pattern_target_phrase,
            with_instruction_hint,
            without_spatial_modifiers,
        )

        hinted = with_instruction_hint(base_prompt, task_instruction)
        # Keep the live runtime config so YAML plan defaults (e.g. peg_in_hole
        # selected_block_color / selected_peg_shape) remain available on /step.
        layout_config: dict[str, Any] = dict(config or {})
        if task_instruction:
            task_section = dict(layout_config.get("task") or {})
            nested = dict(task_section.get(prompts.name) or {})
            nested.setdefault("instruction", task_instruction)
            task_section[prompts.name] = nested
            layout_config["task"] = task_section

        stored_context = None if task_state is None else task_state.task_context
        meta = None if task_state is None else task_state.metadata
        if meta:
            payload = dict(stored_context.payload) if stored_context is not None else {}
            for key in spec.context_fields:
                value = meta.get(key)
                if key not in payload and value not in (None, "", "unknown"):
                    payload[key] = value
            if payload:
                stored_context = TaskContext(
                    instruction=(
                        str(stored_context.instruction if stored_context is not None else "").strip()
                        or task_instruction
                        or str(meta.get("instruction") or "")
                    ),
                    payload=payload,
                    source=str(
                        stored_context.source if stored_context is not None else "metadata"
                    ),
                )

        layout = build_scene_layout(layout_config, task_context=stored_context)

        if task_state is None:
            # Before reset only a declared plan can name the first subtask.
            if not layout.subtasks.steps:
                return hinted
            step = layout.subtasks.steps[0]
            label, category, placement = step.label, step.category, step.placement
        else:
            from task.operations import (
                current_target_category,
                current_target_label,
                current_target_placement_label,
            )

            label = current_target_label(task_state)
            category = current_target_category(task_state)
            placement = current_target_placement_label(task_state)

        fields = {
            "label": label,
            "category": category,
            "placement": placement.replace("_", " ") if placement else "",
        }
        free_phrase = layout.free_target_phrase.format(**fields) if label else ""
        contact_phrase = ""
        if placement:
            contact_phrase = layout.contact_target_phrase.format(**fields)
        elif "{" not in layout.contact_target_phrase:
            # Constant contact phrase (e.g. letter board slots without placement labels).
            contact_phrase = layout.contact_target_phrase

        phrase = resolve_pattern_target_phrase(
            stage_name=decision.confirmed_stage,
            free_phrase=free_phrase,
            contact_phrase=contact_phrase,
            base_prompt=hinted,
        )
        if decision.confirmed_stage in spec.strip_spatial_on_stages:
            phrase = without_spatial_modifiers(phrase)
        return phrase

    def _default_resolve_task_context(
        *,
        config: dict[str, Any] | None = None,
        task_context: Any | None = None,
        episode_prompt: str | None = None,
    ) -> Any | None:
        from task.base import resolve_episode_task_context

        return resolve_episode_task_context(
            config,
            task_name=prompts.name,
            task_context=task_context,
            episode_prompt=episode_prompt,
            parse_plan=parse_episode_plan,
        )

    handler = TaskModule(
        prompts=prompts,
        parse_landmark_output=parse_landmark_output or _default_parse_landmark,
        parse_crop_output=parse_crop_output or _default_parse_crop,
        parse_crop_result=parse_crop_result,
        build_scene_layout=build_scene_layout,
        build_initial_state=build_initial_state or _default_build_initial_state,
        resolve_rule_grounding_prompt=resolve_rule_grounding_prompt or _default_resolve_rule_grounding_prompt,
        refine_scene=refine_scene,
        resolve_global_grounding_box=resolve_global_grounding_box,
        refresh_movables=refresh_movables,
        seed_global_before_grounding=seed_global_before_grounding,
        capture_tracker_parent_bbox=capture_tracker_parent_bbox,
        resolve_segment_render_mode=resolve_segment_render_mode,
        parse_episode_plan=parse_episode_plan,
        resolve_task_context=resolve_task_context or _default_resolve_task_context,
        on_subtask_advance=on_subtask_advance,
    )
    handler_box["handler"] = handler
    return handler


def _exclusion_boxes(
    layout: SceneLayout, scene: SceneAccumulator, placements: Sequence[Any]
) -> list[Any]:
    if layout.exclude_inside:
        boxes = []
        for key in layout.exclude_inside:
            box = scene.metadata.get(f"{key}_bbox")
            if box is None:
                box = next(
                    (
                        list(placement.bbox_xyxy)
                        for placement in placements
                        if placement.label == key
                    ),
                    None,
                )
            if box is not None:
                boxes.append(box)
        return boxes
    if layout.exclude_movables_inside_placements:
        return [placement.bbox_xyxy for placement in placements]
    return []


def understand_scene(
    *,
    image: Any,
    dino_client: Any,
    config: dict[str, Any],
    recognition_padding: int,
    handler: Any,
    task_context: Any | None = None,
) -> Any:
    """Understand the global scene once at reset and return a planned ``TaskState``.

    Flow: locate ``SceneObject`` boxes → recognize movables → build pick/place
    sequence → optional ``refine_scene`` hook → ``plan_next_target``.
    """

    from common.vision.bbox import is_bbox_center_inside
    from task.operations import plan_next_target, recognize_task_objects
    from common.task_types import TargetObject, PlacementCandidate, TaskState

    layout: SceneLayout = handler.build_scene_layout(config, task_context=task_context)
    task_name = layout.name

    landmark_word = (layout.target_word or "").strip()
    scene = SceneAccumulator(metadata=dict(layout.metadata))

    located: dict[str, list[dict[str, Any]]] = {}
    for scene_object in layout.scene_objects:
        located[scene_object.key] = locate_scene_object(
            scene_object,
            image=image,
            dino_client=dino_client,
            config=config,
        )

    soft_failures: list[str] = []
    for scene_object in layout.scene_objects:
        detections = located[scene_object.key]
        if not detections and scene_object.missing_message:
            soft_failures.append(scene_object.missing_message)
        if not detections and scene_object.fallback_bbox_from:
            fallback = located.get(scene_object.fallback_bbox_from) or []
            if fallback:
                detections = [dict(fallback[0])]
        if not scene_object.optional and not detections:
            raise RuntimeError(f"required scene object {scene_object.key!r} not detected during reset")

        SCENE_ROLE_HANDLERS[scene_object.role](
            scene,
            scene_object,
            detections,
            image=image,
            layout=layout,
        )

    if soft_failures:
        reason = "; ".join(soft_failures)
        scene.metadata["reset_failure"] = reason
        _logger.warning("scene reset soft-fail task=%s: %s", task_name, reason)

    placements: list[PlacementCandidate] = scene.placements
    movable_dets = _dedupe_detections(scene.movable_detections)
    container_boxes = _exclusion_boxes(layout, scene, placements)
    if container_boxes:
        movable_dets = [
            det
            for det in movable_dets
            if not any(is_bbox_center_inside(box, det["bbox_xyxy"]) for box in container_boxes)
        ]

    detected_objects = [
        TargetObject(
            id=idx,
            bbox_xyxy=list(det["bbox_xyxy"]),
            confidence=float(det["confidence"]) if det.get("confidence") is not None else None,
            letter=single_letter_identity(det.get("letter"))
            or single_letter_identity(det.get("label")),
            label=str(det.get("label") or det.get("letter") or "") or None,
            category=str(det.get("category") or "") or None,
        )
        for idx, det in enumerate(movable_dets)
    ]

    recognize_crops = layout.recognize_movables and any(
        obj.recognize_crop for obj in layout.scene_objects if obj.role == "movable"
    )
    generic_labels = movable_detector_labels(layout.scene_objects)
    if recognize_crops and detected_objects:
        # Detector/phrase labels ("letter blocks") are class names, not crop IDs.
        if not all(has_crop_identity(obj, generic_labels=generic_labels) for obj in detected_objects):
            blocks = recognize_task_objects(
                image=image,
                detected_objects=detected_objects,
                padding=recognition_padding,
                task_name=task_name,
                config=config,
            )
        else:
            blocks = detected_objects
    else:
        blocks = detected_objects

    subtasks = _build_subtasks(
        layout=layout,
        blocks=blocks,
        landmark_word=landmark_word,
        config=config,
        task_name=task_name,
    )
    scene.metadata["subtask_source"] = subtasks.source

    task_state = TaskState(
        task_name=task_name,
        task_context=task_context,
        blocks=blocks,
        placements=placements,
        target_word=subtasks.target_word or task_name,
        target_labels=subtasks.labels,
        target_categories=subtasks.categories,
        target_placements=subtasks.placements,
        progress_idx=0,
        target_block_id=None,
        target_placement_id=None,
        picked_block_ids=[],
        placed_block_ids=[],
        metadata=scene.metadata,
    )

    if handler.refine_scene is not None:
        task_state = handler.refine_scene(
            task_state,
            image=image,
            dino_client=dino_client,
            config=config,
            recognition_padding=recognition_padding,
            task_context=task_context,
        )

    task_state.target_block_id, task_state.target_placement_id = plan_next_target(task_state)
    return task_state


def locate_scene_object(
    scene_object: SceneObject,
    *,
    image: Any,
    dino_client: Any,
    config: dict[str, Any],
    **detect_kwargs: Any,
) -> list[dict[str, Any]]:
    """Ground one scene object; ``detect_kwargs`` reach the detector unchanged."""
    if scene_object.locator is not None:
        boxes = scene_object.locator.locate(image, config=config)
        if scene_object.select_label:
            box = boxes.get(scene_object.select_label)
            return [] if box is None else [{"bbox_xyxy": box, "confidence": None}]
        return [
            {"bbox_xyxy": box, "confidence": None, "label": label}
            for label, box in boxes.items()
        ]

    phrase = (scene_object.phrase or "").strip()
    if not phrase and not scene_object.dino_prompt:
        return []

    if scene_object.boxes == "all":
        return reset_ground_boxes(
            image=image,
            phrase=phrase or (scene_object.dino_prompt or ""),
            config=config,
            dino_client=dino_client,
            dino_prompt=scene_object.dino_prompt,
            box_threshold=scene_object.box_threshold,
            text_threshold=scene_object.text_threshold,
            max_box_area_ratio=scene_object.max_box_area_ratio,
            keep_top_k=scene_object.keep_top_k,
            **detect_kwargs,
        )

    bbox, det = reset_ground_bbox(
        image=image,
        phrase=phrase or (scene_object.dino_prompt or ""),
        config=config,
        dino_client=dino_client,
        dino_prompt=scene_object.dino_prompt,
        box_threshold=scene_object.box_threshold,
        text_threshold=scene_object.text_threshold,
        max_box_area_ratio=scene_object.max_box_area_ratio,
        **detect_kwargs,
    )
    if bbox is None:
        return []
    if det is not None:
        return [det]
    return [{"bbox_xyxy": bbox, "confidence": None}]


def _collect_placement(
    scene: SceneAccumulator,
    scene_object: SceneObject,
    detections: list[dict[str, Any]],
    *,
    image: Any,
    layout: SceneLayout,
) -> None:
    from common.task_types import PlacementCandidate

    key_bbox: list[float] | None = None
    for placement_label, det in _label_detections(
        image=image,
        detections=detections,
        scene_object=scene_object,
        layout=layout,
    ):
        box = list(det["bbox_xyxy"])
        scene.placements.append(
            PlacementCandidate(
                id=len(scene.placements),
                bbox_xyxy=box,
                label=placement_label,
            )
        )
        scene.metadata[f"{placement_label}_bbox"] = box
        if key_bbox is None:
            key_bbox = box
    if key_bbox is not None:
        scene.metadata.setdefault(f"{scene_object.key}_bbox", key_bbox)

    required_labels = list(scene_object.assign_labels)
    if scene_object.classify and not scene_object.optional:
        classify_spec = layout.classify[scene_object.classify]
        template = scene_object.label_template or "{label}"
        required_labels = [template.format(label=label) for label in classify_spec.labels]
    if not scene_object.optional and required_labels:
        missing = [
            label
            for label in required_labels
            if not any(placement.label == label for placement in scene.placements)
        ]
        if missing:
            raise RuntimeError(f"required placement labels not found during reset: {missing}")


def _collect_movable(
    scene: SceneAccumulator,
    scene_object: SceneObject,
    detections: list[dict[str, Any]],
    *,
    image: Any,
    layout: SceneLayout,
) -> None:
    del image, layout
    for det in detections:
        payload = dict(det)
        if not payload.get("label"):
            if scene_object.label:
                payload["label"] = scene_object.label
            elif scene_object.boxes == "one":
                payload["label"] = without_leading_the(scene_object.phrase or "")
        if scene_object.category and not payload.get("category"):
            payload["category"] = scene_object.category
        scene.movable_detections.append(payload)


def _collect_landmark(
    scene: SceneAccumulator,
    scene_object: SceneObject,
    detections: list[dict[str, Any]],
    *,
    image: Any,
    layout: SceneLayout,
) -> None:
    del image, layout
    if detections:
        scene.metadata[f"{scene_object.key}_bbox"] = list(detections[0]["bbox_xyxy"])


SCENE_ROLE_HANDLERS: dict[str, Callable[..., None]] = {
    "placement": _collect_placement,
    "movable": _collect_movable,
    "landmark": _collect_landmark,
}


def _label_detections(
    *,
    image: Any,
    detections: list[dict[str, Any]],
    scene_object: SceneObject,
    layout: SceneLayout,
) -> list[tuple[str, dict[str, Any]]]:
    """Attach placement labels; classify crops or assign by spatial order."""
    from common.vision.geometry import classify_crop_label, dedupe_detections
    from task.base import detection_area

    default_label = scene_object.label or scene_object.key
    if scene_object.assign_labels:
        candidates = [
            det
            for det in dedupe_detections(detections, iou_threshold=0.4)
            if det.get("bbox_xyxy") is not None
        ]
        ordered = sorted(
            candidates,
            key=_assign_order_key(scene_object.assign_order),
        )[: len(scene_object.assign_labels)]
        return list(zip(scene_object.assign_labels, ordered, strict=False))

    if not scene_object.classify:
        return [(default_label, det) for det in detections if det.get("bbox_xyxy") is not None]

    classify_spec = layout.classify[scene_object.classify]

    candidates = dedupe_detections(detections, iou_threshold=0.4)
    candidates = sorted(candidates, key=detection_area, reverse=True)[: len(classify_spec.labels)]
    template = scene_object.label_template or "{label}"
    labeled: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for det in candidates:
        color = classify_crop_label(image, list(det["bbox_xyxy"]), classify_spec)
        placement_label = template.format(label=color)
        if placement_label in seen:
            continue
        seen.add(placement_label)
        labeled.append((placement_label, det))
    return labeled


def _build_subtasks(
    *,
    layout: SceneLayout,
    blocks: list[Any],
    landmark_word: str,
    config: dict[str, Any],
    task_name: str,
) -> BuiltSubtasks:
    """Build the episode subtask sequence with the configured source strategy."""
    return SUBTASK_BUILDERS[layout.subtasks.source](
        layout, blocks, landmark_word, config, task_name
    )


def _build_fixed_subtasks(
    layout: SceneLayout,
    blocks: list[Any],
    landmark_word: str,
    config: dict[str, Any],
    task_name: str,
) -> BuiltSubtasks:
    del blocks, config
    return BuiltSubtasks(
        labels=[step.label for step in layout.subtasks.steps],
        categories=[step.category for step in layout.subtasks.steps],
        placements=[step.placement for step in layout.subtasks.steps],
        target_word=landmark_word or task_name,
        source="fixed",
    )


def _build_word_subtasks(
    layout: SceneLayout,
    blocks: list[Any],
    landmark_word: str,
    config: dict[str, Any],
    task_name: str,
) -> BuiltSubtasks:
    del task_name
    word = landmark_word or layout.target_word or ""
    labels = list(word)
    categories, placements = _categories_and_placements(labels, blocks, config, layout)
    return BuiltSubtasks(labels, categories, placements, word, "word_chars")


def _build_detected_subtasks(
    layout: SceneLayout,
    blocks: list[Any],
    landmark_word: str,
    config: dict[str, Any],
    task_name: str,
) -> BuiltSubtasks:
    plan = layout.subtasks
    cfg = task_config(config, task_name)
    labels = _configured_sequence(cfg, plan.order_config_key)
    if labels:
        source = "config"
    else:
        order = str(cfg.get("subtask_order") or plan.order)
        ordering = SUBTASK_ORDERINGS[order]
        labels = [label for label in ordering(blocks) if label]
        source = f"detected:{order}"

    categories, placements = _categories_and_placements(labels, blocks, config, layout)
    return BuiltSubtasks(
        labels=labels,
        categories=categories,
        placements=placements,
        target_word=landmark_word or task_name,
        source=source,
    )


SubtaskBuilder = Callable[
    [SceneLayout, list[Any], str, dict[str, Any], str],
    BuiltSubtasks,
]

SUBTASK_BUILDERS: dict[SubtaskSource, SubtaskBuilder] = {
    "fixed": _build_fixed_subtasks,
    "word_chars": _build_word_subtasks,
    "detected": _build_detected_subtasks,
}


def _categories_and_placements(
    labels: list[str],
    blocks: list[Any],
    config: dict[str, Any],
    layout: SceneLayout,
) -> tuple[list[str], list[str]]:
    by_label = {(block.label or "").lower(): block for block in blocks if block.label}
    categories: list[str] = []
    placements: list[str] = []
    for label in labels:
        block = by_label.get(label.lower())
        categories.append((block.category if block else None) or "")
        category = categories[-1]
        placement = ""
        if layout.placement_for is not None:
            placement = layout.placement_for(label, category, config) or ""
        placements.append(placement)
    return categories, placements


def _configured_sequence(task_cfg: dict[str, Any], key: str) -> list[str]:
    """Explicit label order from ``task.<name>.<key>``, when the operator set one."""
    import re

    raw = task_cfg.get(key) or []
    labels: list[str] = []
    for item in raw:
        normalized = " ".join(re.findall(r"[a-z0-9]+", str(item).lower())).strip()
        if normalized:
            labels.append(normalized)
    return labels


def _dedupe_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    def _area(bbox: list[float]) -> float:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))

    def _iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        denom = _area(a) + _area(b) - inter
        return inter / denom if denom > 0 else 0.0

    def _iom(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        min_area = min(_area(a), _area(b))
        return inter / min_area if min_area > 0 else 0.0

    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        if any(
            _iou(list(bbox), list(other["bbox_xyxy"])) >= iou_threshold
            or _iom(list(bbox), list(other["bbox_xyxy"])) >= 0.85
            for other in kept
        ):
            continue
        kept.append(det)
    for idx, det in enumerate(kept):
        det["id"] = idx
    return kept
