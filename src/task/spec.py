"""Load ``configs/tasks/<name>.yaml`` into a ``TaskSpec``.

The YAML is the task definition. This module parses it once, validates the
schema, and returns a ``TaskSpec``. Reset/grounding live in ``task.scene``;
optional Python control flow lives in ``src/task/task_hooks/<name>.py``. Canonical
top-level keys are listed in ``TASK_SPEC_KEYS``. Files may ``extends: other_task``
to overlay a variant (lists replace, mappings merge). Keys starting with ``x-``
are YAML-only anchors and are ignored. After a successful load, builders assume
types and references are already correct.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from common.config_loader import PROJECT_ROOT
from task.base import (
    TaskPrompts,
    grounding_generation_prompt,
    without_leading_the,
)
from task.behaviors import PreciseGroundSpec, ReadFromSpec
from task.types import (
    DEFAULT_CONTACT_TARGET_PHRASE,
    DEFAULT_FREE_TARGET_PHRASE,
    SceneLayout,
    SceneObject,
    SubtaskPlan,
    SubtaskStep,
)

SPEC_DIR = PROJECT_ROOT / "configs" / "tasks"

# Canonical top-level keys. Unknown keys fail load so a new task cannot drift
# onto an undocumented field that the runtime never reads.
TASK_SPEC_KEYS = frozenset(
    {
        "name",
        "extends",
        "hooks",
        "instruction",
        "task_context",
        "plan",
        "scene",
        "read_from",
        "precise_ground",
        "exclude_inside",
        "context",
        "commitment",
        "tracking",
        "recognition",
        "classify",
        "settings",
        "adapters",
        "recognize_movables",
        "exclude_movables_inside_placements",
        "target_word",
        "metadata",
    }
)
_SCENE_OBJECT_KEYS = frozenset(
    {
        "key",
        "role",
        "boxes",
        "optional",
        "ground_prompt",
        "dino_prompt",
        "recognize_crop",
        "max_box_area_ratio",
        "box_threshold",
        "text_threshold",
        "keep_top_k",
        "classify",
        "label_template",
        "label",
        "category",
        "distinguish",
        "as",
        "select_by",
        "locate",
        "fallback_bbox_from",
        "missing_message",
        "render",
    }
)
_CONTEXT_KEYS = frozenset({"source", "order", "order_config_key", "place_by", "steps"})
_CONTEXT_SOURCES = frozenset({"detected", "word_chars", "fixed"})
_COMMITMENT_KEYS = frozenset({"free", "contact", "strip_spatial_on"})
_RECOGNITION_KEYS = frozenset(
    {
        "landmark",
        "crop",
        "landmark_parser",
        "crop_parser",
        "crop_classify",
        "crop_category",
        "crop_min_side",
        "category_fallback",
    }
)
_PLAN_KEYS = frozenset({"fields"})
_TRACKING_KEYS = frozenset({"known_bbox_seeds", "segment_render_mode"})
_READ_FROM_KEYS = frozenset({"relative_to", "region", "parse"})
_PRECISE_GROUND_KEYS = frozenset({"of", "count_from", "axis", "max_slots"})
_CLASSIFY_ENTRY_KEYS = frozenset(
    {"labels", "aliases", "prompt", "unknown_label", "context_pattern"}
)
_ROBUST_DETECTION_KEYS = frozenset(
    {
        "method",
        "classify",
        "prompt",
        "expected_count",
        "require_all_labels",
        "min_crop_side",
        "instance_min_new_tokens",
        "classification_max_new_tokens",
        "classification_pad",
        "search_pad",
        "order_axis",
        "reverse_order",
        "box_filter",
        "enabled",
        "box_threshold",
        "text_threshold",
        "max_box_area_ratio",
        "keep_top_k",
    }
)
_BOX_FILTER_KEYS = frozenset(
    {"method", "max_area_ratio", "max_width_ratio", "max_height_ratio"}
)
_CATEGORY_FALLBACK_KEYS = frozenset({"classify", "grounding_prompt"})
_SUBTASK_STEP_KEYS = frozenset({"pick", "place", "label", "category", "placement"})
_ADAPTERS_KEYS = frozenset({"robust_detection"})
_SCENE_ROLES = frozenset({"movable", "placement", "landmark"})
_BOX_COUNTS = frozenset({"all", "one"})
_ASSIGN_ORDERS = frozenset(
    {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}
)
_PRECISE_AXES = frozenset({"x", "left_to_right"})
_RENAMED_SCENE_KEYS = {
    "recognize": "recognize_crop (true = run recognition.crop on each detected box)",
    "multi": "boxes: all | one (all = keep every matching box)",
    "required": "optional (true = reset continues if this object is missing)",
}
_RENAMED_ROLES = {"pick": "movable", "place": "placement"}
_OPTIONAL_SECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("commitment", _COMMITMENT_KEYS),
    ("recognition", _RECOGNITION_KEYS),
    ("plan", _PLAN_KEYS),
    ("tracking", _TRACKING_KEYS),
    ("read_from", _READ_FROM_KEYS),
    ("precise_ground", _PRECISE_GROUND_KEYS),
)


def _optional(raw: Any, convert: Callable[[Any], Any]) -> Any:
    return None if raw is None else convert(raw)


_BLANK_TEMPLATE_VALUES = frozenset({"", "unknown", "none", "null", "nil"})


def _fill_template(text: Any, context: Mapping[str, Any]) -> Any:
    """Fill ``{field}`` slots; missing or unknown values become empty tokens."""
    if not isinstance(text, str) or "{" not in text:
        return text
    import re

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            return ""
        value = str(context.get(key) or "").strip()
        if value.lower() in _BLANK_TEMPLATE_VALUES:
            return ""
        return value

    filled = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, text)
    return " ".join(filled.split())


def _substitute_known_fields(text: str, context: Mapping[str, Any]) -> str:
    """Fill named task fields without interpreting JSON braces in model prompts."""
    resolved = text
    for key, value in context.items():
        resolved = resolved.replace("{" + str(key) + "}", str(value))
    return resolved


def _with_bare_phrases(context: Mapping[str, Any]) -> dict[str, Any]:
    """Add ``<key>_bare`` for every ``<key>_phrase`` so templates can drop "the"."""
    enriched = dict(context)
    for key, value in list(context.items()):
        if key.endswith("_phrase") and isinstance(value, str):
            enriched.setdefault(f"{key}_bare", without_leading_the(value))
    return enriched


@dataclass(frozen=True)
class ClassifySpec:
    """A finite label set with its prompt and output alias table."""

    key: str
    prompt: str | None = None
    labels: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    unknown_label: str = "unknown"
    context_pattern: str | None = None
    """Regex with a ``{label}`` slot tried before bare words in ``find_in_text``."""

    def _canonical(self, value: Any) -> str | None:
        """Map a normalized phrase onto one label, treating ``left_bowl`` as ``left bowl``."""
        from task.base import normalize_words

        normalized = normalize_words(value)
        if not normalized:
            return None
        if normalized in self.aliases:
            return self.aliases[normalized]
        for label in self.labels:
            if normalize_words(label) == normalized:
                return label
        return None

    def _needles(self) -> tuple[tuple[int, str, str], ...]:
        """Longer normalized phrases first: ``(length, needle, canonical)``."""
        from task.base import normalize_words

        items: dict[str, str] = dict(self.aliases)
        for label in self.labels:
            needle = normalize_words(label)
            if needle:
                items.setdefault(needle, label)
        ranked = (
            (len(needle), needle, canonical)
            for needle, canonical in items.items()
            if needle
        )
        return tuple(sorted(ranked, reverse=True))

    def normalize(self, value: Any) -> str:
        """Map raw model text to one allowed label, or ``unknown_label``."""
        from task.base import normalize_words

        canonical = self._canonical(value)
        if canonical:
            return canonical
        for token in normalize_words(value).split():
            canonical = self._canonical(token)
            if canonical:
                return canonical
        return self.unknown_label

    def require(self, value: Any) -> str:
        """Map a whole answer to one allowed label; raise when it matches none."""
        canonical = self._canonical(value)
        if canonical:
            return canonical
        raise ValueError(f"failed to extract {self.key} from Qwen output: {value!r}")

    def find_in_text(self, text: Any, *, context_pattern: str | None = None) -> str:
        """Pull one label out of free text; longer aliases win ties."""
        import re

        from task.base import normalize_words

        normalized = normalize_words(text)
        needles = self._needles()
        pattern = self.context_pattern if context_pattern is None else context_pattern
        if pattern:
            for _length, token, label in needles:
                if re.search(pattern.format(label=re.escape(token)), normalized):
                    return label
        for _length, token, label in needles:
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return label
        return self.unknown_label


@dataclass(frozen=True)
class CategoryFallbackSpec:
    """Hard-label recognition/grounding fallback for one finite classifier."""

    classify: ClassifySpec
    grounding_prompt: str = "{category} object"


@dataclass(frozen=True)
class KnownBBoxSeedRule:
    """When a stage may reuse the active task object's known reset bbox."""

    stages: tuple[str, ...] = ()
    action_groups: tuple[str, ...] = ()
    progress_indices: tuple[int, ...] = ()

    def matches(self, task_state: Any, stage_name: str) -> bool:
        from task.base import action_pattern_group

        if self.stages and stage_name not in self.stages:
            return False
        if (
            self.action_groups
            and action_pattern_group(stage_name) not in self.action_groups
        ):
            return False
        return (
            not self.progress_indices or task_state.progress_idx in self.progress_indices
        )


@dataclass(frozen=True)
class SegmentRenderRule:
    """When a camera/stage pair should paint a known box instead of tracking it."""

    stages: tuple[str, ...] = ()
    cameras: tuple[str, ...] = ()
    mode: str = "sam2"

    def matches(self, camera: str, stage_name: str) -> bool:
        if self.stages and stage_name not in self.stages:
            return False
        if self.cameras and camera not in self.cameras:
            return False
        return True


@dataclass(frozen=True)
class RobustDetectionSpec:
    """One detection strategy from ``perception.semantic_grounder.robust``."""

    key: str
    method: str
    prompt: str = ""
    classify: ClassifySpec | None = None
    expected_count: int | None = None
    require_all_labels: bool = False
    min_crop_side: int = 256
    instance_min_new_tokens: int = 192
    classification_max_new_tokens: int = 16
    classification_pad: int = 0
    search_pad: int = 0
    order_axis: str = "y"
    reverse_order: bool = False
    box_filter: Callable[..., bool] | None = None
    enabled: bool = True
    # Optional DINO knobs for strategies that still call a phrase detector.
    # Declared in ``configs/tasks/*.yaml``, not runtime configs.
    box_threshold: float | None = None
    text_threshold: float | None = None
    max_box_area_ratio: float | None = None
    keep_top_k: int | None = None

    @property
    def classification_prompt(self) -> str:
        if self.classify is None or not self.classify.prompt:
            raise ValueError(f"detection {self.key!r} has no classification prompt")
        return self.classify.prompt

    @property
    def expected_labels(self) -> frozenset[str] | None:
        if not self.require_all_labels or self.classify is None:
            return None
        return frozenset(self.classify.labels)

    def locate(
        self,
        image: Any,
        *,
        config: Mapping[str, Any] | None = None,
        parent_box: Sequence[float] | None = None,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Run the strategy selected by ``method`` and return boxes keyed by label."""
        runner = ROBUST_DETECTION_RUNNERS.get(self.method, _unsupported_detection)
        return runner(self, image, config, parent_box)


DetectionBoxes = dict[str, tuple[float, float, float, float]]
DetectionRunner = Callable[
    [RobustDetectionSpec, Any, Mapping[str, Any] | None, Sequence[float] | None],
    DetectionBoxes,
]


def _normalizer(spec: RobustDetectionSpec) -> Callable[[Any], str] | None:
    return spec.classify.normalize if spec.classify is not None else None


def _run_separate_and_classify(
    spec: RobustDetectionSpec,
    image: Any,
    config: Mapping[str, Any] | None,
    parent_box: Sequence[float] | None,
) -> DetectionBoxes:
    from perception.semantic_grounder.robust import separate_and_classify_instances

    return separate_and_classify_instances(
        image,
        instance_prompt=spec.prompt,
        classification_prompt=spec.classification_prompt,
        normalize_label=_normalizer(spec),
        expected_labels=spec.expected_labels,
        expected_count=spec.expected_count,
        box_filter=spec.box_filter,
        search_crop=parent_box,
        search_pad=spec.search_pad,
        min_crop_side=spec.min_crop_side,
        classification_pad=spec.classification_pad,
        instance_min_new_tokens=spec.instance_min_new_tokens,
        classification_max_new_tokens=spec.classification_max_new_tokens,
        config=dict(config or {}),
    )


def _run_separate_and_joint_classify(
    spec: RobustDetectionSpec,
    image: Any,
    config: Mapping[str, Any] | None,
    parent_box: Sequence[float] | None,
) -> DetectionBoxes:
    from perception.semantic_grounder.robust import separate_and_jointly_classify_instances

    expected_labels = spec.expected_labels
    if expected_labels is None:
        raise ValueError(
            f"detection {spec.key!r} needs require_all_labels for joint classification"
        )
    return separate_and_jointly_classify_instances(
        image,
        instance_prompt=spec.prompt,
        classification_prompt=spec.classification_prompt,
        normalize_label=_normalizer(spec),
        expected_labels=expected_labels,
        expected_count=spec.expected_count,
        box_filter=spec.box_filter,
        search_crop=parent_box,
        search_pad=spec.search_pad,
        panel_size=spec.min_crop_side,
        classification_pad=spec.classification_pad,
        instance_min_new_tokens=spec.instance_min_new_tokens,
        classification_max_new_tokens=spec.classification_max_new_tokens,
        config=dict(config or {}),
    )


def _run_ordered_instances(
    spec: RobustDetectionSpec,
    image: Any,
    config: Mapping[str, Any] | None,
    parent_box: Sequence[float] | None,
) -> DetectionBoxes:
    from perception.semantic_grounder.robust import locate_instances_by_order

    if spec.classify is None:
        raise ValueError(f"detection {spec.key!r} needs classify labels for ordering")
    return locate_instances_by_order(
        image,
        instance_prompt=spec.prompt,
        labels_in_order=spec.classify.labels,
        order_axis=spec.order_axis,
        reverse=spec.reverse_order,
        expected_count=spec.expected_count,
        box_filter=spec.box_filter,
        search_crop=parent_box,
        search_pad=spec.search_pad,
        instance_min_new_tokens=spec.instance_min_new_tokens,
        config=dict(config or {}),
    )


def _run_classify_regions(
    spec: RobustDetectionSpec,
    image: Any,
    config: Mapping[str, Any] | None,
    parent_box: Sequence[float] | None,
) -> DetectionBoxes:
    from perception.semantic_grounder.robust import classify_regions_in_parent

    if parent_box is None or len(parent_box) != 4:
        raise ValueError(f"detection {spec.key!r} needs a parent_box")
    return classify_regions_in_parent(
        image,
        prompt=spec.prompt,
        parent_box=list(parent_box),
        normalize_label=_normalizer(spec),
        expected_labels=spec.expected_labels,
        search_pad=spec.search_pad,
        min_new_tokens=spec.instance_min_new_tokens,
        config=dict(config or {}),
    )


def _unsupported_detection(
    spec: RobustDetectionSpec,
    image: Any,
    config: Mapping[str, Any] | None,
    parent_box: Sequence[float] | None,
) -> DetectionBoxes:
    del image, config, parent_box
    raise ValueError(f"detection {spec.key!r} has no runnable method (got {spec.method!r})")


ROBUST_DETECTION_RUNNERS: dict[str, DetectionRunner] = {
    "separate_and_classify": _run_separate_and_classify,
    "separate_and_joint_classify": _run_separate_and_joint_classify,
    "ordered_instances": _run_ordered_instances,
    "classify_regions": _run_classify_regions,
}


@dataclass(frozen=True)
class SceneObjectSpec:
    """Declarative form of one ``SceneObject``."""

    key: str
    role: str
    phrase: str | None = None
    dino_prompt: str | None = None
    label: str | None = None
    category: str | None = None
    boxes: str = "one"
    """``all`` keeps every matching box; ``one`` keeps a single box."""
    optional: bool = False
    """When True, reset continues if this object is not found."""
    recognize_crop: bool = True
    """When True, reset runs ``recognition.crop`` on each detected box of this object."""
    max_box_area_ratio: float | None = None
    box_threshold: float | None = None
    text_threshold: float | None = None
    keep_top_k: int | None = None
    classify: str | None = None
    """Optional ``classify.<name>`` key used after detection to label each box."""
    label_template: str | None = None
    """Format ``{label}`` from classify into a placement name (e.g. ``{label}_basket``)."""
    assign_labels: tuple[str, ...] = ()
    """Fixed labels assigned by spatial order when boxes look alike (e.g. bowls)."""
    assign_order: str = "left_to_right"
    """``left_to_right`` / ``right_to_left`` / ``top_to_bottom`` / ``bottom_to_top``."""

    locate_strategy: str | None = None
    """Optional ``perception.semantic_grounder.robust`` key used instead of phrase grounding."""
    select_label: str | None = None
    """Select one labeled result from ``locate_strategy`` after context formatting."""
    fallback_bbox_from: str | None = None
    """Scene-object key whose first box is used when this object is not found."""
    missing_message: str | None = None
    """Soft-failure message recorded when this object is not found."""

    def build(
        self,
        context: Mapping[str, Any],
        robust_detection: Mapping[str, RobustDetectionSpec],
    ) -> SceneObject:
        phrase = _fill_template(self.phrase, context) or ""
        # ``{phrase}`` lets a DINO prompt default to this object's own phrase.
        local = {**context, "phrase": phrase}
        locator = None
        if self.locate_strategy:
            strategy = _fill_template(self.locate_strategy, local)
            if strategy:
                try:
                    locator = robust_detection[strategy]
                except KeyError as exc:
                    raise KeyError(
                        f"scene object {self.key!r} references unknown robust detection "
                        f"{strategy!r}"
                    ) from exc
                locator = replace(
                    locator,
                    prompt=_substitute_known_fields(locator.prompt, local),
                )
        return SceneObject(
            key=self.key,
            role=self.role,  # type: ignore[arg-type]
            phrase=phrase,
            dino_prompt=_fill_template(self.dino_prompt, local) or None,
            boxes=self.boxes,
            optional=self.optional,
            label=_fill_template(self.label, local),
            category=_fill_template(self.category, local),
            max_box_area_ratio=self.max_box_area_ratio,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            keep_top_k=self.keep_top_k,
            recognize_crop=self.recognize_crop,
            classify=self.classify,
            label_template=self.label_template,
            assign_labels=self.assign_labels,
            assign_order=self.assign_order,
            locator=locator,
            select_label=_fill_template(self.select_label, local) or None,
            fallback_bbox_from=self.fallback_bbox_from,
            missing_message=_fill_template(self.missing_message, local) or None,
        )


@dataclass(frozen=True)
class TaskSpec:
    """One task's prompts, scene objects, detection strategies, and sequence rules."""

    name: str
    plugin: str | None = None
    """Optional ``task.task_hooks.<name>`` module bound by the registry."""
    landmark_recognition: str = ""
    crop_recognition: str | None = None
    landmark_parser: str | None = None
    crop_parser: str | None = None
    crop_classify: str | None = None
    crop_category: str = ""
    crop_min_side: int = 0
    category_fallback: CategoryFallbackSpec | None = None
    grounding_task_context: str = ""
    context_fields: tuple[str, ...] = ()
    known_bbox_seed_rules: tuple[KnownBBoxSeedRule, ...] = ()
    segment_render_rules: tuple[SegmentRenderRule, ...] = ()
    classify: dict[str, ClassifySpec] = field(default_factory=dict)
    scene_objects: tuple[SceneObjectSpec, ...] = ()
    robust_detection: dict[str, RobustDetectionSpec] = field(default_factory=dict)
    recognize_movables: bool = True
    exclude_movables_inside_placements: bool = True
    target_word: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    subtasks: SubtaskPlan = field(default_factory=SubtaskPlan)
    free_target_phrase: str = DEFAULT_FREE_TARGET_PHRASE
    contact_target_phrase: str = DEFAULT_CONTACT_TARGET_PHRASE
    strip_spatial_on_stages: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    read_from: ReadFromSpec | None = None
    precise_ground: PreciseGroundSpec | None = None
    exclude_inside: tuple[str, ...] = ()

    def prompts(self) -> TaskPrompts:
        return TaskPrompts(
            name=self.name,
            landmark_recognition=self.landmark_recognition,
            crop_recognition=self.crop_recognition,
            grounding_generation=grounding_generation_prompt(self.grounding_task_context),
        )

    def scene_object(self, key: str) -> SceneObjectSpec:
        for scene_object in self.scene_objects:
            if scene_object.key == key:
                return scene_object
        known = ", ".join(spec.key for spec in self.scene_objects) or "(none)"
        raise KeyError(f"task {self.name!r} has no scene object {key!r}; known: {known}")

    def resolve_scene_object(
        self,
        key: str,
        config: Mapping[str, Any] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> SceneObject:
        """Resolve one scene object outside ``scene_layout`` (per-frame re-detection)."""
        del config
        return self.scene_object(key).build(
            _with_bare_phrases(context or {}),
            self.robust_detection,
        )

    def detect(
        self,
        key: str,
        *,
        image: Any,
        dino_client: Any,
        config: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        **detect_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Re-detect one scene object on a later frame, with the reset settings.

        Same thresholds and prompts ``understand_scene`` uses, so a task that
        refreshes objects every frame does not restate them.
        """
        from task.scene import locate_scene_object

        return locate_scene_object(
            self.resolve_scene_object(key, config, context=context),
            image=image,
            dino_client=dino_client,
            config=dict(config or {}),
            **detect_kwargs,
        )

    def detect_one(self, key: str, **kwargs: Any) -> dict[str, Any] | None:
        """``detect`` for a single-box object; ``None`` when nothing is found."""
        detections = self.detect(key, **kwargs)
        return detections[0] if detections else None

    def locate(
        self,
        key: str,
        image: Any,
        *,
        config: Mapping[str, Any] | None = None,
        parent_box: Sequence[float] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Run a declared ``robust_detection`` strategy; boxes keyed by label."""
        method = self.detection_method(key)
        if context:
            method = replace(
                method,
                prompt=_substitute_known_fields(method.prompt, context),
            )
        return method.locate(
            image,
            config=config,
            parent_box=parent_box,
        )

    def prefers_known_bbox_seed(self, task_state: Any, stage_name: str) -> bool:
        return any(
            rule.matches(task_state, stage_name) for rule in self.known_bbox_seed_rules
        )

    def resolve_segment_render_mode(
        self,
        task_state: Any,
        *,
        camera: str,
        stage_name: str,
        prompt: str = "",
    ) -> str:
        del task_state, prompt
        for rule in self.segment_render_rules:
            if rule.matches(camera, stage_name):
                return rule.mode
        return "sam2"

    def landmark_output_parser(self) -> Callable[[Any], str] | None:
        from task.parsers import get_parser

        return None if not self.landmark_parser else get_parser(self.landmark_parser)

    def crop_output_parser(self) -> Callable[[Any], str] | None:
        from task.parsers import LABEL_CATEGORY_JSON, get_parser

        if not self.crop_parser or self.crop_parser == LABEL_CATEGORY_JSON:
            return None
        return get_parser(self.crop_parser)

    def crop_result_parser(self) -> Callable[[Any], dict[str, str]] | None:
        """Build ``parse_crop_result`` from the declared crop parser.

        ``label_category_json`` reads ``{label, category}`` and normalizes the
        category through the named classify spec; any other parser produces the
        label and pairs it with the fixed ``crop_category``.
        """
        from task.parsers import LABEL_CATEGORY_JSON

        if not self.crop_parser:
            return None

        if self.crop_parser == LABEL_CATEGORY_JSON:
            from task.base import parse_label_category_json

            classify = self.classify_spec(self.crop_classify) if self.crop_classify else None

            def parse_label_category(text: Any) -> dict[str, str]:
                cleaned = str(text).strip()
                if not cleaned:
                    raise ValueError("empty crop recognition output")
                result = parse_label_category_json(cleaned)
                if result["category"] and classify is not None:
                    result["category"] = classify.require(result["category"])
                return result

            return parse_label_category

        from task.parsers import get_parser

        parse_label = get_parser(self.crop_parser)
        category = self.crop_category

        def parse_labeled(text: Any) -> dict[str, str]:
            return {"label": parse_label(text), "category": category}

        return parse_labeled

    def placement_resolver(
        self, mapping: Mapping[str, str] | None = None
    ) -> Callable[..., str] | None:
        """``placement_for`` built from the declared category → placement map."""
        resolved = dict(mapping or self.subtasks.placement_by_category)
        if not resolved:
            return None

        def placement_for(label: str, category: str, config: Any = None) -> str:
            del label, config
            return resolved.get(category or "", "")

        return placement_for

    def classify_spec(self, key: str) -> ClassifySpec:
        try:
            return self.classify[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.classify)) or "(none)"
            raise KeyError(f"task {self.name!r} has no classify {key!r}; known: {known}") from exc

    def detection_method(self, key: str) -> RobustDetectionSpec:
        try:
            return self.robust_detection[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.robust_detection)) or "(none)"
            raise KeyError(
                f"task {self.name!r} has no robust detection {key!r}; known: {known}"
            ) from exc

    def subtask_plan(self, context: Mapping[str, Any]) -> SubtaskPlan:
        """The declared plan with every fixed step filled in from ``context``."""
        plan = self.subtasks
        mapping = {
            str(_fill_template(category, context) or category): self._resolved_destination(
                category, destination, context
            )
            for category, destination in plan.placement_by_category.items()
        }
        mapping = {key: value for key, value in mapping.items() if key}
        mapping = self._place_by_from_prompt(mapping, str(context.get("instruction") or ""))
        if not plan.steps:
            return replace(plan, placement_by_category=mapping)
        return replace(
            plan,
            placement_by_category=mapping,
            steps=tuple(
                SubtaskStep(
                    label=_fill_template(step.label, context),
                    category=_fill_template(step.category, context),
                    placement=_fill_template(step.placement, context),
                    pick=step.pick,
                    place=step.place,
                )
                for step in plan.steps
            ),
        )

    def _resolved_destination(
        self,
        category: str,
        destination: str,
        context: Mapping[str, Any],
    ) -> str:
        for key in (f"{category}_placement", f"selected_{category}_placement"):
            override = str(context.get(key) or "").strip()
            if override and override.lower() not in _BLANK_TEMPLATE_VALUES:
                return override
        filled = _fill_template(destination, context)
        return str(filled or destination)

    def _place_by_from_prompt(
        self,
        mapping: dict[str, str],
        instruction: str,
    ) -> dict[str, str]:
        """Read ``<category>s into the <placement>`` from the episode instruction.

        Same idea as peg-in-hole: the YAML ``instruction`` is the example
        sentence; a caller passes that form with different destinations.
        """
        import re

        placement = self.classify.get("placement")
        if not instruction.strip() or placement is None:
            return mapping
        valid = set(mapping.values())
        for scene_object in self.scene_objects:
            if scene_object.role != "placement":
                continue
            if scene_object.assign_labels:
                valid.update(scene_object.assign_labels)
            else:
                valid.add(scene_object.key)
        updated = dict(mapping)
        for category in mapping:
            pattern = rf"\b{re.escape(category)}s?\s+into\s+(?:the\s+)?{{label}}\b"
            found = placement.find_in_text(instruction, context_pattern=pattern)
            matched = _match_placement_id(found, valid)
            if found != placement.unknown_label and matched:
                updated[category] = matched
        return updated

    def scene_layout(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
        task_context: Any | None = None,
        placement_for: Callable[..., str | None] | None = None,
    ) -> SceneLayout:
        """Build the reset-time ``SceneLayout``.

        ``context`` supplies format values for object phrases and plan metadata
        (peg-in-hole derives them from the episode plan). Resolved object fields
        are added to it as ``<key>_phrase`` / ``<key>_dino_prompt`` /
        ``<key>_label`` so plan metadata can echo a prompt without repeating it.
        """
        del config
        episode_context: dict[str, Any] = {}
        if task_context is not None:
            episode_context.update(dict(task_context.payload))
            instruction = str(task_context.instruction or "").strip()
            if instruction:
                episode_context["instruction"] = instruction
        object_context = _with_bare_phrases(
            _alias_plan_fields({**episode_context, **dict(context or {})})
        )
        scene_objects = tuple(
            spec.build(object_context, self.robust_detection) for spec in self.scene_objects
        )
        objects_by_key = {item.key: item for item in scene_objects}

        derived: dict[str, Any] = {}
        for scene_object in scene_objects:
            derived[f"{scene_object.key}_phrase"] = scene_object.phrase
            derived[f"{scene_object.key}_dino_prompt"] = scene_object.dino_prompt
            derived[f"{scene_object.key}_label"] = scene_object.label
        full_context = {**_with_bare_phrases(derived), **object_context}
        subtasks = self._resolved_subtask_plan(full_context, objects_by_key)

        return SceneLayout(
            name=self.name,
            scene_objects=scene_objects,
            exclude_movables_inside_placements=self.exclude_movables_inside_placements,
            exclude_inside=self.exclude_inside,
            recognize_movables=self.recognize_movables,
            target_word=_fill_template(self.target_word, full_context),
            metadata={
                key: _fill_template(value, full_context) for key, value in self.metadata.items()
            },
            subtasks=subtasks,
            placement_for=placement_for
            or self.placement_resolver(subtasks.placement_by_category),
            free_target_phrase=self.free_target_phrase,
            contact_target_phrase=self.contact_target_phrase,
            classify=dict(self.classify),
        )

    def _resolved_subtask_plan(
        self,
        context: Mapping[str, Any],
        objects_by_key: Mapping[str, Any],
    ) -> SubtaskPlan:
        plan = self.subtask_plan(context)
        if not plan.steps:
            return plan
        resolved = []
        for step in plan.steps:
            label, category, placement = step.label, step.category, step.placement
            if step.pick:
                picked = objects_by_key[step.pick]
                label = label or without_leading_the(picked.phrase or "") or (picked.label or "")
                category = category or (picked.category or "")
            if step.place:
                placed = objects_by_key[step.place]
                placement = (
                    placement
                    or without_leading_the(placed.phrase or "")
                    or (placed.label or placed.key)
                )
            resolved.append(
                replace(step, label=label, category=category, placement=placement)
            )
        return replace(plan, steps=tuple(resolved))


def _alias_plan_fields(context: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``selected_<field>`` payload keys to the short ``<field>`` name."""
    aliased = dict(context)
    for key, value in list(context.items()):
        if key.startswith("selected_") and key[len("selected_") :] not in aliased:
            aliased[key[len("selected_") :]] = value
    return aliased


def _section(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return {} if value is None else dict(value)


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def _tuple_of_str(raw: Any, default: Sequence[str] = ()) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    return tuple(str(item) for item in raw)


def _expand_aliases(raw: Any) -> dict[str, str]:
    """Expand YAML aliases into variant -> canonical lookup.

    Preferred form is one-to-many::

        circular: [circular, circle, round]

    """
    if not raw:
        return {}
    from task.base import normalize_words

    expanded: dict[str, str] = {}
    for canonical_name, aliases in raw.items():
        canonical = str(canonical_name)
        variants = [str(item) for item in aliases]
        for variant in variants:
            variant_n = normalize_words(variant)
            if variant_n:
                expanded[variant_n] = canonical
    return expanded


def _with_label_aliases(aliases: Mapping[str, str], labels: Sequence[str]) -> dict[str, str]:
    """Index each label by its ``normalize_words`` form (``left_bowl`` → ``left bowl``)."""
    from task.base import normalize_words

    merged = dict(aliases)
    for label in labels:
        needle = normalize_words(label)
        if needle:
            merged.setdefault(needle, str(label))
    return merged


def _scene_placement_ids(scene_objects: Sequence[SceneObjectSpec]) -> tuple[str, ...]:
    """Destination ids used by ``place_by``: ``as`` labels, else the scene key."""
    ids: list[str] = []
    seen: set[str] = set()
    for scene_object in scene_objects:
        if scene_object.role != "placement":
            continue
        names = scene_object.assign_labels or (scene_object.key,)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ids.append(name)
    return tuple(ids)


def _match_placement_id(found: str, valid: set[str]) -> str | None:
    """Resolve a classify hit onto a scene destination, ignoring underscore vs space."""
    from task.base import normalize_words

    if found in valid:
        return found
    found_n = normalize_words(found)
    if not found_n:
        return None
    matches = [item for item in valid if normalize_words(item) == found_n]
    return matches[0] if len(matches) == 1 else None


def _bind_placement_classify(
    classify: Mapping[str, ClassifySpec],
    scene_objects: Sequence[SceneObjectSpec],
) -> dict[str, ClassifySpec]:
    """Keep ``classify.placement`` labels in lockstep with the scene destinations.

    ``extends`` merges alias maps, so a bowl overlay would otherwise inherit
    basket labels. Scene keys / ``as`` are the source of truth; leftover aliases
    that name a dropped destination are discarded. ``left_bowl`` still matches
    ``left bowl`` in an instruction via ``normalize_words``.
    """
    ids = _scene_placement_ids(scene_objects)
    existing = classify.get("placement")
    if existing is None or not ids:
        return dict(classify)
    allowed = set(ids)
    aliases = {
        needle: label for needle, label in existing.aliases.items() if label in allowed
    }
    return {
        **classify,
        "placement": replace(
            existing,
            labels=ids,
            aliases=_with_label_aliases(aliases, ids),
        ),
    }


def _build_classify(payload: Mapping[str, Any]) -> dict[str, ClassifySpec]:
    specs: dict[str, ClassifySpec] = {}
    for key, raw in payload.items():
        entry = dict(raw)
        labels = _tuple_of_str(entry.get("labels"))
        specs[str(key)] = ClassifySpec(
            key=str(key),
            prompt=entry.get("prompt"),
            labels=labels,
            aliases=_with_label_aliases(_expand_aliases(entry.get("aliases")), labels),
            unknown_label=str(entry.get("unknown_label", "unknown")),
            context_pattern=entry.get("context_pattern"),
        )
    return specs


def _build_box_filter(raw: Any) -> Callable[..., bool] | None:
    if not raw:
        return None
    from perception.semantic_grounder.robust import compact_box_filter

    options = {key: value for key, value in raw.items() if key != "method"}
    return compact_box_filter(**options)


def _ordered_instance_json_suffix(count: int) -> str:
    rows = ",\n".join('  {"bbox_2d":[x1,y1,x2,y2]}' for _ in range(count))
    return (
        f"Output exactly {count} boxes as JSON and no other text:\n"
        f"[\n{rows}\n]\n"
        "Coordinates are normalized from 0 to 1000 relative to the whole image."
    )


def _with_ordered_instance_format(
    method: str, prompt: str, expected_count: int | None
) -> str:
    """ordered_instances always asks for N bbox_2d boxes; YAML only names the objects."""
    if method != "ordered_instances" or "bbox_2d" in prompt:
        return prompt
    count = 3 if expected_count is None else int(expected_count)
    prompt = prompt.rstrip()
    suffix = _ordered_instance_json_suffix(count)
    return f"{prompt}\n{suffix}" if prompt else suffix


def _metadata_from_plan_fields(
    raw: Mapping[str, Any] | None,
    context_fields: Sequence[str],
) -> dict[str, Any]:
    """Copy ``plan.fields`` into metadata templates when the YAML omits them."""
    metadata = dict(raw or {})
    if metadata or not context_fields:
        return metadata
    derived: dict[str, Any] = {}
    for field_name in context_fields:
        short = (
            field_name[len("selected_") :]
            if field_name.startswith("selected_")
            else field_name
        )
        derived[field_name] = "{" + short + "}"
    derived.setdefault("instruction", "{instruction}")
    return derived


def _build_robust_detection(
    payload: Mapping[str, Any],
    classify: Mapping[str, ClassifySpec],
) -> dict[str, RobustDetectionSpec]:
    strategies: dict[str, RobustDetectionSpec] = {}
    for key, raw in payload.items():
        entry = dict(raw)
        classify_key = entry.get("classify")
        method = str(entry.get("method", ""))
        expected_count = (
            None if entry.get("expected_count") is None else int(entry["expected_count"])
        )
        strategies[str(key)] = RobustDetectionSpec(
            key=str(key),
            method=method,
            prompt=_with_ordered_instance_format(
                method, str(entry.get("prompt") or ""), expected_count
            ),
            classify=classify.get(str(classify_key)) if classify_key else None,
            expected_count=expected_count,
            require_all_labels=bool(entry.get("require_all_labels", False)),
            min_crop_side=int(entry.get("min_crop_side", 256)),
            instance_min_new_tokens=int(entry.get("instance_min_new_tokens", 192)),
            classification_max_new_tokens=int(entry.get("classification_max_new_tokens", 16)),
            classification_pad=int(entry.get("classification_pad", 0)),
            search_pad=int(entry.get("search_pad", 0)),
            order_axis=str(entry.get("order_axis", "y")).lower(),
            reverse_order=bool(entry.get("reverse_order", False)),
            box_filter=_build_box_filter(entry.get("box_filter")),
            enabled=bool(entry.get("enabled", True)),
            box_threshold=_optional(entry.get("box_threshold"), float),
            text_threshold=_optional(entry.get("text_threshold"), float),
            max_box_area_ratio=_optional(entry.get("max_box_area_ratio"), float),
            keep_top_k=_optional(entry.get("keep_top_k"), int),
        )
    return strategies


def _build_scene_objects(payload: Sequence[Mapping[str, Any]]) -> tuple[SceneObjectSpec, ...]:
    scene_objects: list[SceneObjectSpec] = []
    for entry in payload:
        select_by = entry.get("select_by")
        scene_objects.append(
            SceneObjectSpec(
                key=entry["key"],
                role=entry["role"],
                phrase=entry.get("ground_prompt"),
                dino_prompt=entry.get("dino_prompt"),
                label=entry.get("label"),
                category=entry.get("category"),
                boxes=entry.get("boxes", "one"),
                optional=entry.get("optional", False),
                recognize_crop=entry.get("recognize_crop", True),
                max_box_area_ratio=_optional(entry.get("max_box_area_ratio"), float),
                box_threshold=_optional(entry.get("box_threshold"), float),
                text_threshold=_optional(entry.get("text_threshold"), float),
                keep_top_k=_optional(entry.get("keep_top_k"), int),
                classify=entry.get("classify"),
                label_template=entry.get("label_template"),
                assign_labels=tuple(entry.get("as") or ()),
                assign_order=entry.get("distinguish", "left_to_right"),
                locate_strategy=entry.get("locate"),
                select_label=("{" + select_by + "}") if select_by else None,
                fallback_bbox_from=entry.get("fallback_bbox_from"),
                missing_message=entry.get("missing_message"),
            )
        )
    return tuple(scene_objects)


def _build_subtask_plan(payload: Mapping[str, Any]) -> SubtaskPlan:
    """Parse ``context``: where the subtask sequence comes from."""
    return SubtaskPlan(
        source=payload.get("source", "detected"),  # type: ignore[arg-type]
        order=payload.get("order", "detected"),
        order_config_key=payload.get("order_config_key", "object_sequence"),
        placement_by_category=dict(payload.get("place_by") or {}),
        steps=tuple(_build_subtask_step(step) for step in payload.get("steps") or ()),
    )


def _build_subtask_step(step: Mapping[str, Any]) -> SubtaskStep:
    return SubtaskStep(
        label=step.get("label", ""),
        category=step.get("category", ""),
        placement=step.get("placement", ""),
        pick=step.get("pick"),
        place=step.get("place"),
    )


def _build_known_bbox_seed_rules(payload: Sequence[Any]) -> tuple[KnownBBoxSeedRule, ...]:
    rules: list[KnownBBoxSeedRule] = []
    for raw in payload:
        entry = dict(raw)
        stages = _tuple_of_str(entry.get("stages"))
        action_groups = _tuple_of_str(entry.get("action_groups"))
        if not stages and not action_groups:
            raise ValueError("known_bbox_seeds rule needs stages or action_groups")
        unknown_groups = set(action_groups) - {"free", "contact"}
        if unknown_groups:
            raise ValueError(
                f"unknown known_bbox_seeds action_groups: {sorted(unknown_groups)}"
            )
        rules.append(
            KnownBBoxSeedRule(
                stages=stages,
                action_groups=action_groups,
                progress_indices=tuple(
                    int(value) for value in (entry.get("progress_indices") or ())
                ),
            )
        )
    return tuple(rules)


def _build_segment_render_rules(payload: Sequence[Any]) -> tuple[SegmentRenderRule, ...]:
    rules: list[SegmentRenderRule] = []
    for raw in payload:
        entry = dict(raw)
        mode = str(entry.get("mode") or "").strip()
        if not mode:
            raise ValueError("segment_render_mode rule needs a mode")
        rules.append(
            SegmentRenderRule(
                stages=_tuple_of_str(entry.get("stages")),
                cameras=_tuple_of_str(entry.get("cameras")),
                mode=mode,
            )
        )
    return tuple(rules)


def _build_read_from(raw: Mapping[str, Any] | None) -> ReadFromSpec | None:
    if raw is None:
        return None
    return ReadFromSpec(
        relative_to=raw["relative_to"],
        region=raw.get("region", "above"),
        parse=raw.get("parse"),
    )


def _build_precise_ground(raw: Mapping[str, Any] | None) -> PreciseGroundSpec | None:
    if raw is None:
        return None
    max_slots = raw.get("max_slots")
    return PreciseGroundSpec(
        of=raw["of"],
        count_from=raw.get("count_from", "word_length"),
        axis=raw.get("axis", "x"),
        max_slots=None if max_slots is None else int(max_slots),
    )


_AREA_PLACEMENT_RULE = SegmentRenderRule(
    stages=("transport_contact",),
    cameras=("global",),
    mode="area_placement",
)


def _render_rules_from_scene(scene_objects: Sequence[Mapping[str, Any]]) -> tuple[SegmentRenderRule, ...]:
    rules: list[SegmentRenderRule] = []
    for entry in scene_objects:
        raw = entry.get("render")
        if raw is None:
            continue
        if raw == "area":
            rules.append(_AREA_PLACEMENT_RULE)
            continue
        mode = raw.get("mode", "area")
        if mode == "area":
            mode = "area_placement"
        rules.append(
            SegmentRenderRule(
                stages=_tuple_of_str(raw.get("stages"), ("transport_contact",)),
                cameras=_tuple_of_str(raw.get("cameras"), ("global",)),
                mode=mode,
            )
        )
    return tuple(rules)


def _is_anchor_key(key: str) -> bool:
    return str(key).startswith("x-") or str(key).startswith("x_")


def _unknown_keys(payload: Mapping[str, Any], allowed: frozenset[str]) -> list[str]:
    return sorted(
        str(key)
        for key in payload
        if str(key) not in allowed and not _is_anchor_key(str(key))
    )


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    where: str,
) -> None:
    unknown = _unknown_keys(payload, allowed)
    if unknown:
        raise ValueError(f"unknown {where} key(s) {unknown}; allowed: {sorted(allowed)}")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge task YAML mappings; lists and scalars in ``overlay`` replace."""

    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _validate_parser_name(name: Any, where: str) -> None:
    from task.parsers import LABEL_CATEGORY_JSON, PARSERS

    if name is None:
        return
    known = frozenset(PARSERS) | {LABEL_CATEGORY_JSON}
    if name not in known:
        raise ValueError(f"{where} unknown parser {name!r}; known: {sorted(known)}")


def _validate_payload(payload: Mapping[str, Any]) -> None:
    _reject_unknown_keys(payload, TASK_SPEC_KEYS, where="task spec")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("task spec needs a non-empty name")

    for key, allowed in _OPTIONAL_SECTIONS:
        section = payload.get(key)
        if section is None:
            continue
        _require_mapping(section, key)
        _reject_unknown_keys(section, allowed, where=key)

    classify = payload.get("classify") or {}
    if classify:
        _require_mapping(classify, "classify")
    classify_keys = set(classify)
    for key, raw in classify.items():
        _require_mapping(raw, f"classify.{key}")
        _reject_unknown_keys(raw, _CLASSIFY_ENTRY_KEYS, where=f"classify.{key}")

    adapters = payload.get("adapters") or {}
    if adapters:
        _require_mapping(adapters, "adapters")
        _reject_unknown_keys(adapters, _ADAPTERS_KEYS, where="adapters")
    detection = dict(adapters.get("robust_detection") or {}) if adapters else {}
    detection_keys = set(detection)
    for key, raw in detection.items():
        _require_mapping(raw, f"adapters.robust_detection.{key}")
        _reject_unknown_keys(raw, _ROBUST_DETECTION_KEYS, where=f"adapters.robust_detection.{key}")
        method = raw.get("method")
        if method not in ROBUST_DETECTION_RUNNERS:
            known = ", ".join(sorted(ROBUST_DETECTION_RUNNERS)) or "(none)"
            raise ValueError(
                f"adapters.robust_detection.{key}.method must be one of {known}"
            )
        classify_ref = raw.get("classify")
        if classify_ref is not None and classify_ref not in classify_keys:
            raise ValueError(
                f"adapters.robust_detection.{key}.classify {classify_ref!r} is not in classify"
            )
        box_filter = raw.get("box_filter")
        if box_filter is not None:
            _require_mapping(box_filter, f"adapters.robust_detection.{key}.box_filter")
            _reject_unknown_keys(
                box_filter,
                _BOX_FILTER_KEYS,
                where=f"adapters.robust_detection.{key}.box_filter",
            )

    scene = payload.get("scene") or ()
    scene_keys: list[str] = []
    for index, entry in enumerate(scene):
        _require_mapping(entry, f"scene[{index}]")
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"scene[{index}] needs a non-empty key")
        for old, new in _RENAMED_SCENE_KEYS.items():
            if old in entry:
                raise ValueError(f"scene.{key}: {old} was renamed to {new}")
        role = entry.get("role")
        if role in _RENAMED_ROLES:
            raise ValueError(
                f"scene.{key}.role: {role} was renamed to {_RENAMED_ROLES[role]}"
            )
        _reject_unknown_keys(entry, _SCENE_OBJECT_KEYS, where=f"scene.{key}")
        if role not in _SCENE_ROLES:
            raise ValueError(
                f"scene.{key}.role must be one of {sorted(_SCENE_ROLES)}, got {role!r}"
            )
        if not entry.get("ground_prompt") and not entry.get("locate"):
            raise ValueError(f"scene.{key} needs ground_prompt or locate")
        if "recognize_crop" in entry and not isinstance(entry["recognize_crop"], bool):
            raise ValueError(
                f"scene.{key}.recognize_crop must be a boolean; "
                "name crop parsers under recognition.crop_parser"
            )
        boxes = entry.get("boxes")
        if boxes is not None and boxes not in _BOX_COUNTS:
            raise ValueError(
                f"scene.{key}.boxes must be one of {sorted(_BOX_COUNTS)}, got {boxes!r}"
            )
        if "optional" in entry and not isinstance(entry["optional"], bool):
            raise ValueError(f"scene.{key}.optional must be a boolean")
        distinguish = entry.get("distinguish")
        if distinguish is not None and distinguish not in _ASSIGN_ORDERS:
            raise ValueError(
                f"scene.{key}.distinguish must be one of {sorted(_ASSIGN_ORDERS)}"
            )
        labels = entry.get("as")
        if labels is not None and (isinstance(labels, str) or not isinstance(labels, Sequence)):
            raise ValueError(f"scene.{key}.as must be a list of labels")
        render = entry.get("render")
        if render is not None and render != "area" and not isinstance(render, Mapping):
            raise ValueError(f"scene.{key}.render must be 'area' or a mapping")
        classify_key = entry.get("classify")
        if classify_key is not None and classify_key not in classify_keys:
            raise ValueError(f"scene.{key}.classify {classify_key!r} is not in classify")
        locate = entry.get("locate")
        if isinstance(locate, str) and "{" not in locate and locate not in detection_keys:
            raise ValueError(
                f"scene.{key}.locate {locate!r} is not in adapters.robust_detection"
            )
        scene_keys.append(key)
    scene_key_set = set(scene_keys)

    for entry in scene:
        fallback = entry.get("fallback_bbox_from")
        if fallback is not None and fallback not in scene_key_set:
            raise ValueError(
                f"scene.{entry['key']}.fallback_bbox_from {fallback!r} is not a scene key"
            )

    context = payload.get("context") or {}
    if payload.get("context") is not None:
        _require_mapping(context, "context")
    _reject_unknown_keys(context, _CONTEXT_KEYS, where="context")
    source = context.get("source")
    if source is not None and source not in _CONTEXT_SOURCES:
        raise ValueError(
            f"context.source must be one of {sorted(_CONTEXT_SOURCES)}, got {source!r}"
        )
    for step in context.get("steps") or ():
        _require_mapping(step, "context.steps")
        _reject_unknown_keys(step, _SUBTASK_STEP_KEYS, where="context.steps")

    recognition = payload.get("recognition") or {}
    crop_min_side = recognition.get("crop_min_side", 0)
    if type(crop_min_side) is not int or crop_min_side < 0:
        raise ValueError("recognition.crop_min_side must be a non-negative integer")
    _validate_parser_name(recognition.get("landmark_parser"), "recognition.landmark_parser")
    _validate_parser_name(recognition.get("crop_parser"), "recognition.crop_parser")
    crop_classify = recognition.get("crop_classify")
    if crop_classify is not None and crop_classify not in classify_keys:
        raise ValueError(f"recognition.crop_classify {crop_classify!r} is not in classify")
    fallback = recognition.get("category_fallback")
    if fallback is not None:
        _require_mapping(fallback, "recognition.category_fallback")
        _reject_unknown_keys(fallback, _CATEGORY_FALLBACK_KEYS, where="recognition.category_fallback")
        fallback_key = fallback.get("classify")
        if fallback_key not in classify_keys:
            raise ValueError(
                "recognition.category_fallback.classify must name an entry in classify"
            )

    read_from = payload.get("read_from")
    if read_from is not None:
        relative_to = read_from.get("relative_to")
        if not relative_to:
            raise ValueError("read_from needs relative_to")
        if relative_to not in scene_key_set:
            raise ValueError(f"read_from.relative_to {relative_to!r} is not a scene key")
        _validate_parser_name(read_from.get("parse"), "read_from.parse")

    precise_ground = payload.get("precise_ground")
    if precise_ground is not None:
        of = precise_ground.get("of")
        if not of:
            raise ValueError("precise_ground needs of")
        if of not in scene_key_set:
            raise ValueError(f"precise_ground.of {of!r} is not a scene key")
        axis = precise_ground.get("axis", "x")
        if axis not in _PRECISE_AXES:
            raise ValueError(
                f"precise_ground.axis must be one of {sorted(_PRECISE_AXES)}, got {axis!r}"
            )

    for key in payload.get("exclude_inside") or ():
        if key not in scene_key_set:
            raise ValueError(f"exclude_inside {key!r} is not a scene key")

    hooks = payload.get("hooks")
    if hooks is not None and not (isinstance(hooks, str) and hooks.strip()):
        raise ValueError("hooks must be a non-empty string")
    for flag in ("recognize_movables", "exclude_movables_inside_placements"):
        if flag in payload and not isinstance(payload[flag], bool):
            raise ValueError(f"{flag} must be a boolean")


def spec_from_payload(payload: Mapping[str, Any]) -> TaskSpec:
    _validate_payload(payload)
    recognition = _section(payload, "recognition")
    adapters = _section(payload, "adapters")
    tracking = _section(payload, "tracking")
    commitment = _section(payload, "commitment")
    context_section = _section(payload, "context")
    scene_entries = list(payload.get("scene") or ())
    scene_objects = _build_scene_objects(scene_entries)
    classify = _bind_placement_classify(
        _build_classify(_section(payload, "classify")),
        scene_objects,
    )
    fallback_raw = recognition.get("category_fallback")
    category_fallback = None
    if fallback_raw is not None:
        category_fallback = CategoryFallbackSpec(
            classify=classify[fallback_raw["classify"]],
            grounding_prompt=fallback_raw.get("grounding_prompt", "{category} object"),
        )

    robust_payload = adapters.get("robust_detection") or {}
    seed_payload = tracking.get("known_bbox_seeds") or ()
    render_payload = tracking.get("segment_render_mode") or ()
    segment_rules = _build_segment_render_rules(render_payload) + _render_rules_from_scene(
        scene_entries
    )
    instruction = payload.get("instruction") or ""
    task_context = payload.get("task_context") or instruction
    plan = _section(payload, "plan")
    context_fields = _tuple_of_str(plan.get("fields"))
    exclude_inside = _tuple_of_str(payload.get("exclude_inside"))
    plugin = payload.get("hooks")

    return TaskSpec(
        name=payload["name"],
        plugin=plugin,
        landmark_recognition=recognition.get("landmark") or "",
        crop_recognition=recognition.get("crop"),
        landmark_parser=recognition.get("landmark_parser"),
        crop_parser=recognition.get("crop_parser"),
        crop_classify=recognition.get("crop_classify"),
        crop_category=recognition.get("crop_category") or "",
        crop_min_side=int(recognition.get("crop_min_side", 0)),
        category_fallback=category_fallback,
        grounding_task_context=task_context,
        context_fields=context_fields,
        known_bbox_seed_rules=_build_known_bbox_seed_rules(seed_payload),
        segment_render_rules=segment_rules,
        classify=classify,
        scene_objects=scene_objects,
        robust_detection=_build_robust_detection(robust_payload, classify),
        recognize_movables=payload.get("recognize_movables", True),
        exclude_movables_inside_placements=payload.get(
            "exclude_movables_inside_placements", True
        ),
        target_word=payload.get("target_word"),
        metadata=_metadata_from_plan_fields(payload.get("metadata"), context_fields),
        subtasks=_build_subtask_plan(context_section),
        free_target_phrase=commitment.get("free") or DEFAULT_FREE_TARGET_PHRASE,
        contact_target_phrase=commitment.get("contact") or DEFAULT_CONTACT_TARGET_PHRASE,
        strip_spatial_on_stages=_tuple_of_str(commitment.get("strip_spatial_on")),
        settings=dict(payload.get("settings") or {}),
        instruction=instruction,
        read_from=_build_read_from(payload.get("read_from")),
        precise_ground=_build_precise_ground(payload.get("precise_ground")),
        exclude_inside=exclude_inside,
    )


_SPEC_CACHE: dict[str, TaskSpec] = {}


def iter_task_spec_names(*, spec_dir: Path | None = None) -> tuple[str, ...]:
    """YAML stems in ``configs/tasks``, skipping ``_template`` and other ``_*`` files."""

    directory = spec_dir or SPEC_DIR
    return tuple(
        sorted(path.stem for path in directory.glob("*.yaml") if not path.name.startswith("_"))
    )


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return dict(payload)


def _load_yaml_tree(name: str, spec_dir: Path, stack: tuple[str, ...] = ()) -> dict[str, Any]:
    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise ValueError(f"circular task spec extends: {cycle}")
    path = spec_dir / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"task spec not found: {path}")
    payload = _read_yaml_mapping(path)
    extends = payload.pop("extends", None)
    if not extends:
        return payload
    parent = _load_yaml_tree(str(extends), spec_dir, (*stack, name))
    return _deep_merge(parent, payload)


def load_task_spec(name: str, *, spec_dir: Path | None = None) -> TaskSpec:
    """Load and cache ``configs/tasks/<name>.yaml``.

    PyYAML is required to parse task specs, including block scalars and lists
    of mappings.
    """
    directory = spec_dir or SPEC_DIR
    cache_key = f"{directory}/{name}"
    cached = _SPEC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = spec_from_payload(_load_yaml_tree(name, directory))
    _SPEC_CACHE[cache_key] = spec
    return spec


def clear_spec_cache() -> None:
    """Drop cached specs (tests / hot reload)."""
    _SPEC_CACHE.clear()
