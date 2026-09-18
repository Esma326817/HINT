from common.task_types import BBox, TargetObject


def validate_bbox(bbox: BBox) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values")


def bbox_to_int(bbox: BBox) -> list[int]:
    validate_bbox(bbox)
    return [int(round(value)) for value in bbox]


def is_bbox_center_inside(container_bbox: BBox, bbox: BBox) -> bool:
    validate_bbox(container_bbox)
    validate_bbox(bbox)
    container_x1, container_y1, container_x2, container_y2 = container_bbox
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = min(container_x1, container_x2)
    right = max(container_x1, container_x2)
    top = min(container_y1, container_y2)
    bottom = max(container_y1, container_y2)
    return left <= center_x <= right and top <= center_y <= bottom


def filter_blocks_outside_bbox(blocks: list[TargetObject], container_bbox: BBox) -> list[TargetObject]:
    validate_bbox(container_bbox)
    return [block for block in blocks if not is_bbox_center_inside(container_bbox, block.bbox_xyxy)]


def target_object_region_above_board(cutting_board_bbox: BBox) -> list[float]:
    """The strip directly above the cutting board where the word's target object sits.

    Letter blocks are scattered on the table around the board; the target *object*
    that names the word (e.g. a box, a toy) is staged in this strip and is read by
    ``crop_above_cutting_board``. DINO detects it as a block-like region and Qwen,
    forced to emit a letter, mislabels it (a box -> "b"). Excluding this region keeps
    the target object from being read as a letter.
    """
    validate_bbox(cutting_board_bbox)
    x1, y1, x2, y2 = cutting_board_bbox
    left, right = min(x1, x2), max(x1, x2)
    top = min(y1, y2)
    return [float(left), 0.0, float(right), float(top)]


def filter_blocks_in_region(blocks: list[TargetObject], region: BBox) -> list[TargetObject]:
    """Drop blocks whose center falls inside ``region`` (e.g. the target-object strip)."""
    validate_bbox(region)
    return [block for block in blocks if not is_bbox_center_inside(region, block.bbox_xyxy)]


def clip_bbox(bbox: BBox, image_width: int, image_height: int) -> list[int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    x1, y1, x2, y2 = bbox_to_int(bbox)
    x1 = min(max(x1, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    x2 = min(max(x2, 0), image_width)
    y2 = min(max(y2, 0), image_height)
    return [x1, y1, x2, y2]
