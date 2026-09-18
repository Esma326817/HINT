from PIL import Image

from common.vision.bbox import bbox_to_int, clip_bbox
from common.task_types import BBox


def crop_xyxy(image: Image.Image, bbox: BBox, padding: int = 0) -> Image.Image:
    if padding < 0:
        raise ValueError("padding must be non-negative")

    x1, y1, x2, y2 = bbox_to_int(bbox)
    padded_bbox = [x1 - padding, y1 - padding, x2 + padding, y2 + padding]
    clipped_bbox = clip_bbox(padded_bbox, image.width, image.height)
    cx1, cy1, cx2, cy2 = clipped_bbox

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("bbox does not produce a valid crop")

    return image.crop((cx1, cy1, cx2, cy2))


def crop_relative(image: Image.Image, bbox: BBox, region: str) -> Image.Image:
    """Crop a strip of ``image`` relative to an already-grounded box."""
    x1, y1, x2, y2 = bbox_to_int(bbox)
    width, height = image.size
    kind = str(region or "above").strip().lower()
    if kind == "above":
        box = [x1, 0, x2, y1]
    elif kind == "below":
        box = [x1, y2, x2, height]
    elif kind == "inside":
        box = [x1, y1, x2, y2]
    elif kind in {"left", "left_of"}:
        box = [0, y1, x1, y2]
    elif kind in {"right", "right_of"}:
        box = [x2, y1, width, y2]
    else:
        raise ValueError(
            f"unsupported read_from region {region!r}; "
            "use above, below, inside, left_of, or right_of"
        )
    try:
        return crop_xyxy(image, box)
    except ValueError as exc:
        raise ValueError(f"bbox does not leave a valid {kind} crop") from exc


def crop_above_cutting_board(image: Image.Image, cutting_board_bbox: BBox) -> Image.Image:
    try:
        return crop_relative(image, cutting_board_bbox, "above")
    except ValueError as exc:
        raise ValueError("cutting board bbox does not leave a valid target-word crop") from exc


def batch_crop(image: Image.Image, bboxes: list[BBox], padding: int = 0) -> list[Image.Image]:
    return [crop_xyxy(image, bbox, padding=padding) for bbox in bboxes]
