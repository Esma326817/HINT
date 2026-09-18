from PIL import Image, ImageDraw, ImageFont

from common.vision.bbox import bbox_to_int
from common.task_types import TargetObject, PlacementCandidate


def _default_label_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_recognized_letter_blocks(
    image: Image.Image,
    blocks: list[TargetObject],
    *,
    target_block_id: int | None = None,
    outline: tuple[int, int, int] = (0, 255, 0),
    target_outline: tuple[int, int, int] = (255, 200, 0),
    line_width: int = 3,
    target_line_width: int = 4,
) -> Image.Image:
    """Return a copy of ``image`` with each block outlined and labeled (id + recognized letter)."""
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    font = _default_label_font()
    text_bg = (0, 0, 0)
    text_fg = (255, 255, 255)

    for block in blocks:
        x1, y1, x2, y2 = bbox_to_int(block.bbox_xyxy)
        is_target = target_block_id is not None and block.id == target_block_id
        color = target_outline if is_target else outline
        width = target_line_width if is_target else line_width
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"id={block.id} {block.letter or '?'}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tx1 = x1
        ty1 = max(0, y1 - text_h - 6)
        tx2 = tx1 + text_w + 6
        ty2 = ty1 + text_h + 4
        draw.rectangle([tx1, ty1, tx2, ty2], fill=text_bg)
        draw.text((tx1 + 3, ty1 + 2), label, fill=text_fg, font=font)

    return rendered


# The pi05 policy trains on the *_stage_rendered dataset whose global frames use a
# *semi-transparent* pick/place fill (measured alpha ~0.41 over the white board: a
# blue placement region tints the board light-blue rather than covering it solid).
# Keep the online render's alpha identical so inference matches the training data.
DEFAULT_FILL_ALPHA: int = 105  # ~0.41 * 255
DEFAULT_BLOCK_FILL: tuple[int, int, int, int] = (255, 0, 0, DEFAULT_FILL_ALPHA)
DEFAULT_PLACEMENT_FILL: tuple[int, int, int, int] = (0, 0, 255, DEFAULT_FILL_ALPHA)


def draw_pick_and_place(
    image: Image.Image,
    target_block: TargetObject | None,
    target_placement: PlacementCandidate | None,
    line_width: int = 3,
    *,
    block_fill: tuple[int, int, int, int] | None = None,
    placement_fill: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    block_fill = block_fill or DEFAULT_BLOCK_FILL
    placement_fill = placement_fill or DEFAULT_PLACEMENT_FILL

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if target_block is not None:
        draw.rectangle(bbox_to_int(target_block.bbox_xyxy), fill=block_fill)
    if target_placement is not None:
        draw.rectangle(bbox_to_int(target_placement.bbox_xyxy), fill=placement_fill)

    return Image.alpha_composite(base, overlay).convert("RGB")
