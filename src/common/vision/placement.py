from common.vision.bbox import bbox_to_int
from common.task_types import PlacementCandidate


MAX_PLACEMENT_SLOTS = 5
PLACEMENT_LEFT_MARGIN_X_FRAC = 0.21
PLACEMENT_RIGHT_MARGIN_X_FRAC = 0.19
PLACEMENT_MIN_GAP_X_FRAC = 0.015


def slot_label(index: int) -> str:
    """Placement label for the ``index``-th slot, so subtasks can name a target."""
    return f"slot_{index + 1}"


def build_placements_from_cutting_board(
    cutting_board_bbox: list[float] | list[int],
    count: int = 5,
) -> list[PlacementCandidate]:
    x1, y1, x2, y2 = bbox_to_int(cutting_board_bbox)
    board_width = x2 - x1
    board_height = y2 - y1
    if board_width <= 0 or board_height <= 0:
        raise ValueError("cutting board bbox must have positive width and height")
    if count <= 0:
        raise ValueError("placement count must be positive")

    count = min(count, MAX_PLACEMENT_SLOTS)

    board_cy = y1 + board_height / 2.0

    placement_width = max(1, int(round(board_width * 0.12)))
    placement_height = max(1, int(round(board_height * 0.32)))
    height_above_center = max(1, int(round(placement_height * 0.7)))
    height_below_center = max(1, placement_height - height_above_center)
    left_margin_x = max(1, int(round(board_width * PLACEMENT_LEFT_MARGIN_X_FRAC)))
    right_margin_x = max(1, int(round(board_width * PLACEMENT_RIGHT_MARGIN_X_FRAC)))
    min_gap_x = max(1, int(round(board_width * PLACEMENT_MIN_GAP_X_FRAC)))

    # Preserve a visible gap when all five slots are needed.  For shorter words
    # the data-derived margins below remain unchanged; for a packed five-letter
    # word they shrink proportionally instead of allowing slot overlap.
    available_margin_x = board_width - (
        count * placement_width + max(0, count - 1) * min_gap_x
    )
    desired_margin_x = left_margin_x + right_margin_x
    if available_margin_x < desired_margin_x:
        margin_scale = max(2, available_margin_x) / desired_margin_x
        left_margin_x = max(1, int(round(left_margin_x * margin_scale)))
        right_margin_x = max(1, available_margin_x - left_margin_x)

    # Use the complete board as one stable coordinate system.  The outer slot
    # edges retain deliberate margins and all slot centres are evenly spaced.
    # Human demonstrations leave slightly more room on the left than the right.
    first_center_x = x1 + left_margin_x + placement_width / 2.0
    last_center_x = x2 - right_margin_x - placement_width / 2.0
    center_spacing = (
        (last_center_x - first_center_x) / (count - 1)
        if count > 1
        else 0.0
    )
    if count == 1:
        first_center_x = x1 + board_width / 2.0

    placements: list[PlacementCandidate] = []
    for idx in range(count):
        center_x = first_center_x + idx * center_spacing
        center_y = board_cy
        left = int(round(center_x - placement_width / 2.0))
        right = int(round(center_x + placement_width / 2.0))
        top = int(round(center_y - height_above_center))
        bottom = int(round(center_y + height_below_center))
        placements.append(
            PlacementCandidate(
                id=idx,
                bbox_xyxy=[left, top, right, bottom],
                label=slot_label(idx),
            )
        )

    return placements
