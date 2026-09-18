"""Similarity-based target re-acquisition under sparse (skip-frame) inference.

Problem
-------
Online the wrist SAM2 mask jumps because inference runs only every 0.5-1s while
the wrist camera moves continuously. Between two inference calls the previous box
no longer encloses the target, so SAM2 re-prompted with a stale box drifts. This
script tests a *re-identification* alternative: at each (sparse) inference step,
detect the candidate letter blocks in the current frame and pick the one whose
appearance is most similar to the previously tracked target — i.e. re-acquire by
similarity instead of relying on a stale box.

Robustness harness
-------------------
The dataset is 30 Hz. We simulate inference every 0.5-1s by sampling every
``--skip`` frames (15 = 0.5s, 30 = 1s). Ground truth is the *dense* SAM2 video
track (propagated over every frame from the same seed box, exactly the offline
training tracker). At each sampled frame we compare the similarity-picked block's
bbox against the dense GT bbox (IoU). The per-skip success rate answers: "after
skipping N frames, does similarity re-ID still catch the right object?"

Run
---
    python -m perception.tracking.eval_similarity_reacquire \
        --video /dataset/.../videos/chunk-000/images.left_wrist/episode_000000.mp4 \
        --skips 15,20,30 --out-dir outputs/reid_eval

Use ``--gt none`` to skip the SAM2 GT (no GPU) and only dump the picked overlays
for visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from common.config_loader import DEFAULT_CONFIG_PATH, load_reasoning_config
from common.vision.crop import crop_xyxy
from perception.semantic_grounder.dino import DinoClient
from perception.tracking.appearance import describe, pick_most_similar
from perception.tracking.sam2_video import mask_to_bbox_xyxy

BBox = tuple[float, float, float, float]


# ----------------------------------------------------------------------------
# IO + geometry
# ----------------------------------------------------------------------------
def load_frames(video: Path, max_frames: int | None) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video))
    frames: list[Image.Image] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {video}")
    return frames


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_candidates(dino: DinoClient, frame: Image.Image, config: dict) -> list[BBox]:
    from task.task_hooks.letter import detect_letter_blocks

    blocks = detect_letter_blocks(dino, frame, config)
    return [tuple(float(v) for v in b.bbox_xyxy) for b in blocks if len(b.bbox_xyxy) == 4]


def dense_gt_track(frames: list[Image.Image], seed: BBox, config: dict) -> dict[int, BBox]:
    """Dense SAM2 video propagation from ``seed`` — the offline training tracker."""
    from perception.tracking.sam2_video_tracker import Sam2VideoSegmentTracker

    tracker = Sam2VideoSegmentTracker(config)
    masks = tracker.track_segment(frames, seed)
    gt: dict[int, BBox] = {}
    for idx, mask in masks.items():
        box = mask_to_bbox_xyxy(mask)
        if box is not None:
            gt[idx] = box
    return gt


# ----------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------
def run_skip(
    frames: list[Image.Image],
    skip: int,
    dino: DinoClient,
    kind: str,
    padding: int,
    seed_box: BBox,
    gt: dict[int, BBox] | None,
    iou_thr: float,
    update_ref: bool,
    out_dir: Path | None,
    config: dict,
) -> dict:
    reference = describe(crop_xyxy(frames[0], list(seed_box), padding=padding))
    sampled = list(range(skip, len(frames), skip))
    n_eval, n_hit, iou_sum = 0, 0, 0.0

    for t in sampled:
        candidates = detect_candidates(dino, frames[t], config)
        if not candidates:
            n_eval += 1  # a frame with no candidate is a miss for the tracker
            continue
        idx, score, picked_desc = pick_most_similar(
            reference, candidates, frames[t], kind, padding
        )
        if idx < 0:
            n_eval += 1
            continue
        picked = candidates[idx]

        if gt is not None:
            gt_box = gt.get(t)
            if gt_box is not None:
                this_iou = iou(picked, gt_box)
                iou_sum += this_iou
                n_hit += int(this_iou >= iou_thr)
                n_eval += 1
        if update_ref and picked_desc is not None:
            reference = picked_desc
        if out_dir is not None:
            _save_overlay(out_dir, skip, t, frames[t], picked, gt.get(t) if gt else None, score)

    success = (n_hit / n_eval) if (gt is not None and n_eval) else float("nan")
    mean_iou = (iou_sum / n_eval) if (gt is not None and n_eval) else float("nan")
    return {"skip": skip, "n_eval": n_eval, "success": success, "mean_iou": mean_iou}


def _save_overlay(
    out_dir: Path, skip: int, t: int, frame: Image.Image, picked: BBox, gt: BBox | None, score: float
) -> None:
    img = cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    if gt is not None:
        gx1, gy1, gx2, gy2 = (int(v) for v in gt)
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)  # GT = red
    px1, py1, px2, py2 = (int(v) for v in picked)
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 220, 0), 2)  # picked = green
    cv2.putText(img, f"sim={score:.2f}", (px1, max(12, py1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    sub = out_dir / f"skip_{skip:02d}"
    sub.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sub / f"frame_{t:05d}.png"), img)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="Wrist camera mp4 from the dataset.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--skips", default="15,20,30", help="Comma list of frame skips (30Hz: 15=0.5s,30=1s).")
    p.add_argument("--similarity", choices=("hsv", "gray", "combined"), default="combined")
    p.add_argument("--padding", type=int, default=4, help="Crop padding around each candidate box.")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--gt", choices=("sam2", "none"), default="sam2", help="GT track source.")
    p.add_argument("--init-bbox", default=None, help="Seed box x1,y1,x2,y2; else DINO top-1 at frame 0.")
    p.add_argument("--update-ref", action=argparse.BooleanOptionalAction, default=True,
                   help="Carry the picked crop forward as the new reference (track drift).")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out-dir", default=None, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_reasoning_config(args.config)
    dino = DinoClient(config=config)

    frames = load_frames(args.video, args.max_frames)
    print(f"loaded {len(frames)} frames from {args.video.name}")

    if args.init_bbox:
        seed = tuple(float(v) for v in args.init_bbox.split(","))
        if len(seed) != 4:
            raise SystemExit("--init-bbox must be x1,y1,x2,y2")
    else:
        cands = detect_candidates(dino, frames[0], config)
        if not cands:
            raise SystemExit("no letter block detected at frame 0; pass --init-bbox")
        seed = cands[0]  # DINO returns highest-confidence first
    print(f"seed box = {tuple(round(v, 1) for v in seed)}")

    gt = None
    if args.gt == "sam2":
        print("building dense SAM2 GT track ...")
        gt = dense_gt_track(frames, seed, config)
        print(f"  GT masks on {len(gt)}/{len(frames)} frames")

    out_dir = Path(args.out_dir) if args.out_dir else None
    skips = [int(s) for s in args.skips.split(",") if s.strip()]

    print(f"\n{'skip':>5} {'~sec':>5} {'n':>5} {'success':>9} {'mean_iou':>9}")
    for skip in skips:
        res = run_skip(frames, skip, dino, args.similarity, args.padding,
                       seed, gt, args.iou_thr, args.update_ref, out_dir, config)
        print(f"{res['skip']:>5} {skip / 30.0:>5.2f} {res['n_eval']:>5} "
              f"{res['success']:>9.3f} {res['mean_iou']:>9.3f}")
    if out_dir is not None:
        print(f"\noverlays saved under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
