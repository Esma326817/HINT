"""
# video: separate mp4 files under videos/
python scripts/delete_episode_batch.py \
    --root /mnt/hzx/data/lerobot/Zzming/peg_in_hole_v5_dagger \
    --list-file delete_episode.txt \
    --camera_type video

# image: images stored as parquet columns (no separate mp4)
python scripts/delete_episode_batch.py \
    --root /data/datasets/real_world/piper/lerobot/Zzming/shirt_eval_v1.1_bak \
    --list-file delete_episode.txt \
    --camera_type image

# delete_episode.txt (list file) format
# - One or more episode indices per line; separate with spaces or commas.
# - Inclusive ranges: LO-HI (e.g. 0-195 deletes 0 through 195). A token must not start with "-" to be a range.
# - Line comments: anything after # on a line is ignored.

Examples:
    1, 2, 3
    10 11 12
    0-5          # same as 0 1 2 3 4 5
"""
import argparse
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

CAMERA_SUBDIRS = (
    "images.global",
    "images.left_wrist",
    "images.right_wrist",
)


def load_chunks_size(info: dict) -> int:
    return int(info.get("chunks_size", info.get("chunk_size", 1000)))


def chunk_dirname(episode_index: int, chunk_size: int) -> str:
    return f"chunk-{episode_index // chunk_size:03d}"


def iter_chunk_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.glob("chunk-*") if p.is_dir())


def episode_parquet_path(root_dir: Path, episode_index: int, chunk_size: int) -> Path:
    return root_dir / "data" / chunk_dirname(episode_index, chunk_size) / f"episode_{episode_index:06d}.parquet"


def episode_video_path(root_dir: Path, episode_index: int, camera_subdir: str, chunk_size: int) -> Path:
    return root_dir / "videos" / chunk_dirname(episode_index, chunk_size) / camera_subdir / f"episode_{episode_index:06d}.mp4"


def find_episode_parquet(root_dir: Path, episode_index: int, chunk_size: int) -> Path | None:
    p = episode_parquet_path(root_dir, episode_index, chunk_size)
    if p.exists():
        return p
    matches = list((root_dir / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    return matches[0] if matches else None


def find_episode_video(root_dir: Path, episode_index: int, camera_subdir: str, chunk_size: int) -> Path | None:
    p = episode_video_path(root_dir, episode_index, camera_subdir, chunk_size)
    if p.exists():
        return p
    for chunk_dir in iter_chunk_dirs(root_dir / "videos"):
        cand = chunk_dir / camera_subdir / f"episode_{episode_index:06d}.mp4"
        if cand.exists():
            return cand
    return None

def _parquet_subtract_index(path: Path, delta: int) -> None:
    """Subtract delta from the index column in place; preserve parquet schema metadata."""
    table = pq.read_table(path)
    if "index" not in table.schema.names:
        return
    idx = table.column("index")
    j = table.schema.get_field_index("index")
    new_table = table.set_column(j, "index", pc.subtract(idx, pa.scalar(delta, type=idx.type)))
    pq.write_table(new_table.replace_schema_metadata(table.schema.metadata), path)


def _parquet_renumber_to_path_arrow(
    src: Path,
    dst: Path,
    *,
    new_episode_index: int,
    index_offset: int,
    apply_index_shift: bool,
) -> None:
    """Read src, update episode_index (and optional index shift), write to dst; keep HF parquet metadata."""
    table = pq.read_table(src)
    meta = table.schema.metadata
    n = table.num_rows

    if "episode_index" in table.schema.names:
        j = table.schema.get_field_index("episode_index")
        new_ep = pa.array([new_episode_index] * n, type=table.column("episode_index").type)
        table = table.set_column(j, "episode_index", new_ep)

    if apply_index_shift and "index" in table.schema.names:
        idx = table.column("index")
        j = table.schema.get_field_index("index")
        table = table.set_column(j, "index", pc.add(idx, pa.scalar(index_offset, type=idx.type)))

    pq.write_table(table.replace_schema_metadata(meta), dst)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _adjust_index_stats(stats_dict: dict, delta: int) -> None:
    """Shift stats.index min/max/mean lists by delta."""
    index_stats = stats_dict.get("index", {})
    for key in ("min", "max", "mean"):
        if isinstance(index_stats.get(key), list):
            index_stats[key] = [v + delta for v in index_stats[key]]


def parse_episode_indices_txt(path: Path) -> list[int]:
    """
    Parse episode indices to delete from a text file.
    Supports comma/whitespace-separated tokens, multiple lines, # comments, ranges (e.g. 0-195).
    Returns unique indices sorted descending (delete order).
    """
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            if "-" in token and not token.startswith("-"):
                lo, hi = token.split("-", 1)
                indices.extend(range(int(lo), int(hi) + 1))
            else:
                indices.append(int(token, 10))
    if not indices:
        raise ValueError(f"No episode indices parsed from {path}")
    return sorted(set(indices), reverse=True)



def delete_episode(
    root_dir: Path,
    target_episode_index: int,
    camera_type: str,
    *,
    verbose: bool = True,
):
    """
    Delete one episode by index; update meta and subsequent parquet index column.

    video: also delete mp4 under videos/, update total_videos; parquet via pandas.
    image: images in parquet columns only; PyArrow preserves HF schema metadata.
    """
    meta_dir = root_dir / "meta"
    episodes_jsonl = meta_dir / "episodes.jsonl"
    episodes_stats_jsonl = meta_dir / "episodes_stats.jsonl"
    info_json = meta_dir / "info.json"

    chunk_size = load_chunks_size(json.loads(info_json.read_text(encoding="utf-8")))

    # Load episodes.jsonl; split out target episode
    target_length = None
    remaining_episodes = []
    for ep in _read_jsonl(episodes_jsonl):
        if ep["episode_index"] == target_episode_index:
            target_length = ep["length"]
        else:
            remaining_episodes.append(ep)

    if target_length is None:
        raise ValueError(f"Episode index {target_episode_index} not found in {episodes_jsonl}")
    if verbose:
        print(f"✅ Found episode {target_episode_index} with length = {target_length}")

    # Delete mp4 files (video mode)
    video_deleted = 0
    if camera_type == "video":
        for subdir in CAMERA_SUBDIRS:
            video_path = find_episode_video(root_dir, target_episode_index, subdir, chunk_size)
            if video_path is not None and video_path.exists():
                video_path.unlink()
                video_deleted += 1
                if verbose:
                    print(f"🗑️ Deleted {video_path}")
            elif verbose:
                print(f"⚠️ Video not found ({subdir}): episode_{target_episode_index:06d}.mp4")

    # Delete episode parquet
    parquet_path = find_episode_parquet(root_dir, target_episode_index, chunk_size)
    if parquet_path is not None and parquet_path.exists():
        parquet_path.unlink()
        if verbose:
            print(f"🗑️ Deleted {parquet_path}")
    elif verbose:
        print(f"⚠️ Parquet file not found: {episode_parquet_path(root_dir, target_episode_index, chunk_size)}")

    # Shift index column for later episodes in parquet
    for chunk_dir in iter_chunk_dirs(root_dir / "data"):
        for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
            try:
                file_ep_idx = int(parquet_file.stem.split("_")[1])
            except (ValueError, IndexError):
                continue
            if file_ep_idx <= target_episode_index:
                continue

            if camera_type == "image":
                if "index" not in pq.ParquetFile(parquet_file).schema_arrow.names:
                    continue
                t0 = pq.read_table(parquet_file, columns=["index"]).column("index")
                omin, omax = pc.min(t0).as_py(), pc.max(t0).as_py()
                _parquet_subtract_index(parquet_file, target_length)
                if verbose:
                    t1 = pq.read_table(parquet_file, columns=["index"]).column("index")
                    print(f"📝 Updated index in {parquet_file.name}: [{omin}, {omax}] → [{pc.min(t1).as_py()}, {pc.max(t1).as_py()}]")
            else:
                df = pd.read_parquet(parquet_file)
                if "index" in df.columns:
                    omin, omax = df["index"].min(), df["index"].max()
                    df["index"] -= target_length
                    df.to_parquet(parquet_file, index=False)
                    if verbose:
                        print(f"📝 Updated index in {parquet_file.name}: [{omin}, {omax}] → [{df['index'].min()}, {df['index'].max()}]")

    # Rewrite episodes.jsonl
    _write_jsonl(episodes_jsonl, remaining_episodes)
    if verbose:
        print(f"✅ Updated {episodes_jsonl}")

    # Rewrite episodes_stats.jsonl
    remaining_stats = []
    should_adjust = False
    for stat in _read_jsonl(episodes_stats_jsonl):
        if stat.get("episode_index") == target_episode_index:
            should_adjust = True
            continue
        if should_adjust and "stats" in stat:
            _adjust_index_stats(stat["stats"], -target_length)
            if verbose:
                idx_s = stat["stats"].get("index", {})
                print(f"📊 Adjusted index stats for episode {stat.get('episode_index')}: min={idx_s.get('min')}, max={idx_s.get('max')}")
        remaining_stats.append(stat)
    _write_jsonl(episodes_stats_jsonl, remaining_stats)
    if verbose:
        print(f"✅ Updated {episodes_stats_jsonl}")

    # Update info.json
    info = json.loads(info_json.read_text(encoding="utf-8"))
    info["total_episodes"] = len(remaining_episodes)
    info["total_frames"] = sum(ep["length"] for ep in remaining_episodes)
    if camera_type == "video":
        info["total_videos"] -= video_deleted
    info_json.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"✅ Updated {info_json} (total_episodes={info['total_episodes']}, total_frames={info['total_frames']})")

    return (target_length, video_deleted) if camera_type == "video" else target_length


def renumber_episodes(root_dir: Path, camera_type: str, *, verbose: bool = True):
    """
    Renumber episodes to a contiguous 0..N-1 range.
    Updates episode_index and index stats in episodes_stats.jsonl;
    updates episode_index and index columns in each parquet file.

    video: also moves mp4 under videos/; parquet via pandas.
    image: PyArrow for parquet (HF schema metadata); no video files.
    """
    meta_dir = root_dir / "meta"
    info_json = meta_dir / "info.json"
    episodes_jsonl = meta_dir / "episodes.jsonl"
    episodes_stats_jsonl = meta_dir / "episodes_stats.jsonl"

    info = json.loads(info_json.read_text(encoding="utf-8"))
    chunk_size = load_chunks_size(info)
    data_root = root_dir / "data"
    videos_root = root_dir / "videos"

    # Collect all existing episode_index values on disk
    existing_indices: set[int] = set()
    for chunk_dir in iter_chunk_dirs(data_root):
        for f in chunk_dir.glob("episode_*.parquet"):
            try:
                existing_indices.add(int(f.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
    if camera_type == "video":
        for subdir in CAMERA_SUBDIRS:
            for chunk_dir in iter_chunk_dirs(videos_root):
                cam_dir = chunk_dir / subdir
                if not cam_dir.is_dir():
                    continue
                for f in cam_dir.glob("episode_*.mp4"):
                    try:
                        existing_indices.add(int(f.stem.split("_")[1]))
                    except (ValueError, IndexError):
                        continue

    if not existing_indices:
        if verbose:
            print("⚠️ No episodes found to renumber.")
        return

    sorted_indices = sorted(existing_indices)
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}
    if verbose:
        preview = dict(list(old_to_new.items())[:5])
        print(f"🔁 Renumbering episodes: {preview}{'...' if len(old_to_new) > 5 else ''}")

    # Cumulative global index offset map (old episode -> (old_start, new_start, length))
    episodes_data = sorted(_read_jsonl(episodes_jsonl), key=lambda x: x["episode_index"])
    cumulative = 0
    old_to_index_offset: dict[int, tuple[int, int, int]] = {}
    for ep in episodes_data:
        old_to_index_offset[ep["episode_index"]] = (cumulative, cumulative, ep["length"])
        cumulative += ep["length"]

    work_dir = root_dir / f"_renumber_work_{uuid.uuid4().hex}"
    pq_stage = work_dir / "parquet"
    vid_stage = work_dir / "videos"
    pq_stage.mkdir(parents=True, exist_ok=True)
    vid_stage.mkdir(parents=True, exist_ok=True)

    try:
        # Update parquet in memory and stage to work_dir
        for old_idx in sorted(old_to_new):
            old_path = find_episode_parquet(root_dir, old_idx, chunk_size)
            if old_path is None:
                if verbose:
                    print(f"⚠️ Parquet missing for old episode_index={old_idx}, skip")
                continue

            new_idx = old_to_new[old_idx]
            apply_shift = old_idx in old_to_index_offset
            index_offset = 0
            if apply_shift:
                old_start, new_start, _ = old_to_index_offset[old_idx]
                index_offset = new_start - old_start

            staged = pq_stage / f"from_{old_idx:06d}.parquet"

            if camera_type == "image":
                tab = pq.read_table(old_path)
                omin = omax = None
                if "index" in tab.schema.names and apply_shift:
                    idxc = tab.column("index")
                    omin, omax = pc.min(idxc).as_py(), pc.max(idxc).as_py()
                _parquet_renumber_to_path_arrow(
                    old_path, staged,
                    new_episode_index=new_idx,
                    index_offset=index_offset,
                    apply_index_shift=apply_shift,
                )
                old_path.unlink()
                if verbose:
                    if "episode_index" in tab.schema.names:
                        print(f"📝 Updated episode_index in parquet: {old_idx} → {new_idx}")
                    if omin is not None and apply_shift:
                        i2 = pq.read_table(staged, columns=["index"]).column("index")
                        print(f"📝 Updated index in parquet: [{omin}, {omax}] → [{pc.min(i2).as_py()}, {pc.max(i2).as_py()}] (offset={index_offset})")
                    print(f"📁 Staged parquet: {old_path} → (staging)")
            else:
                df = pd.read_parquet(old_path)
                if "episode_index" in df.columns:
                    df["episode_index"] = new_idx
                    if verbose:
                        print(f"📝 Updated episode_index in parquet: {old_idx} → {new_idx}")
                if "index" in df.columns and apply_shift:
                    omin, omax = df["index"].min(), df["index"].max()
                    df["index"] += index_offset
                    if verbose:
                        print(f"📝 Updated index in parquet: [{omin}, {omax}] → [{df['index'].min()}, {df['index'].max()}] (offset={index_offset})")
                df.to_parquet(staged, index=False)
                old_path.unlink()
                if verbose:
                    print(f"📁 Staged parquet: {old_path} → (staging)")

        for old_idx in sorted(old_to_new):
            staged = pq_stage / f"from_{old_idx:06d}.parquet"
            if not staged.exists():
                continue
            dest = episode_parquet_path(root_dir, old_to_new[old_idx], chunk_size)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            shutil.move(str(staged), str(dest))
            if verbose:
                print(f"📁 Placed parquet: → {dest}")

        # Move mp4 files (video mode)
        if camera_type == "video":
            for subdir in CAMERA_SUBDIRS:
                sub_stage = vid_stage / subdir.replace("observation.images.", "").replace(".", "_")
                sub_stage.mkdir(parents=True, exist_ok=True)
                for old_idx in sorted(old_to_new):
                    old_v = find_episode_video(root_dir, old_idx, subdir, chunk_size)
                    if old_v is None:
                        continue
                    staged_v = sub_stage / f"from_{old_idx:06d}.mp4"
                    shutil.move(str(old_v), str(staged_v))
                    if verbose:
                        print(f"🎥 Staged video ({subdir}): {old_v.name}")
                for old_idx in sorted(old_to_new):
                    staged_v = sub_stage / f"from_{old_idx:06d}.mp4"
                    if not staged_v.exists():
                        continue
                    dest_v = episode_video_path(root_dir, old_to_new[old_idx], subdir, chunk_size)
                    dest_v.parent.mkdir(parents=True, exist_ok=True)
                    dest_v.unlink(missing_ok=True)
                    shutil.move(str(staged_v), str(dest_v))
                    if verbose:
                        print(f"🎥 Placed video ({subdir}): → {dest_v}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Rewrite episodes.jsonl
    new_episodes = []
    for ep in episodes_data:
        old_idx = ep["episode_index"]
        if old_idx in old_to_new:
            ep["episode_index"] = old_to_new[old_idx]
            new_episodes.append(ep)
        elif verbose:
            print(f"⚠️ Skipping orphaned episode in meta: {ep}")
    _write_jsonl(episodes_jsonl, new_episodes)
    if verbose:
        print(f"✅ Updated episode_index in {episodes_jsonl}")

    # Rewrite episodes_stats.jsonl
    new_stats = []
    for stat in _read_jsonl(episodes_stats_jsonl):
        old_idx = stat.get("episode_index")
        if old_idx not in old_to_new:
            if verbose:
                print(f"⚠️ Skipping orphaned stats in meta: episode_index={old_idx}")
            continue

        new_idx = old_to_new[old_idx]
        stat["episode_index"] = new_idx

        if "stats" in stat:
            ep_stats = stat["stats"]
            if "episode_index" in ep_stats:
                for key in ("min", "max"):
                    if isinstance(ep_stats["episode_index"].get(key), list):
                        ep_stats["episode_index"][key] = [new_idx]
                if isinstance(ep_stats["episode_index"].get("mean"), list):
                    ep_stats["episode_index"]["mean"] = [float(new_idx)]

            if "index" in ep_stats and old_idx in old_to_index_offset:
                old_start, new_start, _ = old_to_index_offset[old_idx]
                offset = new_start - old_start
                _adjust_index_stats(ep_stats, offset)
                if verbose:
                    print(
                        f"📊 Updated stats for episode {old_idx} → {new_idx}: "
                        f"index offset={offset}, new range=[{ep_stats['index']['min'][0]}, {ep_stats['index']['max'][0]}]"
                    )

        new_stats.append(stat)
    _write_jsonl(episodes_stats_jsonl, new_stats)
    if verbose:
        print(f"✅ Updated episode_index and index in {episodes_stats_jsonl}")

    removed_temps = cleanup_temporary_files(root_dir)
    if verbose:
        for temp_path in removed_temps:
            print(f"🧹 Removed temporary file: {temp_path}")
        print("🎉 Renumbering completed!")


def cleanup_temporary_files(root_dir: Path) -> list[Path]:
    """Remove temp files from interrupted renumber; return paths removed."""
    removed: list[Path] = []
    data_root = root_dir / "data"
    if data_root.exists():
        for temp in data_root.rglob("temp_*.parquet"):
            temp.unlink(missing_ok=True)
            removed.append(temp)
    videos_root = root_dir / "videos"
    if videos_root.exists():
        for temp in videos_root.rglob("temp_*.mp4"):
            temp.unlink(missing_ok=True)
            removed.append(temp)
    for leftover in root_dir.glob("_renumber_work_*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
            removed.append(leftover)
    return removed


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Batch-delete LeRobot episodes from a list file, then renumber remaining episodes."
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (contains data/, meta/)")
    parser.add_argument(
        "--list-file",
        type=Path,
        default=repo_root / "delete_episode.txt",
        help=f"Episode index list file (default: {repo_root / 'delete_episode.txt'})",
    )
    parser.add_argument(
        "--camera_type",
        choices=["video", "image"],
        required=True,
        help=(
            "video: separate mp4 under videos/, parquet has no image columns; "
            "image: images as parquet columns (no separate mp4)"
        ),
    )
    args = parser.parse_args()
    root_dir = args.root.resolve()
    list_path = args.list_file.resolve()
    camera_type = args.camera_type
    info_json = root_dir / "meta" / "info.json"

    episodes_before = int(json.loads(info_json.read_text(encoding="utf-8"))["total_episodes"])
    indices = parse_episode_indices_txt(list_path)
    print(
        f"📋 List: {list_path}  camera_type={camera_type}\n"
        f"   episode_index to delete ({len(indices)} unique, descending delete order): {sorted(indices)}"
    )

    try:
        for ep in tqdm(indices, desc="Deleting episodes", unit="ep"):
            delete_episode(root_dir, ep, camera_type, verbose=False)
        print("🔁 Renumbering episodes...")
        renumber_episodes(root_dir, camera_type, verbose=False)

        episodes_after = int(json.loads(info_json.read_text(encoding="utf-8"))["total_episodes"])
        print(
            f"✅ Done. episode count: {episodes_before} → {episodes_after} "
            f"(deleted {episodes_before - episodes_after} this run)"
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
