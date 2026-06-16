"""
Splits F1_and_F2_annotations into per-fighter 1500-frame parts at 30fps.

Source: F1_and_F2_annotations/{F1|F2} {R}/
Output: F1_F2_Output/{F1|F2}/{fighter0|fighter1}/

Per output video:
  - Frames cropped from source using CUTIE bbox (+20% padding), resized to 256x256
  - Re-encoded at 30fps (source is ~25fps, re-timestamped)
  - 1500 frames per part (last part may be shorter)
  - Companion JSON index + per-fight Round summary JSON
  - label_studio_manifest.csv at output root
"""

import cv2
import json
import os
import csv
import sys
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\XRIG\Downloads\F1_and_F2_annotations")
OUTPUT_DIR = Path(r"C:\Users\XRIG\Desktop\F1_F2_Output")
TARGET_SIZE = 256
FRAMES_PER_PART = 1500
OUTPUT_FPS = 30.0
PADDING_RATIO = 0.20  # 20% bbox padding on each side

# Source video suffixes to exclude (visualisations/debug outputs)
EXCLUDE_KEYWORDS = ["visualization", "visualized", "sam3d", "bbox", "visualization"]


def find_source_video(folder: Path) -> Path | None:
    for f in folder.iterdir():
        if f.suffix.lower() not in (".mp4", ".m4v"):
            continue
        low = f.name.lower()
        if any(k in low for k in EXCLUDE_KEYWORDS):
            continue
        return f
    return None


def load_cutie_bboxes(cutie_path: Path) -> dict[int, dict[int, list[float]]]:
    """Returns {fighter_id: {frame_idx: [x1,y1,x2,y2]}}."""
    with open(cutie_path) as fh:
        data = json.load(fh)
    tracking = data["fighters_tracking_data"]
    result: dict[int, dict[int, list[float]]] = {}
    for fid_str, frames in tracking.items():
        fid = int(fid_str)
        result[fid] = {}
        for entry in frames:
            result[fid][int(entry["frame"])] = entry["bbox"]
    return result


def crop_frame(frame, bbox: list[float], pad: float, size: int):
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    ix1 = max(0, int(x1 - px))
    iy1 = max(0, int(y1 - py))
    ix2 = min(frame.shape[1], int(x2 + px))
    iy2 = min(frame.shape[0], int(y2 + py))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size))


def open_writer(path: Path, fps: float, size: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (size, size))


def process_round(
    folder: Path,
    fight_id: str,
    round_id: str,
    manifest_rows: list[dict],
) -> None:
    src_video = find_source_video(folder)
    if src_video is None:
        print(f"  [SKIP] No source video found in {folder}")
        return

    cutie_path = next(
        (p for p in folder.iterdir() if "cutie" in p.name.lower() and p.suffix == ".json"),
        None,
    )
    if cutie_path is None:
        print(f"  [SKIP] No CUTIE JSON found in {folder}")
        return

    print(f"\n=== {fight_id} {round_id} — {src_video.name} ===")
    bboxes = load_cutie_bboxes(cutie_path)
    fighter_ids = sorted(bboxes.keys())

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {src_video}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Source: {total_frames} frames @ {src_fps:.3f} fps -> output @ {OUTPUT_FPS} fps")

    # Build per-fighter state
    writers: dict[int, cv2.VideoWriter | None] = {fid: None for fid in fighter_ids}
    part_nums: dict[int, int] = {fid: 1 for fid in fighter_ids}
    part_frame_counts: dict[int, int] = {fid: 0 for fid in fighter_ids}
    part_total_frames: dict[int, int] = {fid: 0 for fid in fighter_ids}
    last_bbox: dict[int, list[float]] = {}
    round_parts: dict[int, list[dict]] = {fid: [] for fid in fighter_ids}

    # Output dirs
    out_dirs: dict[int, Path] = {}
    for fid in fighter_ids:
        d = OUTPUT_DIR / fight_id / f"fighter{fid}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[fid] = d

    def start_part(fid: int):
        part_n = part_nums[fid]
        fname = f"{round_id}_fighter{fid}_part{part_n:02d}.mp4"
        path = out_dirs[fid] / fname
        writers[fid] = open_writer(path, OUTPUT_FPS, TARGET_SIZE)
        part_frame_counts[fid] = 0
        print(f"    fighter{fid}: starting {fname}")

    def close_part(fid: int, is_last: bool = False):
        w = writers[fid]
        if w is None:
            return
        w.release()
        writers[fid] = None
        part_n = part_nums[fid]
        fname = f"{round_id}_fighter{fid}_part{part_n:02d}.mp4"
        fc = part_frame_counts[fid]
        offset = part_total_frames[fid]
        round_parts[fid].append(
            {
                "part": f"part{part_n:02d}",
                "file": fname,
                "frame_count": fc,
                "frame_offset": offset,
                "global_frame_range": [offset + 1, offset + fc],
            }
        )
        manifest_rows.append(
            {
                "fight": fight_id,
                "round": round_id,
                "fighter": f"fighter{fid}",
                "part": f"part{part_n:02d}",
                "path": str(out_dirs[fid] / fname),
            }
        )
        part_total_frames[fid] += fc
        part_nums[fid] += 1
        if not is_last:
            start_part(fid)

    # Open first parts
    for fid in fighter_ids:
        start_part(fid)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for fid in fighter_ids:
            bbox = bboxes[fid].get(frame_idx)
            if bbox is None:
                bbox = last_bbox.get(fid)
            else:
                last_bbox[fid] = bbox

            if bbox is None:
                frame_idx += 1
                continue

            cropped = crop_frame(frame, bbox, PADDING_RATIO, TARGET_SIZE)
            writers[fid].write(cropped)
            part_frame_counts[fid] += 1

            if part_frame_counts[fid] >= FRAMES_PER_PART:
                close_part(fid)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    ... frame {frame_idx}/{total_frames}")

    cap.release()

    # Close remaining open parts
    for fid in fighter_ids:
        if writers[fid] is not None:
            close_part(fid, is_last=True)

    # Write per-fighter Round JSON
    for fid in fighter_ids:
        summary = {
            "round": round_id,
            "fight": fight_id,
            "fighter": f"fighter{fid}",
            "part_count": len(round_parts[fid]),
            "total_frames": part_total_frames[fid],
            "source_fps": round(src_fps, 4),
            "output_fps": OUTPUT_FPS,
            "output_size": TARGET_SIZE,
            "frame_indexing_note": (
                "Parts concatenated into ONE continuous timeline. "
                "Each part k is offset by the sum of preceding parts' frame counts "
                f"(e.g. part01 frames 1-{FRAMES_PER_PART} → part02 starts at {FRAMES_PER_PART + 1})."
            ),
            "parts": round_parts[fid],
        }
        json_path = out_dirs[fid] / f"{round_id}.json"
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"    fighter{fid}: wrote {json_path.name} ({len(round_parts[fid])} parts, {part_total_frames[fid]} frames)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    folders = sorted(SOURCE_DIR.iterdir())
    for folder in folders:
        if not folder.is_dir():
            continue
        name = folder.name  # e.g. "F1 R1"
        parts = name.split()
        if len(parts) != 2:
            print(f"[SKIP] Unexpected folder name: {name}")
            continue
        fight_id, round_id = parts[0], parts[1]  # "F1", "R1"
        process_round(folder, fight_id, round_id, manifest_rows)

    # Write manifest CSV
    manifest_path = OUTPUT_DIR / "label_studio_manifest.csv"
    fieldnames = ["fight", "round", "fighter", "part", "path"]
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote manifest: {manifest_path} ({len(manifest_rows)} entries)")
    print("Done.")


if __name__ == "__main__":
    main()
