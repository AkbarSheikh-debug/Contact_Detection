"""
Stable-crop version: one fixed crop window per 1500-frame part per fighter.
The window is derived from the median bbox center + median bbox size (+60% pad)
across that chunk, so the viewport does not move between frames and matches
the Gladius_Output style exactly.

Output: C:/Users/XRIG/Desktop/F1_F2_Output_Stable/
Names : same as F1_F2_Output
  F1/JP_OMeara/1_R1_JP_OMeara_part01.mp4  ...
  F2/Jose_Manuel_Perez/2_R3_Jose_Manuel_Perez_part02.mp4  ...
"""

import cv2
import json
import csv
import numpy as np
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\XRIG\Downloads\F1_and_F2_annotations")
OUTPUT_DIR = Path(r"C:\Users\XRIG\Desktop\F1_F2_Output_Stable")
TARGET_SIZE = 256
FRAMES_PER_PART = 1500
OUTPUT_FPS = 30.0
PADDING_RATIO = 0.60   # 60% of median bbox size added as padding on each side

FIGHTER_NAMES = {"0": "JP_OMeara", "1": "Jose_Manuel_Perez"}
FIGHT_NUMS    = {"F1": "1", "F2": "2"}
EXCLUDE_KEYS  = ["visualization", "visualized", "sam3d", "bbox"]


def find_source_video(folder: Path) -> Path | None:
    for f in folder.iterdir():
        if f.suffix.lower() not in (".mp4", ".m4v"):
            continue
        if any(k in f.name.lower() for k in EXCLUDE_KEYS):
            continue
        return f
    return None


def load_cutie_bboxes(cutie_path: Path) -> dict[str, list[tuple[int, list[float]]]]:
    """Returns {fid: sorted list of (frame_idx, [x1,y1,x2,y2])}."""
    with open(cutie_path) as fh:
        data = json.load(fh)
    result: dict[str, list] = {}
    for fid, frames in data["fighters_tracking_data"].items():
        result[fid] = sorted((int(e["frame"]), e["bbox"]) for e in frames)
    return result


def compute_chunk_window(
    bboxes_in_chunk: list[list[float]],
    vid_w: int,
    vid_h: int,
) -> tuple[int, int, int, int]:
    """
    Fixed crop window for a 1500-frame chunk.
    Uses median bbox center + median bbox size + PADDING_RATIO padding.
    """
    arr = np.array(bboxes_in_chunk, dtype=float)  # (N, 4): x1 y1 x2 y2
    cx = np.median((arr[:, 0] + arr[:, 2]) / 2)
    cy = np.median((arr[:, 1] + arr[:, 3]) / 2)
    mw = np.median(arr[:, 2] - arr[:, 0])
    mh = np.median(arr[:, 3] - arr[:, 1])
    half_w = mw / 2 + mw * PADDING_RATIO
    half_h = mh / 2 + mh * PADDING_RATIO
    x1 = max(0, int(cx - half_w))
    y1 = max(0, int(cy - half_h))
    x2 = min(vid_w, int(cx + half_w))
    y2 = min(vid_h, int(cy + half_h))
    return x1, y1, x2, y2


def crop_and_pad(frame: np.ndarray, win: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop the fixed window, scale to TARGET_SIZE height (maintain aspect ratio),
    centre on TARGET_SIZExTARGET_SIZE black canvas.
    """
    x1, y1, x2, y2 = win
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    scale = TARGET_SIZE / ch
    new_h = TARGET_SIZE
    new_w = max(1, int(cw * scale))
    if new_w > TARGET_SIZE:
        scale = TARGET_SIZE / cw
        new_w = TARGET_SIZE
        new_h = max(1, int(ch * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    y_off = (TARGET_SIZE - new_h) // 2
    x_off = (TARGET_SIZE - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def open_writer(path: Path) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, OUTPUT_FPS, (TARGET_SIZE, TARGET_SIZE))


def process_round(
    folder: Path,
    fight_id: str,
    round_id: str,
    manifest_rows: list[dict],
) -> None:
    src_video = find_source_video(folder)
    if src_video is None:
        print(f"  [SKIP] No source video in {folder.name}")
        return

    cutie_path = next(
        (p for p in folder.iterdir() if "cutie" in p.name.lower() and p.suffix == ".json"),
        None,
    )
    if cutie_path is None:
        print(f"  [SKIP] No CUTIE JSON in {folder.name}")
        return

    print(f"\n=== {fight_id} {round_id} -- {src_video.name} ===")

    bboxes_all = load_cutie_bboxes(cutie_path)
    fids = sorted(bboxes_all.keys())

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {src_video}")
        return

    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    tot_fr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fight_num = FIGHT_NUMS[fight_id]
    print(f"  {tot_fr} fr @ {src_fps:.3f} fps  {vid_w}x{vid_h}")

    # Pre-compute per-chunk stable windows for every fighter
    # chunk k covers source frames [k*FRAMES_PER_PART, (k+1)*FRAMES_PER_PART)
    n_chunks = (tot_fr + FRAMES_PER_PART - 1) // FRAMES_PER_PART
    chunk_windows: dict[str, list[tuple[int,int,int,int]]] = {}
    for fid in fids:
        bbox_dict = dict(bboxes_all[fid])  # frame -> bbox
        windows = []
        for k in range(n_chunks):
            start = k * FRAMES_PER_PART
            end   = min(tot_fr, start + FRAMES_PER_PART)
            chunk_bboxes = [bbox_dict[i] for i in range(start, end) if i in bbox_dict]
            if not chunk_bboxes:
                # fallback: use window from previous chunk
                chunk_bboxes = [bbox_dict[i] for i in bbox_dict if i < end]
            win = compute_chunk_window(chunk_bboxes, vid_w, vid_h)
            cw = win[2]-win[0]; ch = win[3]-win[1]
            print(f"    fighter{fid} ({FIGHTER_NAMES[fid]}) chunk{k+1}: {win}  ({cw}x{ch}px)")
            windows.append(win)
        chunk_windows[fid] = windows

    # Output directories
    out_dirs: dict[str, Path] = {}
    for fid in fids:
        d = OUTPUT_DIR / fight_id / FIGHTER_NAMES[fid]
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[fid] = d

    def vname(fid: str, part_n: int) -> str:
        return f"{fight_num}_{round_id}_{FIGHTER_NAMES[fid]}_part{part_n:02d}.mp4"

    # Per-fighter state
    writers:      dict[str, cv2.VideoWriter | None] = {fid: None for fid in fids}
    part_nums:    dict[str, int]  = {fid: 1 for fid in fids}
    part_counts:  dict[str, int]  = {fid: 0 for fid in fids}
    total_counts: dict[str, int]  = {fid: 0 for fid in fids}
    round_parts:  dict[str, list] = {fid: [] for fid in fids}

    def start_part(fid: str) -> None:
        path = out_dirs[fid] / vname(fid, part_nums[fid])
        writers[fid] = open_writer(path)
        part_counts[fid] = 0
        print(f"    fighter{fid}: {path.name}")

    def close_part(fid: str, is_last: bool = False) -> None:
        w = writers[fid]
        if w is None:
            return
        w.release()
        writers[fid] = None
        n  = part_nums[fid]
        fn = vname(fid, n)
        fc = part_counts[fid]
        off = total_counts[fid]
        round_parts[fid].append({
            "part": f"part{n:02d}",
            "file": fn,
            "frame_count": fc,
            "frame_offset": off,
            "global_frame_range": [off + 1, off + fc],
            "stable_crop_window": list(chunk_windows[fid][n - 1]),
        })
        manifest_rows.append({
            "fight":   fight_id,
            "round":   round_id,
            "fighter": FIGHTER_NAMES[fid],
            "part":    f"part{n:02d}",
            "path":    str(out_dirs[fid] / fn),
        })
        total_counts[fid] += fc
        part_nums[fid]   += 1
        if not is_last:
            start_part(fid)

    for fid in fids:
        start_part(fid)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for fid in fids:
            chunk_k = frame_idx // FRAMES_PER_PART
            win = chunk_windows[fid][chunk_k] if chunk_k < len(chunk_windows[fid]) else chunk_windows[fid][-1]
            writers[fid].write(crop_and_pad(frame, win))
            part_counts[fid] += 1
            if part_counts[fid] >= FRAMES_PER_PART:
                close_part(fid)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    ... {frame_idx}/{tot_fr}")

    cap.release()

    for fid in fids:
        if writers[fid] is not None:
            close_part(fid, is_last=True)

    for fid in fids:
        summary = {
            "round":         round_id,
            "fight":         fight_id,
            "fighter":       FIGHTER_NAMES[fid],
            "part_count":    len(round_parts[fid]),
            "total_frames":  total_counts[fid],
            "source_fps":    round(src_fps, 4),
            "output_fps":    OUTPUT_FPS,
            "output_size":   TARGET_SIZE,
            "frame_indexing_note": (
                "Parts concatenated into ONE continuous timeline. "
                f"Each part k offset by sum of preceding frame counts "
                f"(e.g. part01 frames 1-{FRAMES_PER_PART} -> part02 starts at {FRAMES_PER_PART+1}). "
                "stable_crop_window per part stored in parts array."
            ),
            "parts": round_parts[fid],
        }
        jp = out_dirs[fid] / f"{round_id}.json"
        with open(jp, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"    fighter{fid}: {jp.name}  ({len(round_parts[fid])} parts, {total_counts[fid]} fr)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for folder in sorted(SOURCE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split()
        if len(parts) != 2 or parts[0] not in FIGHT_NUMS:
            print(f"[SKIP] {folder.name}")
            continue
        process_round(folder, parts[0], parts[1], manifest_rows)

    manifest_path = OUTPUT_DIR / "label_studio_manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fight","round","fighter","part","path"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nManifest: {manifest_path}  ({len(manifest_rows)} entries)")
    print("Done.")


if __name__ == "__main__":
    main()
