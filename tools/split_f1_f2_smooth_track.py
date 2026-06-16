"""
Smooth-tracking crop: the crop window follows the fighter across the ring
but uses a Gaussian-smoothed bbox so it glides smoothly rather than
jumping every frame.

Algorithm per fighter per round:
  1. Build a bbox time-series (cx, cy, w, h) from all CUTIE frames.
  2. Apply a Gaussian kernel (SMOOTH_FRAMES wide) to each coordinate.
  3. Per frame: use smoothed (cx, cy) + median size + padding as crop window.
  4. Crop -> resize to 256 height (maintain aspect ratio) -> centre on
     256x256 black canvas.

Output: C:/Users/XRIG/Desktop/F1_F2_Output_SmoothTrack/
Naming: same as previous outputs
  F1/JP_OMeara/1_R1_JP_OMeara_part01.mp4
  F2/Jose_Manuel_Perez/2_R3_Jose_Manuel_Perez_part02.mp4 ...
"""

import cv2
import json
import csv
import numpy as np
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\XRIG\Downloads\F1_and_F2_annotations")
OUTPUT_DIR = Path(r"C:\Users\XRIG\Desktop\F1_F2_Output_SmoothTrack")
TARGET_SIZE   = 256
FRAMES_PER_PART = 1500
OUTPUT_FPS    = 30.0

# Gaussian smoothing window in frames (~1.5 seconds at 30fps).
# Larger = slower/smoother tracking. Smaller = more responsive but more jitter.
SMOOTH_FRAMES = 45

# Padding added around the median bbox size (not the noisy per-frame size).
PADDING_RATIO = 0.25

FIGHTER_NAMES = {"0": "JP_OMeara", "1": "Jose_Manuel_Perez"}
FIGHT_NUMS    = {"F1": "1", "F2": "2"}
EXCLUDE_KEYS  = ["visualization", "visualized", "sam3d", "bbox"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def find_source_video(folder: Path) -> Path | None:
    for f in folder.iterdir():
        if f.suffix.lower() not in (".mp4", ".m4v"):
            continue
        if any(k in f.name.lower() for k in EXCLUDE_KEYS):
            continue
        return f
    return None


def load_cutie_bboxes(cutie_path: Path) -> dict[str, dict[int, list[float]]]:
    with open(cutie_path) as fh:
        data = json.load(fh)
    result: dict[str, dict[int, list[float]]] = {}
    for fid, frames in data["fighters_tracking_data"].items():
        result[fid] = {int(e["frame"]): e["bbox"] for e in frames}
    return result


def gaussian_kernel(n: int) -> np.ndarray:
    """1-D Gaussian kernel of length n (odd)."""
    if n % 2 == 0:
        n += 1
    sigma = n / 6.0
    x = np.arange(n) - n // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_series(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with reflect padding so edges don't collapse to zero."""
    half = len(kernel) // 2
    padded = np.pad(values, half, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def build_smooth_windows(
    bbox_dict: dict[int, list[float]],
    total_frames: int,
    vid_w: int,
    vid_h: int,
) -> list[tuple[int, int, int, int]]:
    """
    Returns a list of (x1,y1,x2,y2) crop windows, one per source frame,
    smoothed so the window glides rather than jumping.
    """
    # Fill any missing frames with linear interpolation using neighbours
    all_frames = sorted(bbox_dict.keys())
    cx_raw = np.zeros(total_frames)
    cy_raw = np.zeros(total_frames)
    bw_raw = np.zeros(total_frames)
    bh_raw = np.zeros(total_frames)

    for fi in all_frames:
        x1, y1, x2, y2 = bbox_dict[fi]
        cx_raw[fi] = (x1 + x2) / 2
        cy_raw[fi] = (y1 + y2) / 2
        bw_raw[fi] = x2 - x1
        bh_raw[fi] = y2 - y1

    # Fill gaps by interpolating between known frames
    known = np.array(all_frames)
    for arr in (cx_raw, cy_raw, bw_raw, bh_raw):
        arr[:] = np.interp(np.arange(total_frames), known, arr[known])

    # Smooth cx/cy so the crop TRACKS the fighter but without jitter.
    # Keep median size (more stable than smoothing size independently).
    kernel = gaussian_kernel(SMOOTH_FRAMES)
    cx_sm = smooth_series(cx_raw, kernel)
    cy_sm = smooth_series(cy_raw, kernel)

    # Use median width/height across the whole round + padding (stable size).
    med_w = float(np.median(bw_raw[bw_raw > 0]))
    med_h = float(np.median(bh_raw[bh_raw > 0]))
    half_w = med_w / 2 + med_w * PADDING_RATIO
    half_h = med_h / 2 + med_h * PADDING_RATIO

    windows: list[tuple[int, int, int, int]] = []
    for i in range(total_frames):
        cx, cy = cx_sm[i], cy_sm[i]
        x1 = max(0, int(round(cx - half_w)))
        y1 = max(0, int(round(cy - half_h)))
        x2 = min(vid_w, int(round(cx + half_w)))
        y2 = min(vid_h, int(round(cy + half_h)))
        windows.append((x1, y1, x2, y2))

    return windows


def crop_and_pad(frame: np.ndarray, win: tuple[int, int, int, int]) -> np.ndarray:
    """Crop, scale to TARGET_SIZE height (aspect-ratio preserving), centre on black canvas."""
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


# ---------------------------------------------------------------------------
# per-round processing
# ---------------------------------------------------------------------------

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

    # Pre-compute smoothed window sequence for every fighter
    smooth_wins: dict[str, list[tuple[int,int,int,int]]] = {}
    for fid in fids:
        wins = build_smooth_windows(bboxes_all[fid], tot_fr, vid_w, vid_h)
        # Print first/last/mid window to show tracking
        sample_idxs = [0, tot_fr//4, tot_fr//2, 3*tot_fr//4, tot_fr-1]
        print(f"  fighter{fid} ({FIGHTER_NAMES[fid]}) smooth-track sample windows:")
        for si in sample_idxs:
            w = wins[si]
            print(f"    frame {si:5d}: ({w[0]},{w[1]},{w[2]},{w[3]})  {w[2]-w[0]}x{w[3]-w[1]}px")
        smooth_wins[fid] = wins

    # Output dirs
    out_dirs: dict[str, Path] = {}
    for fid in fids:
        d = OUTPUT_DIR / fight_id / FIGHTER_NAMES[fid]
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[fid] = d

    def vname(fid: str, part_n: int) -> str:
        return f"{fight_num}_{round_id}_{FIGHTER_NAMES[fid]}_part{part_n:02d}.mp4"

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
        n   = part_nums[fid]
        fn  = vname(fid, n)
        fc  = part_counts[fid]
        off = total_counts[fid]
        round_parts[fid].append({
            "part":               f"part{n:02d}",
            "file":               fn,
            "frame_count":        fc,
            "frame_offset":       off,
            "global_frame_range": [off + 1, off + fc],
        })
        manifest_rows.append({
            "fight":   fight_id,
            "round":   round_id,
            "fighter": FIGHTER_NAMES[fid],
            "part":    f"part{n:02d}",
            "path":    str(out_dirs[fid] / fn),
        })
        total_counts[fid] += fc
        part_nums[fid]    += 1
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
            win = smooth_wins[fid][frame_idx] if frame_idx < len(smooth_wins[fid]) else smooth_wins[fid][-1]
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
            "smooth_frames": SMOOTH_FRAMES,
            "frame_indexing_note": (
                "Parts concatenated into ONE continuous timeline. "
                f"part01 frames 1-{FRAMES_PER_PART}, part02 starts at {FRAMES_PER_PART+1}, etc. "
                f"Crop window tracks fighter position smoothed over {SMOOTH_FRAMES} frames."
            ),
            "parts": round_parts[fid],
        }
        jp = out_dirs[fid] / f"{round_id}.json"
        with open(jp, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"    fighter{fid}: {jp.name}  ({len(round_parts[fid])} parts, {total_counts[fid]} fr)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
