#!/usr/bin/env python3
"""
Prepare Lillyella vs Zoe dataset for impact detection training.

Does everything in one pass:
  1. Reads Gladius action labels (fighter0 + fighter1) and converts
     local part-frame numbers -> global video frame numbers.
  2. Merges SAM3D fighter0 + fighter1 JSON files into v8.py-compatible format.
  3. Copies / links CUTIE tracking data.
  4. Extracts one video clip per action window (with configurable padding).
  5. Writes a manifest.json for the annotation tool.

Usage:
    python dataset/prepare_lillyella_dataset.py
    python dataset/prepare_lillyella_dataset.py --rounds 3 4 5 8
    python dataset/prepare_lillyella_dataset.py --no-clips   # skip extraction
"""

import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
FFMPEG = r"C:\Users\XRIG\Downloads\ffmpeg_extracted\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

VIDEO_FOLDER   = r"C:\Users\XRIG\Downloads\drive-download-20260615T202203Z-3-001\(BLUE) Lillyella Craw Seaman VS (RED) Zoe Hunte-Smith"
GLADIUS_FOLDER = r"C:\Users\XRIG\Desktop\Gladius_Output\(BLUE)_Lillyella_Craw_Seaman_VS_(RED)_Zoe_Hunte-Smith"
OUT_BASE       = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\dataset\lillyella_vs_zoe"

CLIPS_DIR      = os.path.join(OUT_BASE, "clips")
MANIFEST_PATH  = os.path.join(OUT_BASE, "manifest.json")

# Padding around each action window when cutting clips
PAD_BEFORE = 8   # frames before window_start
PAD_END    = 20  # frames after window_end  (captures post-impact reaction)

# Rounds to process (only those with video + SAM3D for both fighters)
ALL_ROUNDS = [3, 4, 5, 8]


# ── Helpers ───────────────────────────────────────────────────────────────────

def gladius_labels_to_global(round_id, fighter_id):
    """Load Gladius action labels for one round/fighter and return a list of
    dicts with global frame numbers (1-indexed, matching video timeline)."""
    fid = f"fighter{fighter_id}"
    path = os.path.join(GLADIUS_FOLDER, fid, f"Round{round_id}.json")
    if not os.path.exists(path):
        print(f"  [WARN] Gladius labels not found: {path}")
        return []

    d = json.load(open(path))
    parts = {p["part"]: p for p in d["parts"]}

    actions = []
    for lb in d["labels"]:
        part_name = lb["part"]
        if part_name not in parts:
            continue
        offset = parts[part_name]["frame_offset"]
        global_start = offset + lb["start"]
        global_end   = offset + lb["end"]
        actions.append({
            "fighter_id":    fighter_id,
            "fighter_type":  f"fighter_{fighter_id}",
            "action":        lb["label"],
            "window_start":  global_start,
            "window_end":    global_end,
            "part":          part_name,
            "confidence":    1.0,
        })
    return actions


def merge_sam3d(round_id):
    """Merge Round{N}_SAM3D_fighter0.json + fighter1.json into one dict
    compatible with v8.py's {0: [...], 1: [...]} format.
    Returns merged dict or None if files are missing."""
    merged = {}
    for fid in (0, 1):
        p = os.path.join(VIDEO_FOLDER, f"Round{round_id}_SAM3D_fighter{fid}.json")
        if not os.path.exists(p):
            print(f"  [WARN] SAM3D not found: {p}")
            continue
        d = json.load(open(p))
        kp = d.get("keypoints_3d", {})
        # fighter0 file stores data under key "0", fighter1 under key "0" too
        # (each file only contains its own fighter)
        raw = kp.get("0") or kp.get(str(fid)) or next(iter(kp.values()), [])
        merged[fid] = raw
        print(f"    SAM3D fighter{fid}: {len(raw)} frame entries")
    return merged if merged else None


def get_video_fps(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def extract_clip_ffmpeg(video_path, start_frame, end_frame, fps, out_path,
                        pad_before=PAD_BEFORE, pad_end=PAD_END):
    """Extract [start_frame-pad, end_frame+pad] from video using ffmpeg."""
    s_frame = max(0, start_frame - pad_before)
    e_frame  = end_frame + pad_end
    # Convert to seconds (frames are 1-indexed)
    t_start = (s_frame - 1) / fps
    t_end   = (e_frame)     / fps
    t_start = max(0.0, t_start)

    cmd = [
        FFMPEG, "-y",
        "-ss", f"{t_start:.6f}",
        "-to", f"{t_end:.6f}",
        "-i", video_path,
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-an",                     # no audio needed for visual annotation
        out_path,
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERR] ffmpeg: {result.stderr[:200]}")
        return False
    return True


def extract_clip_opencv(video_path, start_frame, end_frame, fps, out_path,
                        pad_before=PAD_BEFORE, pad_end=PAD_END):
    """OpenCV fallback clip extractor."""
    import cv2
    s_frame = max(0, start_frame - 1 - pad_before)   # 0-indexed
    e_frame  = end_frame - 1 + pad_end
    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
    for _ in range(e_frame - s_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def process_round(round_id, extract_clips=True):
    print(f"\n{'='*60}")
    print(f"  Round {round_id}")
    print(f"{'='*60}")

    round_dir = os.path.join(OUT_BASE, f"Round{round_id}")
    os.makedirs(round_dir, exist_ok=True)

    # ── 1. Merge action labels (fighter0 + fighter1) ──────────────────────────
    actions = []
    for fid in (0, 1):
        fa = gladius_labels_to_global(round_id, fid)
        actions.extend(fa)
        print(f"  fighter{fid}: {len(fa)} action windows")

    # Sort by window start for clean output
    actions.sort(key=lambda a: a["window_start"])
    actions_path = os.path.join(round_dir, "actions.json")
    json.dump({"fight": "lillyella_vs_zoe", "round": round_id,
               "actions": actions}, open(actions_path, "w"), indent=2)
    print(f"  -> {len(actions)} total actions saved to {actions_path}")

    # ── 2. Merge SAM3D ────────────────────────────────────────────────────────
    sam3d = merge_sam3d(round_id)
    if sam3d:
        sam3d_path = os.path.join(round_dir, "sam3d.json")
        # Save in v8.py compatible format (string keys "0" and "1")
        json.dump({"0": sam3d.get(0, []),
                   "1": sam3d.get(1, []),
                   "contact_events": []},
                  open(sam3d_path, "w"), indent=2)
        print(f"  -> SAM3D merged: {sam3d_path}")
    else:
        print(f"  [WARN] No SAM3D data available for Round{round_id}")

    # ── 3. Copy CUTIE tracking JSON ───────────────────────────────────────────
    cutie_src = os.path.join(VIDEO_FOLDER, f"Round{round_id}.json")
    if os.path.exists(cutie_src):
        cutie_dst = os.path.join(round_dir, "tracking.json")
        shutil.copy2(cutie_src, cutie_dst)
        print(f"  -> Tracking JSON copied")

    # ── 4. Extract video clips ────────────────────────────────────────────────
    video_path = os.path.join(VIDEO_FOLDER, f"Round{round_id}.mp4")
    if not os.path.exists(video_path):
        print(f"  [WARN] Video not found: {video_path} — skipping clip extraction")
        return [], actions

    fps, total_frames = get_video_fps(video_path)
    print(f"  Video: {total_frames} frames @ {fps:.3f} fps")

    clips_meta = []
    if extract_clips:
        os.makedirs(CLIPS_DIR, exist_ok=True)
        use_ffmpeg = os.path.exists(FFMPEG)
        ok_count = 0
        for idx, act in enumerate(actions):
            clip_name = (f"R{round_id}_f{act['fighter_id']}"
                         f"_{idx+1:03d}_{act['action']}"
                         f"_f{act['window_start']:04d}_f{act['window_end']:04d}.mp4")
            clip_path = os.path.join(CLIPS_DIR, clip_name)

            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
                ok_count += 1
            else:
                if use_ffmpeg:
                    ok = extract_clip_ffmpeg(video_path, act["window_start"],
                                             act["window_end"], fps, clip_path)
                else:
                    ok = extract_clip_opencv(video_path, act["window_start"],
                                             act["window_end"], fps, clip_path)
                if ok:
                    ok_count += 1

            clips_meta.append({
                "clip":         clip_name,
                "round":        round_id,
                "fighter_id":   act["fighter_id"],
                "action":       act["action"],
                "window_start": act["window_start"],
                "window_end":   act["window_end"],
                "label":        None,   # to be filled by annotation tool
            })

        print(f"  -> Extracted {ok_count}/{len(actions)} clips")
    else:
        # Just build metadata without extracting
        for idx, act in enumerate(actions):
            clip_name = (f"R{round_id}_f{act['fighter_id']}"
                         f"_{idx+1:03d}_{act['action']}"
                         f"_f{act['window_start']:04d}_f{act['window_end']:04d}.mp4")
            clips_meta.append({
                "clip":         clip_name,
                "round":        round_id,
                "fighter_id":   act["fighter_id"],
                "action":       act["action"],
                "window_start": act["window_start"],
                "window_end":   act["window_end"],
                "label":        None,
            })

    return clips_meta, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=ALL_ROUNDS,
                    help="which rounds to process")
    ap.add_argument("--no-clips", action="store_true",
                    help="skip video clip extraction (just consolidate files)")
    args = ap.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    all_clips = []
    for rnd in args.rounds:
        clips_meta, _ = process_round(rnd, extract_clips=not args.no_clips)
        all_clips.extend(clips_meta)

    # ── Save manifest ─────────────────────────────────────────────────────────
    # Reload existing manifest to preserve any labels already assigned
    existing = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            ex = json.load(open(MANIFEST_PATH))
            existing = {e["clip"]: e.get("label") for e in ex.get("clips", [])}
        except Exception:
            pass

    for c in all_clips:
        if c["clip"] in existing and existing[c["clip"]] is not None:
            c["label"] = existing[c["clip"]]

    labeled   = sum(1 for c in all_clips if c["label"] is not None)
    unlabeled = len(all_clips) - labeled

    manifest = {
        "fight":       "lillyella_vs_zoe",
        "rounds":      args.rounds,
        "clips_dir":   CLIPS_DIR,
        "total_clips": len(all_clips),
        "labeled":     labeled,
        "unlabeled":   unlabeled,
        "clips":       all_clips,
    }
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total clips : {len(all_clips)}")
    print(f"  Labeled     : {labeled}")
    print(f"  Unlabeled   : {unlabeled}")
    print(f"  Manifest    : {MANIFEST_PATH}")
    print(f"  Clips dir   : {CLIPS_DIR}")
    print(f"\n  Next step: run the annotation tool")
    print(f"    python tools/annotate_clips.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
