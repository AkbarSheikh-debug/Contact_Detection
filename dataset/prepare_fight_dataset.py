#!/usr/bin/env python3
"""
Prepare any registered fight (see dataset/fights.py) for impact detection
annotation. Generalized version of prepare_lillyella_dataset.py.

Does everything in one pass:
  1. Reads Gladius action labels (fighter0 + fighter1) and converts
     local part-frame numbers -> global video frame numbers.
  2. Merges per-fighter bbox tracking data into a v8.py/annotate_clips.py
     compatible {"0": [...], "1": [...]} format. Two source formats are
     supported via fights.py's tracking_format:
       - "sam3d_keypoints" (default): RoundN_SAM3D_fighterX.json's
         keypoints_3d, in video_folder.
       - "raw_bbox": fighters_tracking_data from a tracking_files entry,
         in tracking_folder (used for fights without a SAM3D export).
  3. Copies the CUTIE/tracking JSON for reference.
  4. Extracts one video clip per action window (with configurable padding).
  5. Writes a manifest.json for the annotation tool.

Usage:
    python dataset/prepare_fight_dataset.py --fight cameron_vs_liam
    python dataset/prepare_fight_dataset.py --fight 1st_fight --rounds 1 2
    python dataset/prepare_fight_dataset.py --fight jamie_vs_ryan --no-clips
    python dataset/prepare_fight_dataset.py --all
"""

import os
import json
import argparse
import subprocess
import shutil

import fights

# Padding around each action window when cutting clips
PAD_BEFORE = 8   # frames before window_start
PAD_END    = 20  # frames after window_end  (captures post-impact reaction)


# ── Helpers ───────────────────────────────────────────────────────────────────

def gladius_labels_to_global(cfg, round_id, fighter_id):
    """Load Gladius action labels for one round/fighter and return a list of
    dicts with global frame numbers (1-indexed, matching video timeline)."""
    fid = f"fighter{fighter_id}"
    path = os.path.join(cfg["gladius_folder"], fid, f"Round{round_id}.json")
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
        # NOTE: lb["start"]/lb["end"] are already GLOBAL frame numbers for the
        # whole round -- do NOT add frame_offset here.
        actions.append({
            "fighter_id":    fighter_id,
            "fighter_type":  f"fighter_{fighter_id}",
            "action":        lb["label"],
            "window_start":  lb["start"],
            "window_end":    lb["end"],
            "part":          part_name,
            "confidence":    1.0,
        })
    return actions


def video_filename_for_round(cfg, round_id):
    custom = cfg.get("video_filenames")
    if custom and round_id in custom:
        return custom[round_id]
    return f"Round{round_id}.mp4"


def tracking_json_source_path(cfg, round_id):
    """Path to the raw tracking/CUTIE json for this round, regardless of
    tracking_format -- used for the reference tracking.json copy."""
    if cfg.get("tracking_format") == "raw_bbox":
        files = cfg.get("tracking_files", {})
        if round_id not in files:
            return None
        return os.path.join(cfg["tracking_folder"], files[round_id])
    return os.path.join(cfg["video_folder"], f"Round{round_id}.json")


def merge_tracking(cfg, round_id):
    """Merge per-fighter tracking data into v8.py-compatible
    {0: [{"frame":.., "bbox":..}, ...], 1: [...]} format.
    Returns merged dict or None if no source data is available."""
    fmt = cfg.get("tracking_format", "sam3d_keypoints")
    merged = {}

    if fmt == "raw_bbox":
        path = tracking_json_source_path(cfg, round_id)
        if not path or not os.path.exists(path):
            print(f"  [WARN] tracking json not found: {path}")
            return None
        d = json.load(open(path))
        ftd = d.get("fighters_tracking_data", {})
        for fid in (0, 1):
            raw = ftd.get(str(fid), [])
            merged[fid] = [{"frame": e["frame"], "bbox": e["bbox"]} for e in raw]
            print(f"    tracking fighter{fid}: {len(raw)} frame entries")
    else:
        # CUTIE tracking.json (the plain RoundN.json, not _SAM3D_fighterN) has
        # bbox-by-frame for BOTH fighters regardless of whether either
        # fighter's separate SAM3D keypoints export exists -- used as a
        # fallback so a fighter with no SAM3D file still gets a name-tag
        # overlay/swap-detection anchor instead of silently having none.
        cutie_path = os.path.join(cfg["video_folder"], f"Round{round_id}.json")
        cutie_ftd = None
        for fid in (0, 1):
            p = os.path.join(cfg["video_folder"], f"Round{round_id}_SAM3D_fighter{fid}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                kp = d.get("keypoints_3d", {})
                raw = kp.get("0") or kp.get(str(fid)) or next(iter(kp.values()), [])
                merged[fid] = raw
                print(f"    SAM3D fighter{fid}: {len(raw)} frame entries")
                continue
            print(f"  [WARN] SAM3D not found: {p}")
            if cutie_ftd is None:
                cutie_ftd = (json.load(open(cutie_path)).get("fighters_tracking_data", {})
                             if os.path.exists(cutie_path) else {})
            raw = cutie_ftd.get(str(fid), [])
            if raw:
                merged[fid] = [{"frame": e["frame"], "bbox": e["bbox"]} for e in raw]
                print(f"    fallback CUTIE bbox fighter{fid}: {len(raw)} frame entries")

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
    t_start = (s_frame - 1) / fps
    t_end   = (e_frame)     / fps
    t_start = max(0.0, t_start)

    cmd = [
        fights.FFMPEG, "-y",
        "-ss", f"{t_start:.6f}",
        "-to", f"{t_end:.6f}",
        "-i", video_path,
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-an",
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

def process_round(cfg, round_id, extract_clips=True):
    print(f"\n{'='*60}")
    print(f"  {cfg['name']} -- Round {round_id}")
    print(f"{'='*60}")

    round_dir = os.path.join(cfg["out_base"], f"Round{round_id}")
    os.makedirs(round_dir, exist_ok=True)

    # ── 1. Merge action labels (fighter0 + fighter1) ──────────────────────────
    actions = []
    for fid in (0, 1):
        fa = gladius_labels_to_global(cfg, round_id, fid)
        actions.extend(fa)
        print(f"  fighter{fid}: {len(fa)} action windows")

    actions.sort(key=lambda a: a["window_start"])
    actions_path = os.path.join(round_dir, "actions.json")
    json.dump({"fight": cfg["name"], "round": round_id,
               "actions": actions}, open(actions_path, "w"), indent=2)
    print(f"  -> {len(actions)} total actions saved to {actions_path}")

    # ── 2. Merge tracking (bbox-by-frame) ─────────────────────────────────────
    tracking = merge_tracking(cfg, round_id)
    if tracking:
        sam3d_path = os.path.join(round_dir, "sam3d.json")
        json.dump({"0": tracking.get(0, []),
                   "1": tracking.get(1, []),
                   "contact_events": []},
                  open(sam3d_path, "w"), indent=2)
        print(f"  -> tracking merged: {sam3d_path}")
    else:
        print(f"  [WARN] No tracking data available for Round{round_id}")

    # ── 3. Copy reference tracking JSON ───────────────────────────────────────
    cutie_src = tracking_json_source_path(cfg, round_id)
    if cutie_src and os.path.exists(cutie_src):
        cutie_dst = os.path.join(round_dir, "tracking.json")
        shutil.copy2(cutie_src, cutie_dst)
        print(f"  -> Tracking JSON copied")

    # ── 4. Extract video clips ────────────────────────────────────────────────
    video_path = os.path.join(cfg["video_folder"], video_filename_for_round(cfg, round_id))
    if not os.path.exists(video_path):
        print(f"  [WARN] Video not found: {video_path} -- skipping clip extraction")
        return [], actions

    fps, total_frames = get_video_fps(video_path)
    print(f"  Video: {total_frames} frames @ {fps:.3f} fps")

    clips_meta = []
    if extract_clips:
        os.makedirs(cfg["clips_dir"], exist_ok=True)
        use_ffmpeg = os.path.exists(fights.FFMPEG)
        ok_count = 0
        for idx, act in enumerate(actions):
            clip_name = (f"R{round_id}_f{act['fighter_id']}"
                         f"_{idx+1:03d}_{act['action']}"
                         f"_f{act['window_start']:04d}_f{act['window_end']:04d}.mp4")
            clip_path = os.path.join(cfg["clips_dir"], clip_name)

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
                "label":        None,
                "impact_frame": None,
            })

        print(f"  -> Extracted {ok_count}/{len(actions)} clips")
    else:
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
                "impact_frame": None,
            })

    return clips_meta, actions


def process_fight(fight_name, rounds=None, extract_clips=True):
    cfg = fights.get_fight(fight_name)
    rounds = rounds or cfg["rounds"]

    os.makedirs(cfg["out_base"], exist_ok=True)
    os.makedirs(cfg["clips_dir"], exist_ok=True)

    all_clips = []
    for rnd in rounds:
        clips_meta, _ = process_round(cfg, rnd, extract_clips=extract_clips)
        all_clips.extend(clips_meta)

    # Reload existing manifest to preserve any labels (and marked impact
    # frames) already assigned
    existing = {}
    if os.path.exists(cfg["manifest_path"]):
        try:
            ex = json.load(open(cfg["manifest_path"]))
            existing = {e["clip"]: (e.get("label"), e.get("impact_frame"))
                       for e in ex.get("clips", [])}
        except Exception:
            pass

    for c in all_clips:
        if c["clip"] in existing:
            old_label, old_impact_frame = existing[c["clip"]]
            if old_label is not None:
                c["label"] = old_label
            if old_impact_frame is not None:
                c["impact_frame"] = old_impact_frame

    labeled   = sum(1 for c in all_clips if c["label"] is not None)
    unlabeled = len(all_clips) - labeled

    manifest = {
        "fight":       fight_name,
        "rounds":      rounds,
        "clips_dir":   cfg["clips_dir"],
        "total_clips": len(all_clips),
        "labeled":     labeled,
        "unlabeled":   unlabeled,
        "clips":       all_clips,
    }
    json.dump(manifest, open(cfg["manifest_path"], "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE -- {fight_name}")
    print(f"  Total clips : {len(all_clips)}")
    print(f"  Labeled     : {labeled}")
    print(f"  Unlabeled   : {unlabeled}")
    print(f"  Manifest    : {cfg['manifest_path']}")
    print(f"  Clips dir   : {cfg['clips_dir']}")
    print(f"{'='*60}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight", type=str, default=None,
                    choices=fights.all_fight_names(),
                    help="which registered fight to process")
    ap.add_argument("--all", action="store_true",
                    help="process every fight in the registry")
    ap.add_argument("--rounds", type=int, nargs="+", default=None,
                    help="which rounds to process (default: fight's registered rounds)")
    ap.add_argument("--no-clips", action="store_true",
                    help="skip video clip extraction (just consolidate files)")
    args = ap.parse_args()

    if not args.fight and not args.all:
        ap.error("specify --fight <name> or --all")

    fight_names = fights.all_fight_names() if args.all else [args.fight]
    for name in fight_names:
        process_fight(name, rounds=args.rounds, extract_clips=not args.no_clips)
        print(f"\n  Next step: python tools/annotate_clips.py --fight {name}\n")


if __name__ == "__main__":
    main()
