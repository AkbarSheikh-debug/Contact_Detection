#!/usr/bin/env python3
"""
Extract VIDEO+AUDIO clips per ASFormer action window  (sound + action)
======================================================================
Divides the video using the ASFormer action frame ranges instead of raw audio
peaks.  Every clip = one *thrown punch* (window_start → window_end), cut WITH
sound, and tagged with the audio onset loudness inside that window.  So each
clip combines both signals:

    ACTION  : ASFormer says a punch was thrown here (which frames)
    SOUND   : how loud the impact onset was inside that window

You then judge the only thing left — did it LAND?  Sort into punch/ not_punch/.

Filename:
    NNN_<action>_f<start>-<end>_conf<c>_onset<o>.mp4
        NNN     auditioning order (by start frame)
        action  ASFormer label (jab_left, cross_right, …)
        f..-..  the action window in source frames
        conf    ASFormer confidence
        onset   max audio onset ratio inside the window (sound signal)

Output:
    outputs/action_clips/
        <clips>.mp4
        punch/      ← move clips where the punch LANDED
        not_punch/  ← move misses / blocks / no-contact
        index.json, README.txt

Usage:
    python extract_action_clips.py
    python extract_action_clips.py --pre 0.4 --post 0.7 --min-dur 1.4
    python extract_action_clips.py --merge-gap 6     # merge combo windows
"""
import os
import json
import argparse
import subprocess
import sys

import numpy as np

# make the sound module importable from its package location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detectors", "sound"))

DATA = r"/home/jake/Downloads/sam3d_with_world_coords"
VIDEO = os.path.join(DATA, "3.mp4")
WAV = os.path.join(DATA, "3.wav")
ACT_JSON = os.path.join(DATA, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
OUT_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs"
CLIP_DIR = os.path.join(OUT_DIR, "impact_dataset")
FPS = 24.995


def load_onset_envelope():
    """Reuse the sound detector's onset envelope (ratio above local noise)."""
    try:
        from sound_detector import load_wav, compute_onset
        x, sr = load_wav(WAV)
        times, onset, _, _ = compute_onset(x, sr)
        return times, onset
    except Exception as e:
        print(f"[act] onset envelope unavailable ({e}); onset tag = 0")
        return None, None


def max_onset_in_window(times, onset, ws, we, pad_sec=0.3):
    if times is None:
        return 0.0
    lo = ws / FPS - pad_sec
    hi = we / FPS + pad_sec
    m = (times >= lo) & (times <= hi)
    return float(onset[m].max()) if m.any() else 0.0


def merge_windows(actions, gap):
    """Optionally merge action windows whose ranges are within `gap` frames."""
    if gap <= 0:
        return actions
    acts = sorted(actions, key=lambda a: a["window_start"])
    merged = []
    for a in acts:
        if merged and a["window_start"] - merged[-1]["window_end"] <= gap:
            m = merged[-1]
            m["window_end"] = max(m["window_end"], a["window_end"])
            m["action"] = m["action"] + "+" + a["action"]
            m["confidence"] = max(m["confidence"], a["confidence"])
        else:
            merged.append(dict(a))
    return merged


def cut_clip(start_sec, dur, out_path):
    for vcodec, extra in (("mpeg4", ["-q:v", "4"]), ("libx264", [])):
        cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", VIDEO,
               "-t", f"{dur:.3f}", "-c:v", vcodec, *extra, "-c:a", "aac",
               "-pix_fmt", "yuv420p", out_path, "-loglevel", "error"]
        if subprocess.run(cmd).returncode == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=float, default=0.4,
                    help="seconds before window_start")
    ap.add_argument("--post", type=float, default=0.7,
                    help="seconds after window_end (contact often lands here)")
    ap.add_argument("--min-dur", type=float, default=1.4,
                    help="minimum clip duration (short windows get expanded)")
    ap.add_argument("--merge-gap", type=int, default=0,
                    help="merge action windows within this many frames (0=off)")
    ap.add_argument("--clip-dir", default=CLIP_DIR)
    args = ap.parse_args()

    actions = json.load(open(ACT_JSON))["actions"]
    actions = merge_windows(actions, args.merge_gap)
    actions.sort(key=lambda a: a["window_start"])
    print(f"[act] {len(actions)} action windows "
          f"({'merged' if args.merge_gap else 'unmerged'})")

    times, onset = load_onset_envelope()

    os.makedirs(args.clip_dir, exist_ok=True)
    os.makedirs(os.path.join(args.clip_dir, "impact"), exist_ok=True)
    os.makedirs(os.path.join(args.clip_dir, "not_impact"), exist_ok=True)

    index = []
    print(f"[act] cutting clips (window +pre{args.pre}/post{args.post}s, "
          f"min {args.min_dur}s, video+audio) …")
    for i, a in enumerate(actions):
        ws, we = a["window_start"], a["window_end"]
        fighter = 0 if a.get("fighter_type") == "fighter_0" else 1
        act = a["action"]
        conf = float(a.get("confidence", 0.0))
        onset_v = max_onset_in_window(times, onset, ws, we)

        start = max(0.0, ws / FPS - args.pre)
        end = we / FPS + args.post
        if end - start < args.min_dur:                # expand short windows
            c = (start + end) / 2
            start = max(0.0, c - args.min_dur / 2)
            end = start + args.min_dur
        dur = end - start

        name = (f"{i+1:03d}_{act}_f{ws}-{we}_conf{conf:.2f}"
                f"_onset{onset_v:.1f}.mp4")
        ok = cut_clip(start, dur, os.path.join(args.clip_dir, name))
        if ok:
            index.append({"id": i + 1, "file": name, "action": act,
                          "fighter": fighter, "window_start": ws,
                          "window_end": we, "confidence": round(conf, 3),
                          "onset": round(onset_v, 3),
                          "clip_start_sec": round(start, 3),
                          "clip_dur_sec": round(dur, 3)})
        else:
            print(f"  [warn] failed clip {i+1} ({act} f{ws}-{we})")

    json.dump({"n": len(index), "pre": args.pre, "post": args.post,
               "min_dur": args.min_dur, "merge_gap": args.merge_gap,
               "clips": index},
              open(os.path.join(args.clip_dir, "index.json"), "w"), indent=2)

    readme = (
        "IMPACT-DETECTION DATASET — one clip per ASFormer action (punch OR kick)\n"
        "=======================================================================\n\n"
        f"{len(index)} clips, one per ASFormer action window, video+audio.\n"
        "Action TYPE does not matter (punch/kick/jab/hook all fine) — only\n"
        "whether the strike CONNECTED.\n\n"
        "Filename: NNN_<action>_f<start>-<end>_conf<asformer>_onset<sound>.mp4\n"
        "The clip is padded a little past the action window so the moment of\n"
        "CONTACT and the receiver's REACTION are captured (that's where the\n"
        "'did it land' evidence is, and what a temporal model learns from).\n\n"
        "SORT EACH CLIP:\n"
        "  - strike LANDED on head/body            -> ./impact/\n"
        "  - missed / blocked on guard / no contact -> ./not_impact/\n"
        "  - leave ambiguous ones where they are\n\n"
        "When you have clips from several matches sorted this way, they become\n"
        "the labeled dataset to train the temporal impact model.\n"
        "The 'onset' number = impact-sound loudness (a weak hint, not the answer).\n"
    )
    open(os.path.join(args.clip_dir, "README.txt"), "w").write(readme)

    print(f"\n[act] {len(index)} clips → {args.clip_dir}")
    print(f"      sort into  impact/  and  not_impact/  (did the strike connect?)")
    # quick preview sorted by onset loudness
    top = sorted(index, key=lambda c: -c["onset"])[:10]
    print(f"\n  loudest-onset thrown punches (most likely landings):")
    for c in top:
        print(f"    {c['file']}")


if __name__ == "__main__":
    main()
