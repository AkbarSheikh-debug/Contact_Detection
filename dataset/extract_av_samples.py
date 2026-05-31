#!/usr/bin/env python3
"""
Extract VIDEO+AUDIO candidate clips for punch auditioning
=========================================================
Instead of audio-only clips (hard to judge), this cuts a short mp4 WITH sound
around every onset candidate, so you can SEE the punch land and HEAR it, then
sort into punch/ vs not_punch/ much more reliably.

Reuses the same candidates as the audio extraction (outputs/sound_samples/
index.json) so the clips map 1:1 to the audio clips.  If that index is missing,
it recomputes onsets from scratch.

Output:
    outputs/sound_samples_av/
        001_t..._score....mp4   (each ~1.4s, video+audio, centred on the onset)
        punch/      ← move clear punch-landing clips here
        not_punch/  ← move clear non-punch clips here
        README.txt

Usage:
    python extract_av_samples.py
    python extract_av_samples.py --pad 1.0          # longer clips
    python extract_av_samples.py --onset-thr 1.3    # more candidates (if no index)
"""
import os
import json
import argparse
import subprocess
import sys

# make the sound module importable from its package location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detectors", "sound"))

VIDEO = r"/home/jake/Downloads/sam3d_with_world_coords/3.mp4"
WAV = r"/home/jake/Downloads/sam3d_with_world_coords/3.wav"
OUT_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs"
AUDIO_SAMPLE_DIR = os.path.join(OUT_DIR, "sound_samples")
AV_DIR = os.path.join(OUT_DIR, "sound_samples_av")
FPS = 24.995


def candidates_from_index(idx_path):
    data = json.load(open(idx_path))
    return [(s["id"], s["time_sec"], s["timestamp"], s["onset_score"], s["frame"])
            for s in data["samples"]]


def candidates_from_scratch(onset_thr, cooldown):
    """Fallback: recompute onset peaks if no audio index exists."""
    from sound_detector import load_wav, compute_onset, detect_impacts
    x, sr = load_wav(WAV)
    times, onset, _, _ = compute_onset(x, sr)
    imp = detect_impacts(times, onset, onset_thr, cooldown)
    imp.sort(key=lambda z: z[0])
    out = []
    for i, (t, sc) in enumerate(imp):
        mm = int(t // 60); ss = t - mm * 60
        out.append((i + 1, t, f"{mm}:{ss:05.2f}", sc, int(round(t * FPS))))
    return out


def cut_clip(t_center, pad, out_path):
    """ffmpeg-cut a video+audio clip centred at t_center (re-encoded so the
    start is frame-accurate and the clip is self-contained)."""
    start = max(0.0, t_center - pad)
    dur = pad * 2
    # libx264 isn't in this ffmpeg build; mpeg4 is universally available.
    for vcodec, extra in (("mpeg4", ["-q:v", "4"]), ("libx264", [])):
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", VIDEO,
               "-t", f"{dur:.3f}", "-c:v", vcodec, *extra, "-c:a", "aac",
               "-pix_fmt", "yuv420p", out_path, "-loglevel", "error"]
        if subprocess.run(cmd).returncode == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=0.7,
                    help="seconds before/after the onset (clip = 2*pad long)")
    ap.add_argument("--onset-thr", type=float, default=1.5)
    ap.add_argument("--cooldown", type=int, default=12)
    ap.add_argument("--av-dir", default=AV_DIR)
    args = ap.parse_args()

    idx_path = os.path.join(AUDIO_SAMPLE_DIR, "index.json")
    if os.path.exists(idx_path):
        cands = candidates_from_index(idx_path)
        print(f"[av] reusing {len(cands)} candidates from {idx_path}")
    else:
        cands = candidates_from_scratch(args.onset_thr, args.cooldown)
        print(f"[av] computed {len(cands)} candidates from audio onsets")

    os.makedirs(args.av_dir, exist_ok=True)
    os.makedirs(os.path.join(args.av_dir, "punch"), exist_ok=True)
    os.makedirs(os.path.join(args.av_dir, "not_punch"), exist_ok=True)

    index = []
    print(f"[av] cutting {len(cands)} clips (±{args.pad}s, video+audio) …")
    for cid, t, ts, score, frame in cands:
        name = f"{cid:03d}_t{ts.replace(':','m',1)}s_score{score:.2f}_f{frame}.mp4"
        name = name.replace(":", "")
        out_path = os.path.join(args.av_dir, name)
        ok = cut_clip(t, args.pad, out_path)
        if ok:
            index.append({"id": cid, "file": name, "time_sec": round(t, 3),
                          "timestamp": ts, "frame": frame,
                          "onset_score": round(score, 3)})
        else:
            print(f"  [warn] failed to cut clip {cid} @ {ts}")

    json.dump({"pad_sec": args.pad, "n": len(index), "clips": index},
              open(os.path.join(args.av_dir, "index.json"), "w"), indent=2)

    readme = (
        "VIDEO+AUDIO PUNCH CLIPS — re-sort with full context\n"
        "===================================================\n\n"
        f"{len(index)} clips, each ~{2*args.pad:.1f}s with sound, centred on an\n"
        "onset candidate.  Filename = NNN_t<min>m<sec>s_score<onset>_f<frame>.mp4\n\n"
        "WORKFLOW:\n"
        "  1. Play each clip — you can now SEE the punch land and HEAR it.\n"
        "  2. Move clear PUNCH-LANDING clips     -> ./punch/\n"
        "  3. Move clear NON-PUNCH clips         -> ./not_punch/\n"
        "  4. Leave ambiguous ones where they are.\n\n"
        "When done, tell me — I'll retrain the sound detector on your\n"
        "video-verified labels (more reliable than the audio-only sort).\n"
    )
    open(os.path.join(args.av_dir, "README.txt"), "w").write(readme)

    print(f"\n[av] {len(index)} clips written to {args.av_dir}")
    print(f"     sort them into  {args.av_dir}/punch/  and  /not_punch/")
    print(f"     (each clip has video + sound; filename carries timestamp+score)")


if __name__ == "__main__":
    main()
