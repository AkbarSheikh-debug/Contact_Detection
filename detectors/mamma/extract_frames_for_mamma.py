#!/usr/bin/env python3
"""
Step 0 helper — extract the frames MammaNet needs from a fight video.
=====================================================================
MammaNet (MAMMA's per-image 2D stage) runs on individual frames and makes its own
person masks (SAM+YOLO).  We don't need to feed it all ~4500 frames — only the frames
inside each ASFormer action window (+padding), which is where impacts can happen.
This script dumps exactly those frames as zero-padded JPGs plus an index JSON, ready
to hand to MammaNet on a GPU box.

Runnable now (CPU, just video decode).  The MammaNet run itself happens afterwards on a
GPU machine — see detectors/mamma/SETUP.md.

Usage:
    python detectors/mamma/extract_frames_for_mamma.py --folder "<fight folder>"
    python detectors/mamma/extract_frames_for_mamma.py --folder "<dir>" --pre 0.3 --post 0.7
    python detectors/mamma/extract_frames_for_mamma.py --folder "<dir>" --all   # every frame
"""
import os, json, glob, argparse


def find_files(folder):
    fa = glob.glob(os.path.join(folder, "*_full_analysis.json"))
    vids = [v for v in glob.glob(os.path.join(folder, "*.mp4"))
            if "_visualized" not in os.path.basename(v)
            and "_impacts_v9" not in os.path.basename(v)]
    if not fa or not vids:
        raise SystemExit(f"[mamma] need *_full_analysis.json + raw video in {folder}")
    return fa[0], vids[0]


def wanted_frames(fa_path, pre, post, fps, total, grab_all):
    if grab_all:
        return set(range(total))
    fa = json.load(open(fa_path))
    pre_f = int(round(pre * fps)); post_f = int(round(post * fps))
    frames = set()
    for a in fa["actions"]:
        lo = max(0, a["window_start"] - pre_f)
        hi = min(total - 1, a["window_end"] + post_f)
        frames.update(range(lo, hi + 1))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pre", type=float, default=0.3, help="seconds before window_start")
    ap.add_argument("--post", type=float, default=0.7, help="seconds after window_end")
    ap.add_argument("--all", action="store_true", help="extract every frame")
    ap.add_argument("--out", default=None, help="output dir (default <folder>/mamma_frames)")
    args = ap.parse_args()

    import cv2
    fa_path, vid = find_files(args.folder)
    out_dir = args.out or os.path.join(args.folder, "mamma_frames")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise SystemExit(f"[mamma] cannot open {vid}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = wanted_frames(fa_path, args.pre, args.post, fps, total, args.all)
    print(f"[mamma] video={os.path.basename(vid)}  total={total}  fps={fps:.3f}")
    print(f"[mamma] extracting {len(want)} frames -> {out_dir}")

    saved = []
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi in want:
            name = f"frame_{fi:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append({"frame": fi, "file": name})
    cap.release()

    index = {
        "source_video": os.path.basename(vid),
        "fps": round(fps, 3),
        "total_frames": total,
        "n_extracted": len(saved),
        "selection": "all" if args.all else f"asformer_windows +/- ({args.pre}s,{args.post}s)",
        "frames": saved,
    }
    json.dump(index, open(os.path.join(out_dir, "_index.json"), "w"), indent=2)
    print(f"[mamma] done. index -> {os.path.join(out_dir, '_index.json')}")
    print(f"[mamma] next: run MammaNet on these frames (see detectors/mamma/SETUP.md)")


if __name__ == "__main__":
    main()
