#!/usr/bin/env python3
"""
Extract patch + clip datasets from the annotation-tool ground truth.

For every annotation in <gt.json>:
  - patch: 256x256 crop centered on the click point at the annotated frame
           (training scripts random-crop/resize down to 128)
  - clip:  +/-10 frames (21 frames), 320x320 crop centered on the click,
           resized to 128x128 per frame

Labels: LANDED=1, MISS=0.

Usage:
  python tools/extract_gt_dataset.py --gt <gt.json> --video <video.mp4> --out <out.npz>
"""
import json, argparse
import numpy as np
import cv2

PATCH_SRC = 256     # patch crop size at full res
CLIP_SRC  = 320     # clip crop size at full res
CLIP_OUT  = 128     # clip per-frame output size
T_HALF    = 10      # frames either side -> 21 total


def crop_center(img, cx, cy, size):
    """Crop size x size centered on (cx,cy), clamped inside the image."""
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, min(w - size, cx - half))
    y1 = max(0, min(h - size, cy - half))
    return img[y1:y1 + size, x1:x1 + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    anns = [a for a in gt["annotations"] if a["verdict"] in ("LANDED", "MISS")]
    print(f"{len(anns)} annotations (LANDED/MISS)")

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {n_frames} frames "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    patches, clips, labels, frames_l, clicks, parts = [], [], [], [], [], []
    for i, a in enumerate(anns):
        f, cx, cy = a["frame"], a["pixel_x"], a["pixel_y"]
        lo, hi = f - T_HALF, f + T_HALF
        if lo < 0 or hi >= n_frames:
            print(f"  skip frame {f} (clip window out of bounds)")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        clip = []
        ok = True
        for t in range(lo, hi + 1):
            ret, bgr = cap.read()
            if not ret:
                ok = False
                break
            c = crop_center(bgr, cx, cy, CLIP_SRC)
            c = cv2.resize(c, (CLIP_OUT, CLIP_OUT), interpolation=cv2.INTER_AREA)
            clip.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            if t == f:
                patches.append(cv2.cvtColor(
                    crop_center(bgr, cx, cy, PATCH_SRC), cv2.COLOR_BGR2RGB))
        if not ok or len(clip) != 2 * T_HALF + 1:
            print(f"  skip frame {f} (read failure)")
            if len(patches) > len(clips):
                patches.pop()
            continue

        clips.append(np.stack(clip))
        labels.append(1 if a["verdict"] == "LANDED" else 0)
        frames_l.append(f)
        clicks.append((cx, cy))
        parts.append(a["body_part"])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(anns)}")

    cap.release()
    patches = np.stack(patches)          # (N,256,256,3) uint8
    clips   = np.stack(clips)            # (N,21,128,128,3) uint8
    labels  = np.array(labels, np.int64)
    print(f"patches {patches.shape}  clips {clips.shape}  "
          f"pos {labels.sum()} / neg {(labels==0).sum()}")

    np.savez_compressed(
        args.out,
        patches=patches, clips=clips, labels=labels,
        frames=np.array(frames_l), clicks=np.array(clicks),
        body_parts=np.array(parts), video=args.video)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
