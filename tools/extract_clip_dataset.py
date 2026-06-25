#!/usr/bin/env python3
"""
Build a temporal clip dataset (for tools/train_clip_model.py) from the
manifest.json labels produced by tools/annotate_clips.py, across multiple
fights at once.

Unlike tools/extract_gt_dataset.py (which reads precise click-point
annotations: frame + pixel_x/pixel_y + verdict from a different, older
annotation tool), our manifest.json labels are at the ACTION level
(window_start/window_end + impact/not_impact, no click point). So the
"contact frame" and crop center here are estimated the same way the
XGBoost feature extractor does it: the frame within the action window
(+pad) where the striker's wrist comes closest to the receiver's body
(detectors/fusion/v9.min_wrist_body_gap), and the crop is centered on the
union of both fighters' pixel bboxes at that frame (adaptive size, since
camera zoom/distance varies across matches -- a fixed 320px window from
one match's scale doesn't transfer to another).

Output: one combined .npz across all requested fights with an extra
`groups` array (fight name per clip) so train_clip_model.py can do
leave-one-match-out CV.

Usage:
    python tools/extract_clip_dataset.py --fights lillyella_vs_zoe cameron_vs_liam jamie_vs_ryan \
        --out outputs/gt_dataset/combined_clip_gt.npz
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors.fusion import v9
from dataset import fights

CLIP_OUT = 128   # per-frame output size (matches train_clip_model.py)
T_HALF = 10      # frames either side -> 21 total
MIN_CROP = 200   # px, floor so close-up frames don't get a tiny/noisy crop
PAD_FRAC = 1.4   # crop = union-bbox extent * PAD_FRAC


def crop_square(img, cx, cy, size):
    h, w = img.shape[:2]
    size = int(min(size, h, w))
    half = size // 2
    x1 = max(0, min(w - size, int(cx - half)))
    y1 = max(0, min(h - size, int(cy - half)))
    return img[y1:y1 + size, x1:x1 + size]


def find_contact_frame_and_box(clip, persons):
    """Reuses v9's wrist-body-gap scan to find the contact frame, then
    returns the union bbox of both fighters at that frame for cropping."""
    sid = clip["fighter_id"]
    rid = 1 - sid
    sp, rp = persons.get(sid, {}), persons.get(rid, {})
    ws, we = clip["window_start"] - 3, clip["window_end"] + 5

    best_f, best_gap = clip["window_start"], 1e9
    for f in range(ws, we + 1):
        se, re = sp.get(f), rp.get(f)
        g, _ = v9.min_wrist_body_gap(v9.wc(se) if se else None, v9.wc(re) if re else None)
        if g is not None and g < best_gap:
            best_gap, best_f = g, f
    if best_gap >= 1e9:
        return None

    se, re = sp.get(best_f), rp.get(best_f)
    boxes = [e["bbox"] for e in (se, re) if e]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(MIN_CROP, max(x2 - x1, y2 - y1) * PAD_FRAC)
    return best_f, cx, cy, size


def extract_fight(fight_name, clips_out, labels_out, frames_out, groups_out):
    cfg = fights.get_fight(fight_name)
    manifest = json.load(open(cfg["manifest_path"]))
    labeled = [c for c in manifest["clips"] if c["label"] is not None]
    by_round = {}
    for c in labeled:
        by_round.setdefault(c["round"], []).append(c)

    n_ok, n_skip = 0, 0
    for round_id, clip_list in sorted(by_round.items()):
        sam3d_path = os.path.join(cfg["out_base"], f"Round{round_id}", "sam3d.json")
        video_path = fights.get_video_path(cfg, round_id)
        if not os.path.exists(sam3d_path) or not os.path.exists(video_path):
            print(f"  [{fight_name} Round{round_id}] missing sam3d/video -> skip {len(clip_list)} clips")
            n_skip += len(clip_list)
            continue
        s = json.load(open(sam3d_path))
        persons = {0: {e["frame"]: e for e in s.get("0", [])},
                   1: {e["frame"]: e for e in s.get("1", [])}}
        if not (persons[0] and "world_coords" in next(iter(persons[0].values()), {}) and
                persons[1] and "world_coords" in next(iter(persons[1].values()), {})):
            print(f"  [{fight_name} Round{round_id}] missing SAM3D keypoints -> skip {len(clip_list)} clips")
            n_skip += len(clip_list)
            continue

        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for c in clip_list:
            result = find_contact_frame_and_box(c, persons)
            if result is None:
                n_skip += 1
                continue
            best_f, cx, cy, size = result
            lo, hi = best_f - T_HALF, best_f + T_HALF
            if lo < 0 or hi >= n_frames:
                n_skip += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            frames_buf = []
            ok = True
            for t in range(lo, hi + 1):
                ret, bgr = cap.read()
                if not ret:
                    ok = False
                    break
                crop = crop_square(bgr, cx, cy, size)
                crop = cv2.resize(crop, (CLIP_OUT, CLIP_OUT), interpolation=cv2.INTER_AREA)
                frames_buf.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not ok or len(frames_buf) != 2 * T_HALF + 1:
                n_skip += 1
                continue

            clips_out.append(np.stack(frames_buf))
            labels_out.append(1 if c["label"] == "impact" else 0)
            frames_out.append(best_f)
            groups_out.append(fight_name)
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  [{fight_name}] {n_ok} clips extracted...")
        cap.release()

    print(f"[{fight_name}] done: {n_ok} usable, {n_skip} skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", nargs="+",
                     default=["lillyella_vs_zoe", "cameron_vs_liam", "jamie_vs_ryan"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clips, labels, frames, groups = [], [], [], []
    for fight_name in args.fights:
        extract_fight(fight_name, clips, labels, frames, groups)

    clips = np.stack(clips)
    labels = np.array(labels, np.int64)
    frames = np.array(frames)
    groups = np.array(groups)
    print(f"\nTOTAL: {len(labels)} clips  pos={labels.sum()}  neg={(labels == 0).sum()}")
    for g in sorted(set(groups.tolist())):
        m = groups == g
        print(f"  {g:18s} n={m.sum():4d}  pos={labels[m].sum():4d}  neg={(labels[m] == 0).sum():4d}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, clips=clips, labels=labels, frames=frames, groups=groups)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
