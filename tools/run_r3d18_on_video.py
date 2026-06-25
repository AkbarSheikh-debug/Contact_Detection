#!/usr/bin/env python3
"""
Score a fight folder's ASFormer action candidates with the trained r3d_18
video-CNN checkpoint (tools/train_clip_model.py --train-final), then
dedup/NMS and render an annotated video -- same output shape as v9/XGBoost/
keypoint models, so all approaches can be compared on the same video.

Builds the same 21-frame, contact-frame-centered, union-bbox-cropped clip
as tools/extract_clip_dataset.py, driven from this video's ASFormer action
candidates instead of manifest.json labels, then applies the exact eval-time
preprocessing tools/train_clip_model.py's ClipDS uses (center 16 of 21
frames, center-crop 128->112, Kinetics mean/std normalize).

Usage:
    python tools/run_r3d18_on_video.py --folder "C:/Users/XRIG/Downloads/1st_Impact_detection_Fixed_05062026"
    python tools/run_r3d18_on_video.py --folder "<dir>" --video
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_clip_dataset import find_contact_frame_and_box, crop_square, T_HALF, CLIP_OUT  # noqa: E402
from detectors.fusion import v9  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_FRAMES, SIZE = 16, 112
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "outputs", "clip_model_FINAL_alldata_v2.pt")


def make_model(mode="finetune"):
    m = torchvision.models.video.r3d_18(weights=None)
    m.fc = torch.nn.Linear(512, 1)
    return m.to(DEV)


def load_model(ckpt_path):
    m = make_model()
    m.load_state_dict(torch.load(ckpt_path, map_location=DEV))
    m.to(DEV).eval()
    print(f"[r3d18] loaded <- {ckpt_path}")
    return m


def build_clip(cap, n_frames, best_f, cx, cy, size):
    lo, hi = best_f - T_HALF, best_f + T_HALF
    if lo < 0 or hi >= n_frames:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    frames = []
    for t in range(lo, hi + 1):
        ret, bgr = cap.read()
        if not ret:
            return None
        crop = crop_square(bgr, cx, cy, size)
        crop = cv2.resize(crop, (CLIP_OUT, CLIP_OUT), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return np.stack(frames)  # (21, 128, 128, 3)


def preprocess_eval(clip):
    """Mirrors train_clip_model.ClipDS.__getitem__ for train=False."""
    T_total = clip.shape[0]
    t0 = (T_total - T_FRAMES) // 2
    clip = clip[t0:t0 + T_FRAMES]
    x = torch.from_numpy(clip).float() / 255.0
    x = x.permute(3, 0, 1, 2)
    off = (128 - SIZE) // 2
    x = x[:, :, off:off + SIZE, off:off + SIZE]
    return (x - MEAN) / STD


@torch.no_grad()
def score_clip(model, clip):
    x = preprocess_eval(clip).unsqueeze(0).to(DEV)
    return float(torch.sigmoid(model(x).squeeze(1)).item())


def run(folder, thr, cooldown, top, make_video, freeze_sec, make_shots, out_folder=None,
        ckpt_path=None, tag="r3d18"):
    out_folder = out_folder or folder
    ckpt_path = ckpt_path or DEFAULT_CKPT_PATH
    sam3d_p, fa_p, vid = v9.find_files(folder)
    name = os.path.basename(sam3d_p).replace("_sam3d.json", "")
    if not vid:
        raise SystemExit("[r3d18] no raw video found in folder; required for clip extraction.")
    s = json.load(open(sam3d_p))
    fa = json.load(open(fa_p))
    persons = {0: {e["frame"]: e for e in s.get("0", [])}, 1: {e["frame"]: e for e in s.get("1", [])}}
    actions = fa["actions"]
    fps = float(fa.get("processing_stats", {}).get("total_frames", 0)) / \
        float(fa.get("processing_stats", {}).get("video_duration", 1)) or 30.0
    print(f"[r3d18] {len(actions)} ASFormer action candidates, fps~{fps:.3f}")

    model = load_model(ckpt_path)
    cap = cv2.VideoCapture(vid)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scored, skipped = [], 0
    for act in actions:
        sid = 0 if act["fighter_type"] == "fighter_0" else 1
        rid = 1 - sid
        clip_meta = {"fighter_id": sid, "window_start": act["window_start"], "window_end": act["window_end"]}
        result = find_contact_frame_and_box(clip_meta, persons)
        if result is None:
            skipped += 1
            continue
        best_f, cx, cy, size = result
        clip = build_clip(cap, n_frames, best_f, cx, cy, size)
        if clip is None:
            skipped += 1
            continue
        prob = score_clip(model, clip)
        # contact region for the marker: reuse v9's wrist-body-gap region call
        sp, rp = persons.get(sid, {}), persons.get(rid, {})
        _, reg = v9.min_wrist_body_gap(
            v9.wc(sp.get(best_f)) if sp.get(best_f) else None,
            v9.wc(rp.get(best_f)) if rp.get(best_f) else None)
        scored.append(dict(frame=best_f, score=prob, contact_region=reg or "torso",
                            striker_id=sid, receiver_id=rid, action=act["action"],
                            window_start=act["window_start"], window_end=act["window_end"]))
    cap.release()
    print(f"[r3d18] scored {len(scored)} candidates ({skipped} skipped)")

    scored.sort(key=lambda e: -e["score"])
    kept = v9.nms([e for e in scored if e["score"] >= thr], cooldown)
    kept.sort(key=lambda e: e["frame"])
    print(f"[r3d18] {len(kept)} pass thr={thr} (cooldown={cooldown})")
    for i, e in enumerate(sorted(kept, key=lambda x: -x["score"])[:top], 1):
        print(f"  {i:3d}  frame={e['frame']:6d}  t={e['frame']/fps:6.2f}s  "
              f"score={e['score']:.3f}  {e['contact_region']:6s}  "
              f"fighter_{e['striker_id']}  {e['action']}")

    out = {
        "approach": "r3d18_video_cnn", "model": ckpt_path,
        "source_sam3d": os.path.basename(sam3d_p), "fps": round(fps, 3),
        "threshold": thr, "cooldown": cooldown,
        "n_candidates_scored": len(scored), "n_impacts": len(kept),
        "impacts": [{
            "is_impact": True, "impact_frame": e["frame"],
            "timestamp_seconds": round(e["frame"] / fps, 3),
            "impact_score": round(e["score"], 4), "contact_region": e["contact_region"],
            "striker_id": e["striker_id"], "receiver_id": e["receiver_id"], "action": e["action"],
        } for e in kept],
        "all_scored_candidates": [{
            "impact_frame": e["frame"], "timestamp_seconds": round(e["frame"] / fps, 3),
            "impact_score": round(e["score"], 4), "contact_region": e["contact_region"],
            "action": e["action"], "striker_id": e["striker_id"], "receiver_id": e["receiver_id"],
            "window_start": e["window_start"], "window_end": e["window_end"],
        } for e in scored],
    }
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{name}_impacts_{tag}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[r3d18] saved -> {out_path}")

    if make_shots and vid:
        v9.save_impact_shots(out_folder, vid, persons, kept, fps, tag=tag)
        v9.save_impact_strips(out_folder, vid, persons, kept, fps, tag=tag)
    if make_video and vid:
        v9.render_video(out_folder, vid, persons, kept, fps, freeze_sec, tag=tag)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out-folder", default=None,
                    help="write json/video/shots here instead of --folder (still reads source files from --folder)")
    ap.add_argument("--model", default=DEFAULT_CKPT_PATH, help="r3d_18 checkpoint path")
    ap.add_argument("--tag", default="r3d18", help="filename tag for outputs (json/video/shots), e.g. 'r3d18_v2'")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cooldown", type=int, default=18)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--freeze", type=float, default=1.5)
    ap.add_argument("--shots", action="store_true")
    args = ap.parse_args()
    run(args.folder, args.thr, args.cooldown, args.top, args.video, args.freeze, args.shots,
        out_folder=args.out_folder, ckpt_path=args.model, tag=args.tag)


if __name__ == "__main__":
    main()
