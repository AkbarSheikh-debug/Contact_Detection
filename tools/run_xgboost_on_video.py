#!/usr/bin/env python3
"""
Score a fight folder's ASFormer action candidates with the trained XGBoost
impact classifier (tools/train_xgboost_impact.py), then dedup/NMS and
optionally render an annotated video / impact screenshots -- same output
shapes as detectors/fusion/v9.py, so the two can be compared directly.

Usage:
    python tools/run_xgboost_on_video.py --folder "C:/Users/XRIG/Downloads/1st_Impact_detection_Fixed_05062026"
    python tools/run_xgboost_on_video.py --folder "<dir>" --thr 0.5 --video --shots
"""
import os
import sys
import json
import argparse

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_xgboost_impact import extract_features, FEATURE_NAMES
from detectors.fusion import v9


def load_model(path):
    model = XGBClassifier()
    model.load_model(path)
    return model


def score_folder(folder, model, thr, cooldown, top, out_folder=None, tag="xgb"):
    out_folder = out_folder or folder
    sam3d_p, fa_p, vid = v9.find_files(folder)
    name = os.path.basename(sam3d_p).replace("_sam3d.json", "")
    print(f"[xgb] sam3d={os.path.basename(sam3d_p)}  full_analysis={os.path.basename(fa_p)}")

    s = json.load(open(sam3d_p))
    fa = json.load(open(fa_p))
    persons = {0: {e["frame"]: e for e in s.get("0", [])},
               1: {e["frame"]: e for e in s.get("1", [])}}
    actions = fa["actions"]
    fps = float(fa.get("processing_stats", {}).get("total_frames", 0)) / \
        float(fa.get("processing_stats", {}).get("video_duration", 1)) or 30.0
    print(f"[xgb] {len(actions)} ASFormer action candidates, fps~{fps:.3f}")

    conf_by_key = {}
    for act in actions:
        sid = 0 if act["fighter_type"] == "fighter_0" else 1
        conf_by_key[(sid, act["window_start"], act["window_end"])] = act.get("confidence", 1.0)

    scored, skipped = [], 0
    for act in actions:
        sid = 0 if act["fighter_type"] == "fighter_0" else 1
        rid = 1 - sid
        clip = {"fighter_id": sid, "window_start": act["window_start"], "window_end": act["window_end"]}
        result = extract_features(clip, persons, conf_by_key, return_meta=True)
        if result is None:
            skipped += 1
            continue
        feats, best_f, best_reg = result
        prob = float(model.predict_proba(np.array([feats]))[0, 1])
        scored.append(dict(frame=best_f, score=prob, contact_region=best_reg,
                            striker_id=sid, receiver_id=rid, action=act["action"],
                            confidence=act.get("confidence", 1.0),
                            window_start=act["window_start"], window_end=act["window_end"]))

    print(f"[xgb] scored {len(scored)} candidates ({skipped} skipped, no usable keypoints)")
    scored.sort(key=lambda e: -e["score"])
    kept = v9.nms([e for e in scored if e["score"] >= thr], cooldown)
    kept.sort(key=lambda e: e["frame"])

    print(f"[xgb] {len(kept)} pass thr={thr} (cooldown={cooldown} frames)")
    print(f"[xgb] region mix: {dict((r, sum(1 for e in kept if e['contact_region'] == r)) for r in ('head', 'torso'))}")
    print(f"\n  rank  frame   t(s)   score  region   striker  action")
    print("  " + "-" * 60)
    for i, e in enumerate(sorted(kept, key=lambda x: -x["score"])[:top], 1):
        print(f"  {i:3d}  {e['frame']:6d}  {e['frame']/fps:6.2f}  {e['score']:.3f}  "
              f"{e['contact_region']:7s}  fighter_{e['striker_id']}  {e['action']}")

    out = {
        "approach": "xgboost_v1",
        "model": "outputs/xgb_impact_v1.json",
        "source_sam3d": os.path.basename(sam3d_p),
        "source_full_analysis": os.path.basename(fa_p),
        "fps": round(fps, 3),
        "threshold": thr, "cooldown": cooldown,
        "n_candidates_scored": len(scored),
        "n_impacts": len(kept),
        "impacts": [{
            "is_impact": True, "impact_frame": e["frame"],
            "timestamp_seconds": round(e["frame"] / fps, 3),
            "impact_score": round(e["score"], 4),
            "contact_region": e["contact_region"],
            "striker_id": e["striker_id"], "receiver_id": e["receiver_id"],
            "action": e["action"],
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
    print(f"\n[xgb] saved -> {out_path}")
    return out_path, persons, kept, fps, vid, out_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out-folder", default=None,
                    help="write json/video/shots here instead of --folder (still reads source files from --folder)")
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "xgb_impact_v2.json"))
    ap.add_argument("--tag", default="xgb", help="filename tag for outputs (json/video/shots), e.g. 'xgb_v2'")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cooldown", type=int, default=18)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--video", action="store_true", help="render annotated mp4 (reuses v9's renderer)")
    ap.add_argument("--freeze", type=float, default=1.5)
    ap.add_argument("--shots", action="store_true", help="save annotated impact screenshots/strips")
    args = ap.parse_args()

    model = load_model(args.model)
    out_path, persons, kept, fps, vid, out_folder = score_folder(
        folder=args.folder, model=model, thr=args.thr, cooldown=args.cooldown, top=args.top,
        out_folder=args.out_folder, tag=args.tag)

    if args.shots and vid:
        v9.save_impact_shots(out_folder, vid, persons, kept, fps, tag=args.tag)
        v9.save_impact_strips(out_folder, vid, persons, kept, fps, tag=args.tag)
    elif args.shots:
        print("[xgb] no raw video found in folder; cannot save shots.")
    if args.video and vid:
        v9.render_video(out_folder, vid, persons, kept, fps, args.freeze, tag=args.tag)
    elif args.video:
        print("[xgb] no raw video found in folder; cannot render.")


if __name__ == "__main__":
    main()
