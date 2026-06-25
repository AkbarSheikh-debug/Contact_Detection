#!/usr/bin/env python3
"""
Render an actual v9+model COMBINED impact video, using the rank-averaging
that tools/ensemble_v9_with_models.py found to help for asformer/brt/r3d18
(see that script's docstring for why rank-averaging, not raw score-averaging).

Reads v9's and the target model's already-saved *_impacts_v9.json /
*_impacts_<model>.json (all_scored_candidates) for a folder, combines their
ranks 50/50 per candidate, re-applies NMS, and renders/saves shots+strips
through v9's existing renderer -- tagged "v9_<model>" so it sits alongside
the standalone videos without overwriting them.

Usage:
    python tools/render_ensemble_video.py --folder "<source>" --out-folder "<dest>" --model asformer
    python tools/render_ensemble_video.py --folder "<source>" --out-folder "<dest>" --models asformer brt r3d18 --video --shots
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors.fusion import v9


def load_candidates(folder, tag):
    matches = [f for f in os.listdir(folder) if f.endswith(f"_impacts_{tag}.json")]
    if not matches:
        raise SystemExit(f"no *_impacts_{tag}.json found in {folder} -- run the standalone scorer for "
                          f"'{tag}' first (with --out-folder pointing here if applicable).")
    d = json.load(open(os.path.join(folder, matches[0])))
    by_key = {}
    for e in d["all_scored_candidates"]:
        key = (e["striker_id"], e["window_start"], e["window_end"])
        by_key[key] = dict(score=e["impact_score"], frame=e["impact_frame"],
                            contact_region=e["contact_region"], action=e["action"],
                            receiver_id=1 - e["striker_id"])
    return by_key


def rank_normalize(values):
    order = np.argsort(values).argsort()
    return order / max(1, len(values) - 1)


def build_combined(folder, model, thr, cooldown):
    v9_cands = load_candidates(folder, "v9")
    model_cands = load_candidates(folder, model)
    all_keys = sorted(set(v9_cands) | set(model_cands))

    v9_vals = np.array([v9_cands.get(k, {}).get("score", 0.0) for k in all_keys])
    model_vals = np.array([model_cands[k]["score"] if k in model_cands else 0.0 for k in all_keys])
    v9_rank = rank_normalize(v9_vals)
    model_rank = rank_normalize(model_vals)
    combined = 0.5 * v9_rank + 0.5 * model_rank

    events = []
    for key, c_score in zip(all_keys, combined):
        meta = model_cands.get(key) or v9_cands.get(key)
        events.append(dict(frame=meta["frame"], score=float(c_score),
                            contact_region=meta["contact_region"],
                            striker_id=key[0], receiver_id=1 - key[0], action=meta["action"]))

    events.sort(key=lambda e: -e["score"])
    kept = v9.nms([e for e in events if e["score"] >= thr], cooldown)
    kept.sort(key=lambda e: e["frame"])
    return events, kept


def run(folder, out_folder, models, thr, cooldown, top, make_video, freeze_sec, make_shots):
    sam3d_p, fa_p, vid = v9.find_files(folder)
    name = os.path.basename(sam3d_p).replace("_sam3d.json", "")
    s = json.load(open(sam3d_p))
    fa = json.load(open(fa_p))
    persons = {0: {e["frame"]: e for e in s.get("0", [])}, 1: {e["frame"]: e for e in s.get("1", [])}}
    fps = float(fa.get("processing_stats", {}).get("total_frames", 0)) / \
        float(fa.get("processing_stats", {}).get("video_duration", 1)) or 30.0

    os.makedirs(out_folder, exist_ok=True)
    for model in models:
        tag = f"v9_{model}"
        print(f"\n[ens] === {tag} ===")
        events, kept = build_combined(out_folder, model, thr, cooldown)
        print(f"[ens] {len(events)} candidates combined; {len(kept)} pass thr={thr} (cooldown={cooldown})")
        for i, e in enumerate(sorted(kept, key=lambda x: -x["score"])[:top], 1):
            print(f"  {i:3d}  frame={e['frame']:6d}  t={e['frame']/fps:6.2f}s  "
                  f"combined_rank={e['score']:.3f}  {e['contact_region']:6s}  "
                  f"fighter_{e['striker_id']}  {e['action']}")

        out = {
            "approach": tag, "note": "rank-averaged combination of v9 + model scores, NOT a trained ensemble",
            "fps": round(fps, 3), "threshold": thr, "cooldown": cooldown,
            "n_candidates_scored": len(events), "n_impacts": len(kept),
            "impacts": [{
                "is_impact": True, "impact_frame": e["frame"],
                "timestamp_seconds": round(e["frame"] / fps, 3),
                "impact_score": round(e["score"], 4), "contact_region": e["contact_region"],
                "striker_id": e["striker_id"], "receiver_id": e["receiver_id"], "action": e["action"],
            } for e in kept],
        }
        out_path = os.path.join(out_folder, f"{name}_impacts_{tag}.json")
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"[ens] saved -> {out_path}")

        if make_shots and vid:
            v9.save_impact_shots(out_folder, vid, persons, kept, fps, tag=tag)
            v9.save_impact_strips(out_folder, vid, persons, kept, fps, tag=tag)
        if make_video and vid:
            v9.render_video(out_folder, vid, persons, kept, fps, freeze_sec, tag=tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="source folder with sam3d/full_analysis/video")
    ap.add_argument("--out-folder", default=None,
                    help="folder containing the already-scored *_impacts_v9.json / *_impacts_<model>.json "
                         "(and where outputs are written); defaults to --folder")
    ap.add_argument("--models", nargs="+", default=["asformer", "brt", "r3d18"],
                    choices=["xgb", "tcn", "asformer", "brt", "r3d18"])
    ap.add_argument("--thr", type=float, default=0.7, help="combined-rank threshold (0-1)")
    ap.add_argument("--cooldown", type=int, default=18)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--freeze", type=float, default=1.5)
    ap.add_argument("--shots", action="store_true")
    args = ap.parse_args()

    out_folder = args.out_folder or args.folder
    run(args.folder, out_folder, args.models, args.thr, args.cooldown, args.top,
        args.video, args.freeze, args.shots)


if __name__ == "__main__":
    main()
