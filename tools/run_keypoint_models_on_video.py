#!/usr/bin/env python3
"""
Score a fight folder's ASFormer action candidates with the trained keypoint
sequence models (TCN / ASFormer / BRT, tools/svtas_models/train_compare.py),
then dedup/NMS and render annotated videos -- same output shape as v9/XGBoost,
so all approaches can be compared on the same video.

Builds the exact same 88-dim feature sequence as
tools/extract_keypoint_dataset.py (verified MHR70 joints: head/shoulders/
elbows/hips/wrists -- see config.py's KP70_* fix), driven from this video's
ASFormer action candidates (full_analysis.json) instead of manifest.json
labels. Assumes no CUTIE identity-swap correction is needed (only fights
with an identity_marker configured in dataset/fights.py need that; pass
--swap-marker if the target fight has one).

Usage:
    python tools/run_keypoint_models_on_video.py --folder "C:/Users/XRIG/Downloads/1st_Impact_detection_Fixed_05062026"
    python tools/run_keypoint_models_on_video.py --folder "<dir>" --models tcn asformer brt --video
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + os.sep + "svtas_models")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_keypoint_dataset import (  # noqa: E402
    cam_space_coords, build_reference_frame, compute_frame_row, pad_and_mask,
    hand_from_action, forward_fill_entry, FEATURE_NAMES, PAD_BEFORE, PAD_END,
)
import config as cfg  # noqa: E402
from model_factory import build_model  # noqa: E402
from detectors.fusion import v9  # noqa: E402

T_MAX = 41  # matches training (outputs/keypoint_dataset/combined.npz observed max)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "outputs", "svtas_models")


def load_models(names):
    models = {}
    for name in names:
        ckpt_path = os.path.join(CKPT_DIR, f"{name}_FINAL_alldata.pt")
        m = build_model(name, num_features=len(FEATURE_NAMES))
        m.load_state_dict(torch.load(ckpt_path, map_location=DEV))
        m.to(DEV).eval()
        models[name] = m
        print(f"[kp] loaded {name} <- {ckpt_path}")
    return models


def extract_action_sequence(act, sam3d_round, fighter_id):
    """Mirrors extract_keypoint_dataset.extract_clip, driven from an
    ASFormer action dict instead of a manifest clip entry. No swap timeline
    (caller's fight has no identity_marker)."""
    action = act["action"]
    window_start = act["window_start"]
    window_end = act["window_end"]

    frame_lo = max(0, window_start - PAD_BEFORE)
    frame_hi = window_end + PAD_END
    frame_list = list(range(frame_lo, frame_hi + 1))

    striker_track = sam3d_round[str(fighter_id)]
    receiver_track = sam3d_round[str(1 - fighter_id)]

    wrist_idx, is_left = hand_from_action(action)
    elbow_idx, shoulder_idx = cfg.ACTION_ARM_CHAIN[wrist_idx]

    ref_origin, ref_scale = build_reference_frame(receiver_track, window_start)

    rows = []
    last_striker, last_receiver = None, None
    state = {}
    for f in frame_list:
        s_entry, last_striker = forward_fill_entry(striker_track, f, last_striker)
        r_entry, last_receiver = forward_fill_entry(receiver_track, f, last_receiver)
        s_coords = (cam_space_coords(s_entry) - ref_origin) / ref_scale
        r_coords = (cam_space_coords(r_entry) - ref_origin) / ref_scale
        rows.append(compute_frame_row(s_coords, r_coords, wrist_idx, elbow_idx, shoulder_idx,
                                       is_left, state))
    return np.stack(rows)


def find_contact_frame_and_region(act, persons, fighter_id):
    """Reuses v9's wrist-body-gap scan (on world_coords, independent of the
    receiver-relative features above) purely for the impact marker's
    frame/region in rendering -- not used by the model itself."""
    sid, rid = fighter_id, 1 - fighter_id
    sp, rp = persons.get(sid, {}), persons.get(rid, {})
    ws, we = act["window_start"] - 3, act["window_end"] + 5
    best_f, best_gap, best_reg = act["window_start"], 1e9, "torso"
    for f in range(ws, we + 1):
        se, re = sp.get(f), rp.get(f)
        g, reg = v9.min_wrist_body_gap(v9.wc(se) if se else None, v9.wc(re) if re else None)
        if g is not None and g < best_gap:
            best_gap, best_f, best_reg = g, f, reg
    return best_f, best_reg


@torch.no_grad()
def score_with_model(model, seq, mask):
    x = torch.from_numpy(seq).float().unsqueeze(0).to(DEV)
    m = torch.from_numpy(mask).float().unsqueeze(0).to(DEV)
    return float(torch.sigmoid(model(x, m)).item())


def run(folder, model_names, thr, cooldown, top, make_video, freeze_sec, make_shots, out_folder=None,
        tag_suffix=""):
    out_folder = out_folder or folder
    sam3d_p, fa_p, vid = v9.find_files(folder)
    name = os.path.basename(sam3d_p).replace("_sam3d.json", "")
    s = json.load(open(sam3d_p))
    fa = json.load(open(fa_p))
    sam3d_round = {tid: {e["frame"]: e for e in s.get(tid, [])} for tid in ("0", "1")}
    persons = {0: sam3d_round["0"], 1: sam3d_round["1"]}
    actions = fa["actions"]
    fps = float(fa.get("processing_stats", {}).get("total_frames", 0)) / \
        float(fa.get("processing_stats", {}).get("video_duration", 1)) or 30.0
    print(f"[kp] {len(actions)} ASFormer action candidates, fps~{fps:.3f}")

    models = load_models(model_names)

    scored = {mn: [] for mn in model_names}
    skipped = 0
    for act in actions:
        fighter_id = 0 if act["fighter_type"] == "fighter_0" else 1
        try:
            seq = extract_action_sequence(act, sam3d_round, fighter_id)
        except Exception:
            skipped += 1
            continue
        padded, mask = pad_and_mask(seq.astype(np.float32), T_MAX)
        best_f, best_reg = find_contact_frame_and_region(act, persons, fighter_id)
        for mn, model in models.items():
            prob = score_with_model(model, padded, mask)
            scored[mn].append(dict(frame=best_f, score=prob, contact_region=best_reg,
                                    striker_id=fighter_id, receiver_id=1 - fighter_id,
                                    action=act["action"], window_start=act["window_start"],
                                    window_end=act["window_end"]))
    print(f"[kp] scored {len(actions) - skipped} candidates per model ({skipped} skipped)")

    results = {}
    for mn in model_names:
        tag = mn + tag_suffix
        evs = sorted(scored[mn], key=lambda e: -e["score"])
        kept = v9.nms([e for e in evs if e["score"] >= thr], cooldown)
        kept.sort(key=lambda e: e["frame"])
        print(f"\n[kp:{mn}] {len(kept)} pass thr={thr} (cooldown={cooldown})")
        for i, e in enumerate(sorted(kept, key=lambda x: -x["score"])[:top], 1):
            print(f"  {i:3d}  frame={e['frame']:6d}  t={e['frame']/fps:6.2f}s  "
                  f"score={e['score']:.3f}  {e['contact_region']:6s}  "
                  f"fighter_{e['striker_id']}  {e['action']}")

        out = {
            "approach": f"keypoint_{mn}", "model": f"outputs/svtas_models/{mn}_FINAL_alldata.pt",
            "source_sam3d": os.path.basename(sam3d_p), "fps": round(fps, 3),
            "threshold": thr, "cooldown": cooldown,
            "n_candidates_scored": len(scored[mn]), "n_impacts": len(kept),
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
            } for e in scored[mn]],
        }
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{name}_impacts_{tag}.json")
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"[kp:{mn}] saved -> {out_path}")
        results[mn] = (kept, persons, fps)

        if make_shots and vid:
            v9.save_impact_shots(out_folder, vid, persons, kept, fps, tag=tag)
            v9.save_impact_strips(out_folder, vid, persons, kept, fps, tag=tag)
        if make_video and vid:
            v9.render_video(out_folder, vid, persons, kept, fps, freeze_sec, tag=tag)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out-folder", default=None,
                     help="write json/video/shots here instead of --folder (still reads source files from --folder)")
    ap.add_argument("--models", nargs="+", default=["tcn", "asformer", "brt"],
                     choices=["tcn", "asformer", "brt"])
    ap.add_argument("--tag-suffix", default="", help="appended to each model's tag, e.g. '_v2'")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cooldown", type=int, default=18)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--freeze", type=float, default=1.5)
    ap.add_argument("--shots", action="store_true")
    args = ap.parse_args()

    run(args.folder, args.models, args.thr, args.cooldown, args.top,
        args.video, args.freeze, args.shots, out_folder=args.out_folder, tag_suffix=args.tag_suffix)


if __name__ == "__main__":
    main()
