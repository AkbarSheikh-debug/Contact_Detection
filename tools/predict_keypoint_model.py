#!/usr/bin/env python3
"""
Run the trained keypoint impact/not_impact model on a brand-new video that
has already been processed by the SAM3D pipeline (a *_sam3d.json keypoint
file and a *_full_analysis.json action-window file, in the same format as
the live production pipeline -- NOT the lillyella_vs_zoe training format).

Reuses the geometry/feature-engineering helpers from
tools/extract_keypoint_dataset.py (camera-space reconstruction, the
receiver-anchored shared frame, the 9 engineered features + COCO-17 raw
keypoints) but with a generic per-action extraction path: no manifest, no
round-based file layout, and no swap-correction (this video's sam3d.json
track_id was spot-checked at 3 widely-spaced frames -- start/middle/end --
and found stable, with no identity swap, unlike the training fight which
needed correction; see conversation for the swap-correction approach if a
future video DOES show a swap and needs it).

Ensembles all 4 leave-one-round-out checkpoints (one per training fold) by
averaging their predicted probabilities -- a reasonable choice for a truly
new, unseen video, since each fold saw a different 3/4 of the training data.

Usage:
  python tools/predict_keypoint_model.py --sam3d <path_sam3d.json> --actions <path_full_analysis.json>
  python tools/predict_keypoint_model.py --sam3d <path> --actions <path> --out outputs/predictions.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_keypoint_dataset import (  # noqa: E402
    PAD_BEFORE, PAD_END, FEATURE_NAMES,
    cam_space_coords, hand_from_action, forward_fill_entry,
    build_reference_frame, compute_frame_row,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "svtas_models"))
from model_factory import build_model, MODEL_NAMES  # noqa: E402 -- spans tcn/gru/asformer/brt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_model"
DEFAULT_CKPTS = [os.path.join(CKPT_DIR, f"tcn_round{r}_best.pt") for r in (3, 4, 5, 8)]


def load_sam3d_tracks(path):
    d = json.load(open(path))
    return {tid: {e["frame"]: e for e in d[tid]} for tid in ("0", "1")}


def load_actions(path):
    d = json.load(open(path))
    return d["actions"]


def fighter_type_to_id(fighter_type):
    m = re.match(r"fighter_(\d+)$", fighter_type)
    if not m:
        raise ValueError(f"Unexpected fighter_type: {fighter_type!r}")
    return int(m.group(1))


def extract_action_features(action, tracks):
    """Generic per-action extraction: no manifest/round-based file lookups,
    no swap correction (striker_tid = fighter_id directly, since this
    video's tracking was spot-checked stable). Shares the exact same
    reference-frame + per-frame feature logic as
    extract_keypoint_dataset.extract_clip via build_reference_frame /
    compute_frame_row, so training and inference can never silently drift
    apart on the feature definitions."""
    fighter_id = fighter_type_to_id(action["fighter_type"])
    action_name = action["action"]
    window_start = action["window_start"]
    window_end = action["window_end"]

    frame_lo = max(0, window_start - PAD_BEFORE)
    frame_hi = window_end + PAD_END
    frame_list = list(range(frame_lo, frame_hi + 1))

    striker_tid = str(fighter_id)
    receiver_tid = str(1 - fighter_id)
    striker_track = tracks[striker_tid]
    receiver_track = tracks[receiver_tid]

    wrist_idx, is_left = hand_from_action(action_name)
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

    return np.stack(rows)  # (T, F)


def load_ensemble(ckpt_paths, num_features, model_type="tcn"):
    models = []
    for p in ckpt_paths:
        if not os.path.exists(p):
            print(f"  [WARN] checkpoint not found, skipping: {p}")
            continue
        m = build_model(model_type, num_features)
        m.load_state_dict(torch.load(p, map_location=DEV))
        m.to(DEV).eval()
        models.append(m)
    if not models:
        raise RuntimeError("No checkpoints loaded -- nothing to ensemble.")
    return models


def predict_proba(models, seq):
    x = torch.from_numpy(seq).float().unsqueeze(0).to(DEV)        # [1,T,F]
    mask = torch.ones(1, seq.shape[0], dtype=torch.float32, device=DEV)
    probs = []
    with torch.no_grad():
        for m in models:
            logit = m(x, mask)
            probs.append(torch.sigmoid(logit).item())
    return float(np.mean(probs)), probs


def add_calibration(results, impact_pctile=90, borderline_pctile=70, max_plausible_dist=1.5):
    """Add a within-video percentile rank + 3-tier label alongside the raw
    probability. The model's absolute 0.5 threshold was calibrated on one
    training video's distance scale; on a new video with a systematically
    different scale (different camera/pipeline run), absolute probabilities
    can run uniformly low even when the model's RELATIVE ranking is still
    informative (verified: rank correlation with the underlying distance
    feature held up on an out-of-distribution video). This surfaces the
    top-ranked candidates within THIS video as "borderline"/"impact" even
    if their raw probability alone wouldn't clear 0.5, rather than silently
    dropping them.

    Tier logic: "impact" if prob > 0.5 (the model is absolutely confident)
    OR percentile >= impact_pctile (top of this video's own distribution);
    "borderline" if percentile >= borderline_pctile; else "not_impact".

    max_plausible_dist: a physical sanity gate, in shoulder-widths. A
    candidate whose closest approach (min_wrist_to_torso_dist) never got
    within this many shoulder-widths of the receiver's torso CANNOT be a
    landed punch, no matter how it ranks against this video's other
    candidates -- this catches the failure mode where a whole video's
    distance signal is uniformly weak (tracking corruption, bad scale, a
    different SAM3D pipeline run than training used) and percentile-only
    calibration would otherwise surface implausible candidates as "impact"
    just for being the least-bad of a bad batch. Found on cameron_vs_liam
    Round 2: candidates with dist 1.2-5.9 shoulder-widths (1-2.5m at a
    typical zoom) were getting labeled "impact" purely by relative rank.
    Overrides the percentile tier, never the raw prob>0.5 case -- a model
    that's absolutely confident gets to keep that call even if the distance
    feature looks off (e.g. a momentary tracking glitch IS one explanation
    the model itself might be reacting to, not just noise).
    """
    probs = np.array([r["impact_probability"] for r in results])
    ranks = np.argsort(np.argsort(probs))  # 0 = lowest
    n = len(probs)
    n_gated = 0
    for i, r in enumerate(results):
        pctile = 100.0 * ranks[i] / max(n - 1, 1)
        r["video_percentile"] = round(float(pctile), 1)
        implausible = r.get("min_wrist_to_torso_dist", 0.0) > max_plausible_dist
        r["implausible_distance"] = implausible

        if r["impact_probability"] > 0.5:
            r["calibrated_label"] = "impact"
        elif implausible:
            r["calibrated_label"] = "not_impact"
            n_gated += 1
        elif pctile >= impact_pctile:
            r["calibrated_label"] = "impact"
        elif pctile >= borderline_pctile:
            r["calibrated_label"] = "borderline"
        else:
            r["calibrated_label"] = "not_impact"
    if n_gated:
        print(f"  [calibration] {n_gated}/{n} percentile-driven candidates gated to "
              f"not_impact (min_wrist_to_torso_dist > {max_plausible_dist} shoulder-widths)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3d", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--model-type", default="tcn", choices=MODEL_NAMES,
                     help="architecture the checkpoints were trained with (tcn/gru/asformer/brt)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-plausible-dist", type=float, default=1.5,
                     help="physical sanity gate (shoulder-widths) for the calibrated "
                          "impact tier -- see add_calibration() docstring")
    args = ap.parse_args()

    print(f"Loading SAM3D keypoints from {args.sam3d} ...")
    tracks = load_sam3d_tracks(args.sam3d)
    print(f"Loading actions from {args.actions} ...")
    actions = load_actions(args.actions)
    print(f"{len(actions)} candidate actions, device={DEV}")

    models = load_ensemble(args.ckpts, num_features=len(FEATURE_NAMES), model_type=args.model_type)
    print(f"Loaded {len(models)} model(s) for ensembling: {[os.path.basename(p) for p in args.ckpts if os.path.exists(p)]}")

    wtd_col = FEATURE_NAMES.index("wrist_to_receiver_torso_dist")
    reach_col = FEATURE_NAMES.index("reach_ratio")

    results = []
    n_skipped = 0
    for i, a in enumerate(actions):
        try:
            seq = extract_action_features(a, tracks)
            prob, per_model = predict_proba(models, seq)
        except Exception as e:
            print(f"  [SKIP] action {i} ({a['action']} @ frame {a['frame']}): {e}")
            n_skipped += 1
            continue

        results.append({
            "frame": a["frame"],
            "window_start": a["window_start"],
            "window_end": a["window_end"],
            "timestamp_seconds": a.get("timestamp_seconds"),
            "fighter_type": a["fighter_type"],
            "action": a["action"],
            "asformer_confidence": a.get("confidence"),
            "target": a.get("target"),
            "impact_probability": prob,
            "predicted_label": "impact" if prob > args.threshold else "not_impact",
            "per_model_probs": per_model,
            "min_wrist_to_torso_dist": float(seq[:, wtd_col].min()),
            "min_reach_ratio": float(seq[:, reach_col].min()),
        })

    results = add_calibration(results, max_plausible_dist=args.max_plausible_dist)

    n_impact = sum(1 for r in results if r["predicted_label"] == "impact")
    n_cal_impact = sum(1 for r in results if r["calibrated_label"] == "impact")
    n_cal_border = sum(1 for r in results if r["calibrated_label"] == "borderline")
    print(f"\n{len(results)} predictions ({n_skipped} skipped)")
    print(f"  raw 0.5-threshold:  impact={n_impact}  not_impact={len(results) - n_impact}")
    print(f"  calibrated 3-tier:  impact={n_cal_impact}  borderline={n_cal_border}  "
          f"not_impact={len(results) - n_cal_impact - n_cal_border}")
    print(f"  mean impact_probability: {np.mean([r['impact_probability'] for r in results]):.3f}")

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.sam3d))[0].replace("_sam3d", "")
        out_path = os.path.join(
            r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_model",
            f"{base}_predictions.json",
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({
        "sam3d_path": args.sam3d, "actions_path": args.actions,
        "checkpoints": args.ckpts, "threshold": args.threshold,
        "n_actions": len(results), "n_impact": n_impact,
        "n_not_impact": len(results) - n_impact,
        "predictions": results,
    }, open(out_path, "w"), indent=2)
    print(f"\nSaved predictions -> {out_path}")


if __name__ == "__main__":
    main()
