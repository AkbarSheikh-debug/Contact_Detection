#!/usr/bin/env python3
"""
Impact-anchored keypoint feature extraction for training.

Unlike extract_keypoint_dataset.py (anchored at window_start), this script
anchors each clip's feature window at the manually-marked impact_frame --
the exact contact frame verified in the annotation tool.

For clips without a marked frame (not_impact labels, or impact clips not
yet reviewed), the anchor falls back to window_end (ASFormer's peak-action
estimate, the closest proxy we have).

Consistent anchoring means the model always sees the same temporal structure:
  frames[0 .. PRE-1]   : striker approaching
  frames[PRE]          : anchor (contact or best estimate)
  frames[PRE+1 .. end] : receiver reaction (head snap, body recoil)

Two extra features per frame give the model explicit timing information:
  frame_offset_from_anchor : (frame - anchor) / POST_ANCHOR
                             0.0 at contact, negative before, positive after
  has_anchor_frame         : 1.0 if exact impact_frame marked, 0.0 if proxy

Usage:
  python tools/extract_impact_anchored_dataset.py
  python tools/extract_impact_anchored_dataset.py --fights 1st_fight 2nd_fight
  python tools/extract_impact_anchored_dataset.py --out outputs/keypoint_dataset/anchored.npz
"""
import os
import sys
import json
import argparse

import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)       # tools/ — for annotate_clips
sys.path.insert(0, _REPO_ROOT)  # repo root — for config.py

import annotate_clips as ac
import config as cfg
from dataset import fights

# ── window parameters ────────────────────────────────────────────────────────
PRE_ANCHOR  = 12   # frames before anchor  (≈0.5 s at 25 fps)
POST_ANCHOR = 20   # frames after anchor   (≈0.8 s at 25 fps)
T_FIXED     = PRE_ANCHOR + POST_ANCHOR + 1   # 33 frames, fixed-length window
REF_FRAMES  = 5    # frames to average for the coordinate reference frame
REACH_SENTINEL = 5.0

# Verified joint subset (see config.py + extract_keypoint_dataset.py comments
# on why these 13 joints are trusted and knees/ankles are excluded).
RAW_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 41, 62]

FEATURE_NAMES = (
    [f"striker_kp{j}_{ax}"  for j in RAW_JOINTS for ax in ("x", "y", "z")]
    + [f"receiver_kp{j}_{ax}" for j in RAW_JOINTS for ax in ("x", "y", "z")]
    + [
        "wrist_to_receiver_torso_dist",
        "wrist_to_receiver_head_dist",
        "wrist_velocity",
        "wrist_acceleration",
        "receiver_head_velocity",
        "receiver_head_acceleration",
        "arm_extension_ratio",
        "striker_hip_to_receiver_hip_dist",
        "is_left_hand",
        "reach_ratio",
        # Impact-anchoring extras:
        "frame_offset_from_anchor",   # (f - anchor) / POST_ANCHOR
        "has_anchor_frame",           # 1.0=exact mark, 0.0=proxy
    ]
)

LABEL_TO_Y = {"impact": 1, "not_impact": 0}

DEFAULT_OUT = os.path.join(_REPO_ROOT, "outputs", "keypoint_dataset", "1st_fight_anchored.npz")


# ── SAM3D / swap helpers ─────────────────────────────────────────────────────

def configure_for_fight(fight_cfg):
    ac.VIDEO_FOLDER        = fight_cfg["video_folder"]
    ac.IDENTITY_MARKER     = fight_cfg.get("identity_marker")
    ac.ROUNDS_DIR          = fight_cfg["out_base"]
    ac.FIGHTER_NAMES       = fight_cfg["fighter_names"]
    ac.FIGHTER_SHORT_NAMES = fight_cfg["fighter_short_names"]


def load_round_sam3d(rounds_dir, round_id):
    path = os.path.join(rounds_dir, f"Round{round_id}", "sam3d.json")
    d = json.load(open(path))
    return {tid: {e["frame"]: e for e in d[tid]} for tid in ("0", "1")}


def get_or_build_swap_timeline(rounds_dir, round_id, has_marker):
    if not has_marker:
        return {}, 1
    sidecar = os.path.join(rounds_dir, f"Round{round_id}", "swap_timeline.json")
    if os.path.exists(sidecar):
        d = json.load(open(sidecar))
        return {int(k): v for k, v in d["timeline"].items()}, d["stride"]
    result = ac.get_round_swap_timeline(round_id)
    if result is None:
        raise RuntimeError(f"Cannot build swap timeline for Round{round_id} -- "
                           "original video / SAM3D bbox data missing.")
    timeline, stride = result
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    json.dump({"stride": stride,
               "timeline": {str(k): v for k, v in timeline.items()}},
              open(sidecar, "w"), indent=2)
    return timeline, stride


def lookup_swap(timeline, stride, frame):
    return timeline.get((frame // stride) * stride, False)


# ── geometry helpers ──────────────────────────────────────────────────────────

def cam_coords(entry):
    nc    = np.asarray(entry["normalized_coords"], dtype=np.float64)
    cam_t = np.asarray(entry["pred_cam_t"],        dtype=np.float64)
    return nc + cam_t   # (70, 3) shared camera-space coords


def hip_center(c):
    return (c[cfg.KP70_LEFT_HIP] + c[cfg.KP70_RIGHT_HIP]) / 2.0


def torso_center(c):
    return (c[cfg.KP70_LEFT_SHOULDER] + c[cfg.KP70_RIGHT_SHOULDER]
            + c[cfg.KP70_LEFT_HIP]    + c[cfg.KP70_RIGHT_HIP]) / 4.0


def head_center(c):
    idx = [cfg.KP70_NOSE, cfg.KP70_LEFT_EYE, cfg.KP70_RIGHT_EYE,
           cfg.KP70_LEFT_EAR, cfg.KP70_RIGHT_EAR]
    return c[idx].mean(axis=0)


def shoulder_width(c):
    return float(np.linalg.norm(
        c[cfg.KP70_LEFT_SHOULDER] - c[cfg.KP70_RIGHT_SHOULDER]))


def hand_from_action(action):
    if action.endswith("_left"):
        return cfg.KP70_LEFT_WRIST, 1.0
    if action.endswith("_right"):
        return cfg.KP70_RIGHT_WRIST, 0.0
    # fallback for bare "jab"/"cross" etc. without side suffix
    return cfg.ACTION_HAND_MAP.get(action, cfg.KP70_RIGHT_WRIST), 0.0


def forward_fill(track, frame, last):
    e = track.get(frame)
    if e is not None:
        return e, e
    if last is not None:
        return last, last
    raise RuntimeError(f"No SAM3D entry at or before frame {frame} to fill from")


def build_ref(receiver_track, start_frame):
    """Average hip-center + shoulder-width over REF_FRAMES starting at start_frame."""
    origins, scales, last = [], [], None
    for i in range(REF_FRAMES):
        e, last = forward_fill(receiver_track, start_frame + i, last)
        c = cam_coords(e)
        origins.append(hip_center(c))
        scales.append(shoulder_width(c))
    return np.mean(origins, axis=0), max(float(np.mean(scales)), 1e-4)


def make_frame_row(s_c, r_c, wrist_idx, elbow_idx, shoulder_idx, is_left, state):
    """Per-frame feature row: 13-joint keypoints for both fighters + 10 engineered."""
    s17 = s_c[RAW_JOINTS].reshape(-1)   # 13 joints × 3 = 39
    r17 = r_c[RAW_JOINTS].reshape(-1)   # 39

    wrist    = s_c[wrist_idx]
    elbow    = s_c[elbow_idx]
    shoulder = s_c[shoulder_idx]
    r_torso  = torso_center(r_c)
    r_head   = head_center(r_c)
    s_hip    = hip_center(s_c)
    r_hip    = hip_center(r_c)

    w2t  = float(np.linalg.norm(wrist - r_torso))
    w2h  = float(np.linalg.norm(wrist - r_head))
    h2h  = float(np.linalg.norm(s_hip - r_hip))
    w2e  = float(np.linalg.norm(wrist - elbow))
    e2s  = float(np.linalg.norm(elbow - shoulder))
    full = w2e + e2s
    arm_ext = float(np.linalg.norm(wrist - shoulder)) / full if full > 1e-4 else 0.0
    reach   = w2t / full if full > 1e-4 else REACH_SENTINEL

    pw, pv_w = state.get("pw"), state.get("pv_w")
    ph, pv_h = state.get("ph"), state.get("pv_h")
    wv = 0.0 if pw is None else float(np.linalg.norm(wrist  - pw))
    wa = 0.0 if pv_w is None else (wv - pv_w)
    hv = 0.0 if ph is None else float(np.linalg.norm(r_head - ph))
    ha = 0.0 if pv_h is None else (hv - pv_h)
    state.update(pw=wrist, pv_w=wv, ph=r_head, pv_h=hv)

    eng = [w2t, w2h, wv, wa, hv, ha, arm_ext, h2h, is_left, reach]
    return np.concatenate([s17, r17, eng]).astype(np.float32)


# ── clip extraction ───────────────────────────────────────────────────────────

def extract_clip(clip_entry, sam3d_round, timeline, stride):
    fighter_id   = clip_entry["fighter_id"]
    action       = clip_entry["action"]
    window_end   = clip_entry["window_end"]
    impact_frame = clip_entry.get("impact_frame")

    # anchor = exact contact if marked, else ASFormer's action peak
    if impact_frame is not None:
        anchor     = impact_frame
        has_anchor = 1.0
    else:
        anchor     = window_end
        has_anchor = 0.0

    frame_lo = max(0, anchor - PRE_ANCHOR)
    frame_hi = anchor + POST_ANCHOR

    swap    = lookup_swap(timeline, stride, frame_lo)
    s_tid   = str((1 - fighter_id) if swap else fighter_id)
    r_tid   = str(1 - int(s_tid))
    s_track = sam3d_round[s_tid]
    r_track = sam3d_round[r_tid]

    wrist_idx, is_left = hand_from_action(action)
    elbow_idx, shoulder_idx = cfg.ACTION_ARM_CHAIN[wrist_idx]

    # Build coordinate reference from the PRE-anchor period (stable, pre-contact)
    ref_origin, ref_scale = build_ref(r_track, frame_lo)

    rows, ls, lr, state = [], None, None, {}
    for f in range(frame_lo, frame_hi + 1):
        se, ls = forward_fill(s_track, f, ls)
        re, lr = forward_fill(r_track, f, lr)
        sc = (cam_coords(se) - ref_origin) / ref_scale
        rc = (cam_coords(re) - ref_origin) / ref_scale
        row    = make_frame_row(sc, rc, wrist_idx, elbow_idx, shoulder_idx, is_left, state)
        offset = (f - anchor) / float(POST_ANCHOR)   # -PRE/POST at start, 0 at anchor, +1 at end
        rows.append(np.append(row, [offset, has_anchor]).astype(np.float32))

    return np.stack(rows)   # (T_actual, F)


def pad_and_mask(seq, t_max):
    t, _ = seq.shape
    if t >= t_max:
        return seq[:t_max], np.ones(t_max, dtype=np.float32)
    pad    = np.repeat(seq[-1:], t_max - t, axis=0)
    padded = np.concatenate([seq, pad], axis=0)
    mask   = np.concatenate([np.ones(t, dtype=np.float32),
                              np.zeros(t_max - t, dtype=np.float32)])
    # zero velocity/acc in padded region (repeat-last-row gives non-zero there)
    for col_name in ("wrist_velocity", "wrist_acceleration",
                     "receiver_head_velocity", "receiver_head_acceleration"):
        c = FEATURE_NAMES.index(col_name)
        padded[t:, c] = 0.0
    return padded, mask


# ── per-fight extraction ──────────────────────────────────────────────────────

def extract_fight(fight_name):
    fight_cfg  = fights.get_fight(fight_name)
    configure_for_fight(fight_cfg)
    rounds_dir = fight_cfg["out_base"]
    has_marker = fight_cfg.get("identity_marker") is not None

    manifest = json.load(open(fight_cfg["manifest_path"]))
    clips    = [c for c in manifest["clips"] if c.get("label") in LABEL_TO_Y]
    n_exact  = sum(1 for c in clips if c.get("impact_frame") is not None)
    print(f"[{fight_name}] {len(clips)} labeled clips  "
          f"({n_exact} with exact impact_frame, "
          f"{len(clips) - n_exact} with proxy anchor)")

    rounds_needed  = sorted({c["round"] for c in clips})
    sam3d_by_round = {}
    swap_by_round  = {}
    for r in rounds_needed:
        sam3d_by_round[r] = load_round_sam3d(rounds_dir, r)
        swap_by_round[r]  = get_or_build_swap_timeline(rounds_dir, r, has_marker)

    results, n_skip = [], 0
    for c in clips:
        try:
            tl, st = swap_by_round[c["round"]]
            seq    = extract_clip(c, sam3d_by_round[c["round"]], tl, st)
            results.append((c, seq, fight_name))
        except Exception as e:
            print(f"  [SKIP] {c['clip']}: {e}")
            n_skip += 1

    if n_skip:
        print(f"  WARNING: {n_skip}/{len(clips)} clips skipped "
              f"(missing keypoints or frame out of range)")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", nargs="+", default=["1st_fight"],
                    choices=fights.all_fight_names())
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    raw = []
    for fn in args.fights:
        raw.extend(extract_fight(fn))

    if not raw:
        raise RuntimeError("No clips extracted across all requested fights.")

    t_max = T_FIXED
    print(f"\nFixed window: {t_max} frames  "
          f"(PRE={PRE_ANCHOR} | anchor | POST={POST_ANCHOR})")
    print(f"Feature vector: {len(FEATURE_NAMES)} features per frame")

    X_l, mask_l, y_l = [], [], []
    rounds_l, fids_l, acts_l = [], [], []
    clips_l, wlens_l, fnames_l, groups_l = [], [], [], []

    for c, seq, fn in raw:
        p, m = pad_and_mask(seq, t_max)
        X_l.append(p);  mask_l.append(m)
        y_l.append(LABEL_TO_Y[c["label"]])
        rounds_l.append(c["round"]);  fids_l.append(c["fighter_id"])
        acts_l.append(c["action"]);   clips_l.append(c["clip"])
        wlens_l.append(seq.shape[0]); fnames_l.append(fn)
        groups_l.append(f"{fn}_R{c['round']}")

    X    = np.stack(X_l).astype(np.float32)
    mask = np.stack(mask_l).astype(np.float32)
    y    = np.array(y_l, dtype=np.int64)

    assert X.shape[-1] == len(FEATURE_NAMES), (
        f"feature count mismatch: X has {X.shape[-1]} cols, "
        f"FEATURE_NAMES has {len(FEATURE_NAMES)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,    mask=mask,    y=y,
        round=np.array(rounds_l,  dtype=np.int64),
        fighter_id=np.array(fids_l, dtype=np.int64),
        action=np.array(acts_l,   dtype="<U32"),
        clip_name=np.array(clips_l, dtype="<U64"),
        window_len=np.array(wlens_l, dtype=np.int64),
        feature_names=np.array(FEATURE_NAMES, dtype="<U64"),
        fight=np.array(fnames_l,  dtype="<U32"),
        group=np.array(groups_l,  dtype="<U40"),
    )

    n_exact = sum(1 for c, _, _ in raw if c.get("impact_frame") is not None)
    print(f"\nSaved {X.shape[0]} clips -> {args.out}")
    print(f"  X: {X.shape}   mask: {mask.shape}")
    print(f"  y: impact={int(y.sum())}  not_impact={int((1 - y).sum())}")
    print(f"  exact impact_frame anchor: {n_exact}/{len(raw)} clips")
    print(f"  proxy (window_end) anchor: {len(raw) - n_exact}/{len(raw)} clips")
    print(f"  groups: {dict(zip(*np.unique(np.array(groups_l), return_counts=True)))}")


if __name__ == "__main__":
    main()
