#!/usr/bin/env python3
"""
Extract a 3D-keypoint feature-sequence dataset for impact / not_impact
classification, from the hand-labeled manifest + per-round SAM3D keypoints.

For every labeled clip in manifest.json:
  - resolve which SAM3D track_id ("0"/"1") is the striker vs receiver at
    this clip's frames, correcting for CUTIE identity swaps during clinches
    (reuses the green-trim swap timeline already validated in
    tools/annotate_clips.py for the annotation GUI's name overlay)
  - reconstruct camera-space 3D coordinates as
    normalized_coords + pred_cam_t -- raw normalized_coords alone do NOT
    preserve cross-person relative geometry (each fighter's normalized_coords
    is independently root-relative; verified empirically: correlation with
    pixel-space distance was ~0). normalized_coords + pred_cam_t reconstructs
    a properly shared camera-space position (verified: XY-plane correlation
    with pixel-space distance 0.65-0.90 across all 4 rounds).
  - anchor + scale into a shared frame relative to the RECEIVER's body at
    window_start, so striker keypoints are expressed in
    receiver-body-relative units (preserves the "how close is the wrist to
    the receiver" signal that per-person-independent normalization would
    destroy)
  - compute 9 engineered physics features (wrist-to-torso/head distance,
    wrist velocity/acceleration, receiver head velocity/acceleration,
    arm-extension ratio, hip-to-hip distance, hand flag) + a reduced
    COCO-17 raw-keypoint subset for both fighters, per frame
  - pad every clip to a fixed length (repeating the last real frame, not
    zeros) and export a mask marking which frames are real

Multiple fights can be combined into one dataset (--fights a b c): each
clip is tagged with its source fight, and the leave-one-round-out grouping
key becomes "<fight>_R<round>" (not bare round number) so a fold can never
mix rounds from two different fights under the same group -- two fights'
"Round3" are unrelated videos and must not be treated as the same held-out
unit.

Usage:
  python tools/extract_keypoint_dataset.py
  python tools/extract_keypoint_dataset.py --fights lillyella_vs_zoe cameron_vs_liam
  python tools/extract_keypoint_dataset.py --fights cameron_vs_liam --out outputs/keypoint_dataset/cameron_vs_liam.npz
  python tools/extract_keypoint_dataset.py --max-len 40
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_clips as ac  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
import fights  # noqa: E402

DEFAULT_FIGHTS = ["lillyella_vs_zoe"]
DEFAULT_OUT = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_dataset\lillyella_vs_zoe.npz"
COMBINED_OUT = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_dataset\combined.npz"

PAD_BEFORE = 8   # frames before window_start (matches prepare_lillyella_dataset.py)
PAD_END = 20     # frames after window_end

RAW_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 41, 62]
# Verified joints only: head(0-4) + shoulders(5,6) + elbows(7,8) + hips(9,10)
# + wrists(41 right, 62 left) -- see config.py's KP70_* comment for how these
# were verified. The old "RAW_JOINTS = range(17)" silently included indices 9/10
# (actually hips, mislabeled as wrists) and 11-16 (unverified -- this
# skeleton is NOT COCO-17 ordered past the elbows, so plain range(17) was
# wrong). 13 joints instead of 17, no knee/ankle (not located in this format).

LABEL_TO_Y = {"impact": 1, "not_impact": 0}

REF_FRAMES = 5     # average the reference scale/origin over this many frames
                   # (instead of a single window_start frame) for stability
                   # against one noisy pose estimate
REACH_SENTINEL = 5.0  # reach_ratio value when the arm length is degenerate (~0)

FEATURE_NAMES = (
    [f"striker_kp{j}_{ax}" for j in RAW_JOINTS for ax in ("x", "y", "z")]
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
    ]
)


# ── SAM3D loading ────────────────────────────────────────────────────────────

def configure_for_fight(fight_cfg):
    """Push this fight's paths/marker into tools.annotate_clips' module
    globals so its get_round_swap_timeline() (the only place that knows how
    to scan a round video for the green-trim identity-swap signal) operates
    on the right fight. Mirrors what annotate_clips.main() does for the GUI."""
    ac.VIDEO_FOLDER     = fight_cfg["video_folder"]
    ac.IDENTITY_MARKER  = fight_cfg.get("identity_marker")
    ac.ROUNDS_DIR       = fight_cfg["out_base"]
    ac.FIGHTER_NAMES    = fight_cfg["fighter_names"]
    ac.FIGHTER_SHORT_NAMES = fight_cfg["fighter_short_names"]


def load_round_sam3d(rounds_dir, round_id):
    """frame -> entry dict, per track_id ('0'/'1'), for one round's sam3d.json."""
    path = os.path.join(rounds_dir, f"Round{round_id}", "sam3d.json")
    d = json.load(open(path))
    return {tid: {e["frame"]: e for e in d[tid]} for tid in ("0", "1")}


def get_or_build_swap_timeline(rounds_dir, round_id, has_marker):
    """Load the cached swap-timeline sidecar if present, else compute it via
    tools.annotate_clips.get_round_swap_timeline (which needs the original
    full-round video) and persist it so future runs don't need that video.

    Fights with no identity_marker configured (most of them -- see
    dataset/fights.py) have no swap signal to detect in the first place;
    for those we skip the whole sidecar/video machinery and return an
    empty timeline, which lookup_swap_cached() naturally reads as
    "never swapped" (trusting the raw tracker IDs, same as the GUI does)."""
    if not has_marker:
        return {}, 1

    sidecar = os.path.join(rounds_dir, f"Round{round_id}", "swap_timeline.json")
    if os.path.exists(sidecar):
        d = json.load(open(sidecar))
        timeline = {int(k): v for k, v in d["timeline"].items()}
        return timeline, d["stride"]

    print(f"  (no cached swap_timeline for Round{round_id}, computing from video, ~15s)")
    result = ac.get_round_swap_timeline(round_id)
    if result is None:
        raise RuntimeError(
            f"Could not build swap timeline for Round{round_id} -- original "
            f"round video / sam3d bbox data missing. Identity-swap correction "
            f"cannot be skipped silently (it would corrupt receiver-relative "
            f"features for any round with a real swap)."
        )
    timeline, stride = result
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    json.dump({"stride": stride, "timeline": {str(k): v for k, v in timeline.items()}},
              open(sidecar, "w"), indent=2)
    return timeline, stride


def lookup_swap_cached(timeline, stride, frame):
    nearest = (frame // stride) * stride
    return timeline.get(nearest, False)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def cam_space_coords(entry):
    """normalized_coords (70,3) + pred_cam_t broadcast -> shared camera-space
    coords. See module docstring: raw normalized_coords alone do NOT
    preserve cross-person relative geometry; this reconstruction does."""
    nc = np.asarray(entry["normalized_coords"], dtype=np.float64)
    cam_t = np.asarray(entry["pred_cam_t"], dtype=np.float64)
    return nc + cam_t  # (70, 3)


def hip_center(coords70):
    return (coords70[cfg.KP70_LEFT_HIP] + coords70[cfg.KP70_RIGHT_HIP]) / 2.0


def shoulder_width(coords70):
    return float(np.linalg.norm(
        coords70[cfg.KP70_LEFT_SHOULDER] - coords70[cfg.KP70_RIGHT_SHOULDER]))


def torso_center(coords70):
    return (coords70[cfg.KP70_LEFT_SHOULDER] + coords70[cfg.KP70_RIGHT_SHOULDER]
            + coords70[cfg.KP70_LEFT_HIP] + coords70[cfg.KP70_RIGHT_HIP]) / 4.0


def head_center(coords70):
    idx = [cfg.KP70_NOSE, cfg.KP70_LEFT_EYE, cfg.KP70_RIGHT_EYE,
           cfg.KP70_LEFT_EAR, cfg.KP70_RIGHT_EAR]
    return coords70[idx].mean(axis=0)


def hand_from_action(action):
    """'jab_left' -> (LEFT_WRIST, is_left=1.0); 'cross_right' -> (RIGHT_WRIST, 0.0)."""
    if action.endswith("_left"):
        return cfg.KP70_LEFT_WRIST, 1.0
    if action.endswith("_right"):
        return cfg.KP70_RIGHT_WRIST, 0.0
    raise ValueError(f"Cannot derive hand from action name: {action!r}")


# ── Per-clip extraction ──────────────────────────────────────────────────────

def forward_fill_entry(track_frames, frame, last_valid):
    """Return (entry, new_last_valid). Carries the last seen entry forward
    over occasional per-frame tracking dropouts so clip length stays fixed."""
    e = track_frames.get(frame)
    if e is not None:
        return e, e
    if last_valid is not None:
        return last_valid, last_valid
    raise RuntimeError(f"No SAM3D entry at or before frame {frame} to forward-fill from")


def build_reference_frame(receiver_track, window_start, ref_frames=REF_FRAMES):
    """Average the receiver's hip-center + shoulder-width over the first
    `ref_frames` frames starting at window_start, instead of a single frame.
    Reduces sensitivity to one noisy/foreshortened pose estimate at the
    clip's start, which otherwise directly sets the scale for every distance
    feature in the whole clip."""
    origins, scales = [], []
    last_valid = None
    for i in range(ref_frames):
        entry, last_valid = forward_fill_entry(receiver_track, window_start + i, last_valid)
        coords = cam_space_coords(entry)
        origins.append(hip_center(coords))
        scales.append(shoulder_width(coords))
    ref_origin = np.mean(origins, axis=0)
    ref_scale = max(float(np.mean(scales)), 1e-4)
    return ref_origin, ref_scale


def compute_frame_row(s_coords, r_coords, wrist_idx, elbow_idx, shoulder_idx, is_left, state):
    """Build one frame's feature row (raw verified-joint keypoints + engineered
    features) from already shared-frame-transformed striker/receiver
    coords. `state` is a dict carrying prev_wrist/prev_wrist_vel/
    prev_head/prev_head_vel across calls (mutated in place) so velocity/
    acceleration features work; pass a fresh {} at the start of each clip."""
    striker17 = s_coords[RAW_JOINTS].reshape(-1)   # 13 joints * 3 = 39
    receiver17 = r_coords[RAW_JOINTS].reshape(-1)  # 39

    wrist = s_coords[wrist_idx]
    elbow = s_coords[elbow_idx]
    shoulder = s_coords[shoulder_idx]
    r_torso = torso_center(r_coords)
    r_head = head_center(r_coords)
    s_hip = hip_center(s_coords)
    r_hip = hip_center(r_coords)

    wrist_to_torso = float(np.linalg.norm(wrist - r_torso))
    wrist_to_head = float(np.linalg.norm(wrist - r_head))
    hip_to_hip = float(np.linalg.norm(s_hip - r_hip))

    w2s = float(np.linalg.norm(wrist - shoulder))
    w2e = float(np.linalg.norm(wrist - elbow))
    e2s = float(np.linalg.norm(elbow - shoulder))
    full_arm = w2e + e2s
    arm_ext = w2s / full_arm if full_arm > 1e-4 else 0.0

    # Self-relative "reach" feature: wrist-to-torso distance measured in
    # units of the STRIKER'S OWN current arm length, instead of the
    # receiver's shoulder width. Answers "is the target within one arm's
    # length" directly, which should generalize better across fighter pairs
    # of different relative body proportions than a receiver-only scale.
    reach_ratio = wrist_to_torso / full_arm if full_arm > 1e-4 else REACH_SENTINEL

    prev_wrist = state.get("prev_wrist")
    prev_wrist_vel = state.get("prev_wrist_vel")
    prev_head = state.get("prev_head")
    prev_head_vel = state.get("prev_head_vel")

    wrist_vel = 0.0 if prev_wrist is None else float(np.linalg.norm(wrist - prev_wrist))
    wrist_acc = 0.0 if prev_wrist_vel is None else (wrist_vel - prev_wrist_vel)
    head_vel = 0.0 if prev_head is None else float(np.linalg.norm(r_head - prev_head))
    head_acc = 0.0 if prev_head_vel is None else (head_vel - prev_head_vel)

    state["prev_wrist"], state["prev_wrist_vel"] = wrist, wrist_vel
    state["prev_head"], state["prev_head_vel"] = r_head, head_vel

    engineered = [
        wrist_to_torso, wrist_to_head, wrist_vel, wrist_acc,
        head_vel, head_acc, arm_ext, hip_to_hip, is_left, reach_ratio,
    ]
    return np.concatenate([striker17, receiver17, engineered]).astype(np.float32)


def extract_clip(clip_entry, sam3d_round, swap_timeline, swap_stride):
    fighter_id = clip_entry["fighter_id"]
    action = clip_entry["action"]
    window_start = clip_entry["window_start"]
    window_end = clip_entry["window_end"]

    frame_lo = max(0, window_start - PAD_BEFORE)
    frame_hi = window_end + PAD_END
    frame_list = list(range(frame_lo, frame_hi + 1))

    swap = lookup_swap_cached(swap_timeline, swap_stride, frame_lo)
    striker_tid = str((1 - fighter_id) if swap else fighter_id)
    receiver_tid = str(1 - int(striker_tid))

    striker_track = sam3d_round[striker_tid]
    receiver_track = sam3d_round[receiver_tid]

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

    return np.stack(rows)  # (T_actual, F)


def pad_and_mask(seq, t_max):
    t_actual, f = seq.shape
    if t_actual >= t_max:
        return seq[:t_max], np.ones(t_max, dtype=np.float32)
    pad_rows = np.repeat(seq[-1:], t_max - t_actual, axis=0)
    padded = np.concatenate([seq, pad_rows], axis=0)
    mask = np.concatenate([np.ones(t_actual, dtype=np.float32),
                            np.zeros(t_max - t_actual, dtype=np.float32)])
    # Defensive: zero out velocity/acceleration-style features in the padded
    # region regardless of what the repeat-last-row naturally gives (should
    # already be 0, this just guarantees it).
    vel_acc_cols = [
        FEATURE_NAMES.index("wrist_velocity"),
        FEATURE_NAMES.index("wrist_acceleration"),
        FEATURE_NAMES.index("receiver_head_velocity"),
        FEATURE_NAMES.index("receiver_head_acceleration"),
    ]
    for c in vel_acc_cols:
        padded[t_actual:, c] = 0.0
    return padded, mask


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_fight(fight_name):
    """Returns a list of (clip_entry, seq, fight_name) for every labeled clip
    in this fight's manifest that could be extracted."""
    fight_cfg = fights.get_fight(fight_name)
    configure_for_fight(fight_cfg)
    rounds_dir = fight_cfg["out_base"]
    has_marker = fight_cfg.get("identity_marker") is not None

    manifest = json.load(open(fight_cfg["manifest_path"]))
    clips = [c for c in manifest["clips"] if c["label"] in LABEL_TO_Y]
    n_skipped_unlabeled = len(manifest["clips"]) - len(clips)
    print(f"[{fight_name}] {len(clips)} labeled clips ({n_skipped_unlabeled} unlabeled skipped)")

    rounds_needed = sorted({c["round"] for c in clips})
    print(f"[{fight_name}] Loading SAM3D + swap timelines for rounds {rounds_needed}...")
    sam3d_by_round = {}
    swap_by_round = {}
    for r in rounds_needed:
        sam3d_by_round[r] = load_round_sam3d(rounds_dir, r)
        swap_by_round[r] = get_or_build_swap_timeline(rounds_dir, r, has_marker)

    results = []
    skipped = 0
    for i, c in enumerate(clips):
        try:
            timeline, stride = swap_by_round[c["round"]]
            seq = extract_clip(c, sam3d_by_round[c["round"]], timeline, stride)
            results.append((c, seq, fight_name))
        except Exception as e:
            print(f"  [SKIP] {c['clip']}: {e}")
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(clips)} extracted")

    if skipped:
        print(f"*** [{fight_name}] WARNING: {skipped}/{len(clips)} clips skipped "
              f"(usually missing real SAM3D keypoints for one fighter in that round) ***")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", nargs="+", default=DEFAULT_FIGHTS,
                     choices=fights.all_fight_names(),
                     help="one or more registered fights to combine into one dataset")
    ap.add_argument("--out", default=None,
                     help="default: <fight>.npz for a single fight, else combined.npz")
    ap.add_argument("--max-len", type=int, default=None,
                     help="override the fixed sequence length (default: dataset's observed max)")
    args = ap.parse_args()
    out_path = args.out or (DEFAULT_OUT if len(args.fights) == 1 else COMBINED_OUT)

    raw_seqs = []
    for fight_name in args.fights:
        raw_seqs.extend(extract_fight(fight_name))

    if not raw_seqs:
        raise RuntimeError("No clips extracted across all requested fights -- nothing to save.")

    t_max = args.max_len or max(seq.shape[0] for _, seq, _ in raw_seqs)
    print(f"\nT_MAX = {t_max}  (observed range: "
          f"{min(seq.shape[0] for _, seq, _ in raw_seqs)}-{max(seq.shape[0] for _, seq, _ in raw_seqs)})")

    X, mask, y, rounds, fighter_ids, actions, clip_names, window_lens, fight_names, groups = (
        [], [], [], [], [], [], [], [], [], []
    )
    for c, seq, fight_name in raw_seqs:
        padded, m = pad_and_mask(seq, t_max)
        X.append(padded)
        mask.append(m)
        y.append(LABEL_TO_Y[c["label"]])
        rounds.append(c["round"])
        fighter_ids.append(c["fighter_id"])
        actions.append(c["action"])
        clip_names.append(c["clip"])
        window_lens.append(seq.shape[0])
        fight_names.append(fight_name)
        groups.append(f"{fight_name}_R{c['round']}")

    X = np.stack(X).astype(np.float32)
    mask = np.stack(mask).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    rounds = np.array(rounds, dtype=np.int64)
    fighter_ids = np.array(fighter_ids, dtype=np.int64)
    actions = np.array(actions, dtype="<U32")
    clip_names = np.array(clip_names, dtype="<U64")
    window_lens = np.array(window_lens, dtype=np.int64)
    fight_names = np.array(fight_names, dtype="<U32")
    groups = np.array(groups, dtype="<U40")
    feature_names = np.array(FEATURE_NAMES, dtype="<U64")

    assert X.shape[-1] == len(FEATURE_NAMES), \
        f"feature count mismatch: X has {X.shape[-1]} cols, FEATURE_NAMES has {len(FEATURE_NAMES)}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X, mask=mask, y=y, round=rounds, fighter_id=fighter_ids,
        action=actions, clip_name=clip_names, window_len=window_lens,
        feature_names=feature_names, fight=fight_names, group=groups,
    )

    print(f"\nSaved {X.shape[0]} clips -> {out_path}")
    print(f"  X: {X.shape}   mask: {mask.shape}   F={X.shape[-1]} features")
    print(f"  y: impact={int(y.sum())}  not_impact={int((1 - y).sum())}")
    print(f"  groups: {dict(zip(*np.unique(groups, return_counts=True)))}")


if __name__ == "__main__":
    main()
