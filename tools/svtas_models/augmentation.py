#!/usr/bin/env python3
"""
Augmentation for the (T, 88) padded keypoint-feature sequences produced by
tools/extract_keypoint_dataset.py. Operates on the already-extracted feature
vector (not raw SAM3D keypoints), so each transform has to respect what each
of the 88 columns actually means -- a generic per-column noise/scale would
silently corrupt the dimensionless ratio columns (arm_extension_ratio,
reach_ratio) and the is_left_hand flag.

Column layout (see extract_keypoint_dataset.FEATURE_NAMES / RAW_JOINTS):
  [0:39]   striker 13 verified MHR70 joints x (x,y,z): head(5)+shoulders(2)+
           elbows(2)+hips(2)+wrists(2), in that order -- raw shoulder-width-
           normalized positions. NOT COCO-17 order past the elbows (see
           config.py's KP70_* fix); wrists are the LAST pair, not slots 9/10.
  [39:78]  receiver, same 13-joint layout
  [78] wrist_to_receiver_torso_dist   -- distance, scale-dependent
  [79] wrist_to_receiver_head_dist    -- distance, scale-dependent
  [80] wrist_velocity                 -- distance/frame, scale-dependent
  [81] wrist_acceleration             -- distance/frame^2, scale-dependent
  [82] receiver_head_velocity         -- distance/frame, scale-dependent
  [83] receiver_head_acceleration     -- distance/frame^2, scale-dependent
  [84] arm_extension_ratio            -- dimensionless ratio, do NOT scale
  [85] striker_hip_to_receiver_hip_dist -- distance, scale-dependent
  [86] is_left_hand                   -- binary flag, flip on mirror only
  [87] reach_ratio                    -- dimensionless ratio, do NOT scale

Four transforms, each independently toggleable:
  - mirror:    exact, geometrically correct. A left-right flip is an isometry,
               so every distance/velocity/acceleration/ratio feature is
               UNCHANGED by it -- only the raw per-joint x-coordinates (sign
               flip + left/right joint swap) and is_left_hand (inverted) need
               to actually change. This is the one "free lunch" augmentation
               here: zero approximation error.
  - time_warp: resample the real (unpadded) portion of the sequence to a
               random rate in [0.85, 1.15x] via linear interpolation, then
               re-pad with extract_keypoint_dataset.pad_and_mask so the
               result is shaped identically to an unaugmented sample.
               Simulates faster/slower strikes than were actually recorded.
  - scale_jitter: multiply only the scale-DEPENDENT columns (raw keypoints +
               the 7 distance/velocity/acceleration columns) by one random
               factor in [0.95, 1.05], leaving the 2 dimensionless ratios and
               the hand flag untouched. Approximates noise in the per-clip
               shoulder-width normalization itself.
  - jitter:    small Gaussian noise on the scale-dependent columns only
               (sigma relative to each column's own std across the training
               fold, passed in). A pragmatic approximation, not re-derived
               from noisy raw keypoints -- documented as such; doesn't touch
               the ratio/flag columns for the same reason scale_jitter
               doesn't.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"\..")
from extract_keypoint_dataset import FEATURE_NAMES, pad_and_mask  # noqa: E402

N_JOINTS = 13  # head(5) + shoulders(2) + elbows(2) + hips(2) + wrists(2), see RAW_JOINTS
N_RAW = N_JOINTS * 3 * 2  # striker(39) + receiver(39) raw keypoint columns = 78
COL_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

SCALE_DEP_COLS = list(range(N_RAW)) + [
    COL_IDX["wrist_to_receiver_torso_dist"], COL_IDX["wrist_to_receiver_head_dist"],
    COL_IDX["wrist_velocity"], COL_IDX["wrist_acceleration"],
    COL_IDX["receiver_head_velocity"], COL_IDX["receiver_head_acceleration"],
    COL_IDX["striker_hip_to_receiver_hip_dist"],
]
RATIO_COLS = [COL_IDX["arm_extension_ratio"], COL_IDX["reach_ratio"]]
HAND_COL = COL_IDX["is_left_hand"]

# Left<->right joint swap, by POSITION in RAW_JOINTS = [0,1,2,3,4,5,6,7,8,9,10,41,62]
# i.e. nose(0)->self, L/R eye(1,2), L/R ear(3,4), L/R shoulder(5,6), L/R elbow(7,8),
# L/R hip(9,10), then R/L wrist at positions 11,12 (RAW_JOINTS[11]=41=R wrist,
# RAW_JOINTS[12]=62=L wrist) -- NOT positions 9/10 the way COCO would have it.
_LR_SWAP = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7,
            9: 10, 10: 9, 11: 12, 12: 11}
_SWAP_PERM = np.array([_LR_SWAP[j] for j in range(N_JOINTS)])


def _mirror_block(block_n):
    """block_n: (..., 39) = 13 joints x (x,y,z). Negate x, swap L/R joints."""
    pts = block_n.reshape(*block_n.shape[:-1], N_JOINTS, 3).copy()
    pts[..., 0] *= -1.0
    pts = pts[..., _SWAP_PERM, :]
    return pts.reshape(*block_n.shape[:-1], N_JOINTS * 3)


def mirror(seq, mask):
    """Exact left-right flip. seq: (T, 88). Returns (seq', mask) unchanged shape."""
    out = seq.copy()
    out[:, 0:39] = _mirror_block(seq[:, 0:39])
    out[:, 39:78] = _mirror_block(seq[:, 39:78])
    out[:, HAND_COL] = 1.0 - seq[:, HAND_COL]
    return out, mask


def time_warp(seq, mask, t_max, rate_range=(0.85, 1.15), rng=None):
    """Resample the real (mask==1) portion at a random speed, then re-pad to
    t_max frames using the same repeat-last-frame convention as training."""
    rng = rng or np.random
    t_actual = int(mask.sum())
    if t_actual < 4:
        return seq, mask  # too short to meaningfully warp
    real = seq[:t_actual]
    rate = rng.uniform(*rate_range)
    new_len = max(2, int(round(t_actual * rate)))
    old_idx = np.linspace(0, t_actual - 1, num=t_actual)
    new_idx = np.linspace(0, t_actual - 1, num=new_len)
    warped = np.stack([np.interp(new_idx, old_idx, real[:, c]) for c in range(real.shape[1])], axis=1)
    warped[:, HAND_COL] = real[0, HAND_COL]  # binary flag must not get interpolated to a fraction
    padded, new_mask = pad_and_mask(warped.astype(np.float32), t_max)
    return padded, new_mask


def scale_jitter(seq, mask, scale_range=(0.95, 1.05), rng=None):
    rng = rng or np.random
    out = seq.copy()
    factor = rng.uniform(*scale_range)
    out[:, SCALE_DEP_COLS] *= factor
    return out, mask


def gaussian_jitter(seq, mask, col_std, sigma_frac=0.03, rng=None):
    """col_std: (88,) per-column std computed over the TRAINING fold only
    (passed in, not recomputed per-sample, so augmentation strength is
    consistent and doesn't leak test-fold statistics)."""
    rng = rng or np.random
    out = seq.copy()
    t_actual = int(mask.sum())
    noise = rng.normal(0.0, sigma_frac, size=(t_actual, len(SCALE_DEP_COLS))) * col_std[SCALE_DEP_COLS]
    out[:t_actual][:, SCALE_DEP_COLS] += noise
    return out, mask


def augment_batch(X, mask, t_max, col_std, rng=None,
                   p_mirror=0.5, p_time_warp=0.5, p_scale=0.5, p_jitter=0.5):
    """Apply the enabled transforms independently (each with its own
    probability) to every sample in a batch. Returns new (X, mask) arrays of
    the same shape -- safe to call once per training epoch on the training
    fold only (never on the held-out test fold)."""
    rng = rng or np.random
    X_out = X.copy()
    mask_out = mask.copy()
    for i in range(X.shape[0]):
        seq, m = X_out[i], mask_out[i]
        if rng.random() < p_mirror:
            seq, m = mirror(seq, m)
        if rng.random() < p_time_warp:
            seq, m = time_warp(seq, m, t_max, rng=rng)
        if rng.random() < p_scale:
            seq, m = scale_jitter(seq, m, rng=rng)
        if rng.random() < p_jitter:
            seq, m = gaussian_jitter(seq, m, col_std, rng=rng)
        X_out[i], mask_out[i] = seq, m
    return X_out, mask_out
