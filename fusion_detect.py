"""
Multi-signal impact fusion detector.
=====================================

Combines:
  1. action JSON      (candidate frames, ~97% recall potential)
  2. contact_events   (per-frame mesh contact prob + distance)
  3. world_coords     (per-fighter 3D wrist-to-receiver distance)
  4. wrist velocity   (2D & 3D motion)
  5. audio onset      (glove-impact transient)

Scoring: each candidate action gets a weighted sum of normalised features.
NMS over time with cooldown, then threshold.
Evaluated against 31 GT impacts.
"""

import json
import os
import math
import numpy as np
import cv2
import librosa
from itertools import product

# ── Paths ──────────────────────────────────────────────────────────────────────
FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
VIDEO_IN   = os.path.join(FOLDER, "3.mp4")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
OUT_JSON   = os.path.join(FOLDER, "3_fusion_detections.json")
OUT_VIDEO  = os.path.join(FOLDER, "3_fusion_impacts.mp4")

FPS = 24.995

GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]

# 70-joint keypoint indices (MHR convention; first 17 are COCO)
KP_NOSE, KP_LEAR, KP_REAR = 0, 3, 4
KP_LSH, KP_RSH = 5, 6
KP_LWR, KP_RWR = 9, 10
KP_LEL, KP_REL = 7, 8
HEAD_KPS = [KP_NOSE, 1, 2, KP_LEAR, KP_REAR]
WRIST_KPS = [KP_LWR, KP_RWR]
HAND_KPS  = [KP_LWR, KP_RWR, KP_LEL, KP_REL]  # treat elbow as fallback


def parse_ts_to_frame(ts, fps=FPS):
    p = ts.split(":")
    if len(p) == 2: s = int(p[0]) + int(p[1]) / fps
    else: s = int(p[0]) * 60 + int(p[1]) + int(p[2]) / fps
    return int(round(s * fps))


GT_FRAMES = [parse_ts_to_frame(t) for t in GT_TIMESTAMPS]


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    with open(SAM3D_JSON) as f:
        sam3d = json.load(f)
    with open(ACT_JSON) as f:
        actions = json.load(f)["actions"]
    contact_events = sam3d.get("contact_events", [])
    # frame -> {track_id -> entry}
    frame_persons = {}
    for tid_key in ("0", "1"):
        if tid_key not in sam3d:
            continue
        for entry in sam3d[tid_key]:
            fn = entry["frame"]
            tid = int(entry["track_id"])
            frame_persons.setdefault(fn, {})[tid] = entry
    return sam3d, actions, contact_events, frame_persons


# ── 3D wrist distance to opponent's head (per-fighter) ────────────────────────
def head_center(entry):
    wc = entry.get("world_coords") or entry.get("normalized_coords")
    if wc is None: return None
    pts = []
    for k in HEAD_KPS:
        if k < len(wc):
            pts.append(wc[k])
    if not pts: return None
    return np.mean(pts, axis=0)


def torso_center(entry):
    wc = entry.get("world_coords") or entry.get("normalized_coords")
    if wc is None: return None
    pts = []
    for k in (5, 6, 11, 12):
        if k < len(wc):
            pts.append(wc[k])
    if not pts: return None
    return np.mean(pts, axis=0)


def hand_pos(entry, k):
    wc = entry.get("world_coords") or entry.get("normalized_coords")
    if wc is None or k >= len(wc): return None
    return np.array(wc[k], dtype=float)


def min_hand_to_target(striker_entry, receiver_entry):
    """Returns min distance from striker hand (wrist or elbow) to receiver head OR torso.
    Per-fighter coords are used independently then offset-aligned via head Z-mean."""
    if striker_entry is None or receiver_entry is None:
        return None
    rh = head_center(receiver_entry)
    rt = torso_center(receiver_entry)
    if rh is None or rt is None: return None
    best = float("inf")
    for k in HAND_KPS:
        hp = hand_pos(striker_entry, k)
        if hp is None: continue
        # try XY-only distance (ignore Z to avoid cross-person Z error)
        d_h_xy = np.linalg.norm(hp[:2] - rh[:2])
        d_t_xy = np.linalg.norm(hp[:2] - rt[:2])
        best = min(best, d_h_xy, d_t_xy)
    return best if best < float("inf") else None


def wrist_velocity_3d(frame_persons, fn, striker_id, window=4):
    """Average wrist speed (3D) over a window."""
    e_now  = frame_persons.get(fn, {}).get(striker_id)
    e_prev = frame_persons.get(fn - window, {}).get(striker_id)
    if not e_now or not e_prev:
        return 0.0
    best = 0.0
    for k in HAND_KPS:
        p_now  = hand_pos(e_now,  k)
        p_prev = hand_pos(e_prev, k)
        if p_now is None or p_prev is None: continue
        v = np.linalg.norm(p_now - p_prev) / window
        best = max(best, v)
    return best


def receiver_head_decel(frame_persons, fn, receiver_id, win=4):
    """Detect receiver head acceleration spike around contact frame.
    Measures the change in head velocity (decel/accel) at frame fn.
    Returns a normalized score 0..1.
    """
    h_prev = h_now = h_next = None
    e_prev = frame_persons.get(fn - win, {}).get(receiver_id)
    e_now  = frame_persons.get(fn,       {}).get(receiver_id)
    e_next = frame_persons.get(fn + win, {}).get(receiver_id)
    if not e_prev or not e_now or not e_next:
        return 0.0
    h_prev = head_center(e_prev)
    h_now  = head_center(e_now)
    h_next = head_center(e_next)
    if h_prev is None or h_now is None or h_next is None:
        return 0.0
    v1 = (h_now  - h_prev) / win
    v2 = (h_next - h_now)  / win
    accel = np.linalg.norm(v2 - v1)
    # Use a stricter threshold so this signal is actually discriminative
    # Calibrated: most quiet frames have accel < 0.05, real impacts ~0.15+
    return min(1.0, max(0.0, (accel - 0.05) / 0.25))


def wrist_decel(frame_persons, fn, striker_id, win=3):
    """Striker wrist deceleration spike (hand decel at impact)."""
    e_prev = frame_persons.get(fn - win, {}).get(striker_id)
    e_now  = frame_persons.get(fn,       {}).get(striker_id)
    e_next = frame_persons.get(fn + win, {}).get(striker_id)
    if not e_prev or not e_now or not e_next:
        return 0.0
    best = 0.0
    for k in WRIST_KPS:
        p_prev = hand_pos(e_prev, k)
        p_now  = hand_pos(e_now,  k)
        p_next = hand_pos(e_next, k)
        if p_prev is None or p_now is None or p_next is None: continue
        v1 = np.linalg.norm(p_now  - p_prev) / win
        v2 = np.linalg.norm(p_next - p_now)  / win
        # decel: v1 large, v2 small (hand decelerates)
        decel = max(0.0, v1 - v2)
        best = max(best, decel)
    # Calibrated: quiet frames < 0.02, real decel events 0.08+
    return min(1.0, max(0.0, (best - 0.02) / 0.15))


# ── Audio onset detection ─────────────────────────────────────────────────────
def extract_audio_onsets(video_path, fps):
    """Run a band-limited spectral-flux onset detector over the video audio.
    Tuned for boxing-glove impact transients (sharp, broadband, mostly 1-6 kHz).
    """
    print("Loading audio...")
    y, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
    print(f"  audio: {len(y)/sr:.1f}s @ {sr}Hz")

    # Pre-emphasis to boost the impact transient
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Two bands: low-mid (1-3kHz) for body thuds, high-mid (3-6kHz) for glove slaps
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_mid = (freqs >= 1000) & (freqs <= 6000)
    S_band = S[band_mid, :]

    # Spectral flux (positive change)
    flux = np.maximum(0, np.diff(S_band, axis=1)).sum(axis=0).astype(np.float64)
    flux = np.concatenate([[0.0], flux])

    # Adaptive normalisation (local median)
    flux_norm = flux - np.median(flux)
    flux_norm = np.maximum(0, flux_norm)

    times = librosa.frames_to_time(np.arange(len(flux_norm)), sr=sr, hop_length=256)

    # Stricter peak picking: looking for ~50-100 impacts total over 3 min
    onsets = librosa.util.peak_pick(
        flux_norm, pre_max=20, post_max=20, pre_avg=30, post_avg=30,
        delta=float(flux_norm.std() * 1.2), wait=15)

    onset_times = times[onsets]
    onset_frames = (onset_times * fps).astype(int)
    onset_strengths = flux_norm[onsets] / (flux_norm.max() + 1e-9)
    # Keep only strong onsets
    keep = onset_strengths > 0.15
    onset_frames = onset_frames[keep]
    onset_strengths = onset_strengths[keep]
    print(f"  detected {len(onset_frames)} audio onsets (strong)")
    return list(zip(onset_frames.tolist(), onset_strengths.tolist()))


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(det_frames, gt_frames=GT_FRAMES, tol=30):
    matched_gt = set()
    matched_det = set()
    pairs = []
    for di, df in enumerate(det_frames):
        best, best_d = None, tol + 1
        for gi, gf in enumerate(gt_frames):
            if gi in matched_gt: continue
            d = abs(df - gf)
            if d < best_d:
                best_d, best = d, gi
        if best is not None:
            matched_gt.add(best)
            matched_det.add(di)
            pairs.append((di, best, best_d))
    tp = len(matched_gt)
    fp = len(det_frames) - len(matched_det)
    fn = len(gt_frames) - tp
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, matched_gt=matched_gt, pairs=pairs)


# ── Candidate scoring ─────────────────────────────────────────────────────────
def build_candidates(actions, contact_events, frame_persons, onsets,
                     w_act, w_ce, w_dist, w_vel, w_audio, w_sig, w_decel, w_headacc,
                     ce_window=8, audio_window=10, vel_window=4):
    onsets_by_frame = {}
    for f, s in onsets:
        onsets_by_frame.setdefault(f, 0.0)
        onsets_by_frame[f] = max(onsets_by_frame[f], s)

    def score_at(fn, sid, base_act_conf, is_sig, action_obj):
        rid = 1 - sid
        s_act = base_act_conf

        # contact_events near this frame
        s_ce = 0.0
        best_dist_ce = None
        best_region = None
        for ce in contact_events:
            if abs(ce["frame"] - fn) <= ce_window and ce["striker_id"] == sid:
                region_w = 1.0 if ce["contact_region"] in ("head", "torso") else 0.5
                ev_score = ce["contact_prob"] * region_w
                if ev_score > s_ce:
                    s_ce = ev_score
                    best_dist_ce = ce["contact_3d_distance_m"]
                    best_region = ce["contact_region"]

        # 3D XY hand-to-target distance
        s_dist = 0.0
        striker_entry  = frame_persons.get(fn, {}).get(sid)
        receiver_entry = frame_persons.get(fn, {}).get(rid)
        d = min_hand_to_target(striker_entry, receiver_entry)
        if d is not None:
            s_dist = max(0.0, 1.0 - d / 0.5)

        # wrist 3D motion
        v3 = wrist_velocity_3d(frame_persons, fn, sid, window=vel_window)
        s_vel = min(1.0, v3 / 0.10)

        # audio onset near this frame
        s_audio = 0.0
        for df in range(-audio_window, audio_window + 1):
            if (fn + df) in onsets_by_frame:
                s_audio = max(s_audio, onsets_by_frame[fn + df] *
                              (1.0 - abs(df) / (audio_window + 1)))

        # is_significant
        s_sig = 1.0 if is_sig else 0.0

        # wrist decel + receiver head accel (biomechanical signal)
        s_decel   = wrist_decel(frame_persons, fn, sid)
        s_headacc = receiver_head_decel(frame_persons, fn, rid)

        score = (w_act * s_act + w_ce * s_ce + w_dist * s_dist +
                 w_vel * s_vel + w_audio * s_audio + w_sig * s_sig +
                 w_decel * s_decel + w_headacc * s_headacc)
        return dict(
            frame=fn, score=score, action=action_obj, sid=sid,
            s_act=s_act, s_ce=s_ce, s_dist=s_dist,
            s_vel=s_vel, s_audio=s_audio, s_sig=s_sig,
            s_decel=s_decel, s_headacc=s_headacc,
            ce_dist=best_dist_ce, ce_region=best_region,
        )

    candidates = []
    seen = set()
    for a in actions:
        fn = a["frame"]
        sid = int(a["fighter_type"].split("_")[1])
        candidates.append(score_at(fn, sid, a["confidence"], a.get("is_significant"), a))
        seen.add((fn, sid))

    # Also evaluate at every audio-onset frame for each fighter, if no action there
    for f, s in onsets:
        for sid in (0, 1):
            if (f, sid) in seen:
                continue
            # Only add if there is some plausible contact OR strong audio
            cand = score_at(f, sid, 0.0, False, None)
            # Reject obvious noise: needs at least some support beyond audio alone
            if cand["s_dist"] > 0.5 or cand["s_ce"] > 0.2 or cand["s_audio"] > 0.5 or cand["s_headacc"] > 0.4:
                candidates.append(cand)
                seen.add((f, sid))

    return candidates


def nms_temporal(candidates, cooldown, score_min):
    """Sort by score desc; keep if not within cooldown of an already-kept frame."""
    kept = []
    for c in sorted(candidates, key=lambda x: -x["score"]):
        if c["score"] < score_min: continue
        if any(abs(c["frame"] - k["frame"]) < cooldown for k in kept):
            continue
        kept.append(c)
    return sorted(kept, key=lambda x: x["frame"])


# ── Main: sweep & report ──────────────────────────────────────────────────────
def main():
    print("Loading SAM3D + actions...")
    sam3d, actions, contact_events, frame_persons = load_data()
    print(f"  {len(actions)} actions, {len(contact_events)} contact_events, "
          f"{len(frame_persons)} frames with persons")

    onsets = extract_audio_onsets(VIDEO_IN, FPS)

    # ── Grid search over weights & thresholds ─────────────────────────────────
    print("\nGrid search (this can take a minute)...")
    best = None
    n_eval = 0
    for w_act, w_ce, w_dist, w_audio, w_decel, w_headacc in product(
            [0.4, 0.6],            # action conf weight
            [0.0, 0.5, 1.0],       # contact_events weight
            [0.0, 0.4, 0.8],       # 3D dist weight
            [0.5, 1.0, 1.5],       # audio weight
            [0.0, 0.4, 0.8],       # wrist decel
            [0.0, 0.4, 0.8]):      # receiver head accel
        w_vel = 0.3
        w_sig = 0.4
        cands = build_candidates(actions, contact_events, frame_persons, onsets,
                                 w_act, w_ce, w_dist, w_vel, w_audio, w_sig,
                                 w_decel, w_headacc)
        for cd in [10, 12, 15, 20]:
            for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10]:
                kept = nms_temporal(cands, cd, thr)
                res = evaluate([k["frame"] for k in kept], GT_FRAMES, tol=30)
                n_eval += 1
                if best is None or res["f1"] > best["f1"]:
                    best = dict(res, w_act=w_act, w_ce=w_ce, w_dist=w_dist,
                                w_vel=w_vel, w_audio=w_audio, w_sig=w_sig,
                                w_decel=w_decel, w_headacc=w_headacc,
                                cd=cd, thr=thr, kept=kept)
    print(f"  evaluated {n_eval} configs")

    print(f"\nBest F1={best['f1']:.3f}  P={best['p']:.3f}  R={best['r']:.3f}")
    print(f"  TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    print(f"  weights: act={best['w_act']} ce={best['w_ce']} dist={best['w_dist']} "
          f"vel={best['w_vel']} audio={best['w_audio']} sig={best['w_sig']} "
          f"decel={best['w_decel']} headacc={best['w_headacc']}")
    print(f"  cooldown={best['cd']}  threshold={best['thr']}")
    print(f"  detections: {len(best['kept'])}")

    # ── Detailed match report ─────────────────────────────────────────────────
    print("\nDetailed match report (best config):")
    pairs_by_gt = {gi: (di, d) for (di, gi, d) in best["pairs"]}
    for gi, ts in enumerate(GT_TIMESTAMPS):
        gf = GT_FRAMES[gi]
        if gi in pairs_by_gt:
            di, dd = pairs_by_gt[gi]
            k = best["kept"][di]
            a = k["action"]
            act_name = a['action'] if a else 'audio_only'
            print(f"  [HIT]  {ts:8s} (f{gf:4d}) <- det f{k['frame']:4d} d={dd:2d}  "
                  f"score={k['score']:.2f} act={act_name:<14s} "
                  f"sid={k['sid']} ce={k['s_ce']:.2f} dist={k['s_dist']:.2f} "
                  f"audio={k['s_audio']:.2f} decel={k['s_decel']:.2f} hacc={k['s_headacc']:.2f}")
        else:
            print(f"  [MISS] {ts:8s} (f{gf:4d})")

    print("\nFalse positives:")
    matched_det_idx = {di for (di, _, _) in best["pairs"]}
    for i, k in enumerate(best["kept"]):
        if i not in matched_det_idx:
            act_name = k['action']['action'] if k['action'] else 'audio_only'
            print(f"  [FP] f{k['frame']:4d} score={k['score']:.2f} "
                  f"{act_name:14s} sid={k['sid']} "
                  f"ce={k['s_ce']:.2f} dist={k['s_dist']:.2f} audio={k['s_audio']:.2f} "
                  f"decel={k['s_decel']:.2f} hacc={k['s_headacc']:.2f}")

    # ── Save detections ───────────────────────────────────────────────────────
    out = {"config": {k: best[k] for k in ["w_act","w_ce","w_dist","w_vel","w_audio","w_sig","w_decel","w_headacc","cd","thr","f1","p","r","tp","fp","fn"]},
           "detections": [{"frame": k["frame"], "score": k["score"],
                           "striker_id": k["sid"],
                           "action": k["action"]["action"] if k["action"] else "audio_only",
                           "ce_dist": k["ce_dist"], "ce_region": k["ce_region"]}
                          for k in best["kept"]]}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
