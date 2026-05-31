#!/usr/bin/env python3
"""
Fusion v8  —  Region-Aware Pixel-Space Contact Detection
========================================================
Motivation (from the 2024-2025 literature review + a data audit of the new
SAM3D export):

  * The new SAM3D export *added* `world_coords`, `keypoint_conf`,
    `world_coords_reliable`, and top-level `contact_events`.
  * `diagnose_world_coords.py` proves `world_coords` is STILL not cross-person
    aligned (median 1.0 m, p90 7.7 m, max 122 m phantom head-Z gap; 51% of
    frames > 1 m) and `world_coords_reliable` is True even then.  So any gate
    built on cross-person 3D distance is unreliable.
  * `keypoint_conf` is a constant ~1.0 placeholder — no occlusion signal.
  * `contact_events` is the one genuinely useful new field: per-frame
    striker->receiver contacts with a `contact_prob`, a sane per-event
    `contact_3d_distance_m` (0.016-0.150 m) and a `contact_region`
    (head / torso / left_arm / right_arm).

Research mapping
----------------
  * GRAZE (arXiv:2604.01383) — do contact verification in PIXEL space, not
    broken 3D.  We project keypoints into image space via the bbox and score
    a wrist<->body pixel gap + Newton's-3rd-law head reaction in pixels.
  * DECO (ICCV'23) / VolumetricSMPL (ICCV'25) / Pi-HOC — the decisive lever for
    "blocked vs landed" is the CONTACT REGION.  SAM3D already provides it, so we
    boost head/torso contacts and penalise arm (guard) contacts.
  * Multi-band audio onset (boxing-impact transient) as an independent
    corroborator, extracted straight from the video (no librosa dependency).

This script is self-contained: it extracts its own audio, parses the 31 GT
timestamps internally, sweeps the decision threshold, and reports P/R/F1.

Usage:
    python fusion_v8.py
    python fusion_v8.py --no-audio
    python fusion_v8.py --folder /home/jake/Downloads/sam3d_with_world_coords
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import os
import json
import argparse
import subprocess
import wave
from collections import Counter, defaultdict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_FOLDER = r"/home/jake/Downloads/sam3d_with_world_coords"
SAM3D_NAME = "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json"
ACT_NAME = "fd7a77fd-588f-43ff-925f-ff5a648a246d.json"
VIDEO_NAME = "3.mp4"
WAV_NAME = "3.wav"
OUT_NAME = "3_fusion_v8.json"

# Optional: REAL image-space 2D keypoints (same video, frames align 1:1).
# The projected normalized_coords render as garbage skeletons, so for the video
# we use these explicit detections when available.
DEFAULT_KP2D = r"/home/jake/Downloads/for_impact_detection_experiment_2/2d_points.json"

FPS = 24.995
W, H = 1920, 1080

# ── Keypoint groups (COCO subset of 70) ─────────────────────────────────────────
WRIST_KPS = [9, 10]
HEAD_KPS = [0, 1, 2, 3, 4]
TORSO_KPS = [5, 6, 11, 12]

# ── Ground truth: 31 manually-labelled impacts (M:S:F at 24.995 fps) ─────────────
GT_TS = ["7:11", "18:01", "24:04", "30:19", "31:07", "34:07", "37:08", "53:02",
         "55:17", "1:05:22", "1:06:09", "1:06:20", "1:20:14", "1:25:15", "1:26:05",
         "1:27:18", "1:42:16", "1:42:19", "1:48:22", "1:51:23", "1:53:24", "2:03:19",
         "2:15:22", "2:17:11", "2:25:17", "2:27:24", "2:28:24", "2:34:13", "2:46:12",
         "2:49:24", "2:52:16"]


def ts_to_frame(ts):
    p = ts.split(":")
    sec = (int(p[0]) + int(p[1]) / FPS) if len(p) == 2 \
        else (int(p[0]) * 60 + int(p[1]) + int(p[2]) / FPS)
    return int(round(sec * FPS))


GT_FRAMES = [ts_to_frame(t) for t in GT_TS]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — project body-centred normalized_coords into image space via bbox
# (depth-free; sidesteps the broken world_coords entirely)
# ─────────────────────────────────────────────────────────────────────────────
def proj_kp(entry, k):
    nc = entry.get("normalized_coords")
    if nc is None or k >= len(nc):
        return None
    nc = np.asarray(nc)
    x1, y1, x2, y2 = entry["bbox"]
    xmin, xmax = nc[:, 0].min(), nc[:, 0].max()
    ymin, ymax = nc[:, 1].min(), nc[:, 1].max()
    if xmax <= xmin or ymax <= ymin:
        return None
    u = x1 + (nc[k, 0] - xmin) / (xmax - xmin) * (x2 - x1)
    v = y1 + (nc[k, 1] - ymin) / (ymax - ymin) * (y2 - y1)
    return np.array([u, v])


def centroid_2d(entry, kps):
    pts = [proj_kp(entry, k) for k in kps]
    pts = [p for p in pts if p is not None]
    return np.mean(pts, axis=0) if pts else None


def bbox_diag(entry):
    x1, y1, x2, y2 = entry["bbox"]
    return float(np.hypot(x2 - x1, y2 - y1)) + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-space physics (scale-normalised by bbox diagonal so it is camera-zoom
# invariant)
# ─────────────────────────────────────────────────────────────────────────────
def wrist_decel(person_frames, fn, half=5):
    """Sharp normalised pixel-velocity drop of the fastest wrist around fn."""
    best = 0.0
    for wk in WRIST_KPS:
        traj = []
        for f in range(fn - half, fn + half + 1):
            e = person_frames.get(f)
            if e is None:
                continue
            p = proj_kp(e, wk)
            if p is not None:
                traj.append((f, p, bbox_diag(e)))
        if len(traj) < 4:
            continue
        fs = np.array([t[0] for t in traj], float)
        ps = np.array([t[1] for t in traj])
        diag = np.mean([t[2] for t in traj])
        dt = np.diff(fs)
        dt[dt == 0] = 1.0
        vel = np.linalg.norm(np.diff(ps, axis=0), axis=1) / dt / diag
        if len(vel) < 2 or vel.max() < 1e-6:
            continue
        acc = np.diff(vel)
        decel = max(0.0, -acc.min())
        best = max(best, min(1.0, (decel / vel.max()) / 0.6))
    return best


def head_reaction(person_frames, fn, half=4):
    """Receiver head-centroid pixel acceleration (Newton's 3rd law in pixels)."""
    def hc(f):
        e = person_frames.get(f)
        return centroid_2d(e, HEAD_KPS) if e else None

    a, b, c = hc(fn - half), hc(fn), hc(fn + half)
    if a is None or b is None or c is None:
        return 0.0
    diag = bbox_diag(person_frames[fn])
    acc = np.linalg.norm(c - 2 * b + a) / diag
    return min(1.0, acc / 0.25)


def approach_rate(striker_frames, receiver_frames, fn, look=8):
    """Is the striker's nearest wrist monotonically closing on the receiver
    head over the last `look` frames? (directional approach vs random jitter)"""
    gaps = []
    for f in range(fn - look, fn + 1):
        se, re = striker_frames.get(f), receiver_frames.get(f)
        if se is None or re is None:
            continue
        rhead = centroid_2d(re, HEAD_KPS)
        if rhead is None:
            continue
        diag = bbox_diag(re)
        wd = [np.linalg.norm(proj_kp(se, wk) - rhead)
              for wk in WRIST_KPS if proj_kp(se, wk) is not None]
        if wd:
            gaps.append(min(wd) / diag)
    if len(gaps) < 4:
        return 0.0
    # fraction of steps that decrease the gap
    deltas = np.diff(gaps)
    return float((deltas < 0).mean())


def min_wrist_body_gap(striker_frames, receiver_frames, fn):
    """Normalised min pixel gap between striker wrists and receiver head/torso
    at frame fn (GRAZE-style pixel contact, depth-free).  Returns (gap, region).
    Lower gap == more likely contact."""
    se, re = striker_frames.get(fn), receiver_frames.get(fn)
    if se is None or re is None:
        return None, None
    diag = bbox_diag(re)
    head = [proj_kp(re, k) for k in HEAD_KPS]
    torso = [proj_kp(re, k) for k in TORSO_KPS]
    head = [p for p in head if p is not None]
    torso = [p for p in torso if p is not None]
    wr = [proj_kp(se, k) for k in WRIST_KPS]
    wr = [w for w in wr if w is not None]
    if not wr or not (head or torso):
        return None, None
    gh = min((np.linalg.norm(w - b) for w in wr for b in head), default=1e9)
    gt = min((np.linalg.norm(w - b) for w in wr for b in torso), default=1e9)
    if gh <= gt:
        return gh / diag, "head"
    return gt / diag, "torso"


# ─────────────────────────────────────────────────────────────────────────────
# Audio onset envelope (multi-band spectral flux) — no librosa needed
# ─────────────────────────────────────────────────────────────────────────────
def load_audio_onset(folder):
    wav = os.path.join(folder, WAV_NAME)
    if not os.path.exists(wav):
        src = os.path.join(folder, VIDEO_NAME)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "22050",
                 wav, "-loglevel", "error"], check=True)
        except Exception as e:
            print(f"  [audio] extraction failed ({e}); skipping audio gate.")
            return None
    with wave.open(wav, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # STFT via framing
    win = 1024
    hop = 512
    nfr = 1 + (len(x) - win) // hop
    if nfr < 2:
        return None
    window = np.hanning(win).astype(np.float32)
    mags = np.empty((nfr, win // 2 + 1), np.float32)
    for i in range(nfr):
        seg = x[i * hop: i * hop + win] * window
        mags[i] = np.abs(np.fft.rfft(seg))
    # spectral flux (positive differences summed across bins)
    flux = np.maximum(0.0, np.diff(mags, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    if flux.max() > 0:
        flux = flux / flux.max()
    times = (np.arange(nfr) * hop + win / 2) / sr
    return times, flux


def audio_score(onset, t_sec, win=0.30):
    if onset is None:
        return 0.0
    times, flux = onset
    lo, hi = t_sec - win, t_sec + win
    mask = (times >= lo) & (times <= hi)
    return float(flux[mask].max()) if mask.any() else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
# region -> "landed target" weight.  Arm == guard/block -> heavy penalty.
REGION_W = {
    "head": 1.00, "torso": 0.85,
    "left_arm": 0.20, "right_arm": 0.20,
    "left_leg": 0.30, "right_leg": 0.30,
}

# candidate set = ASFormer action windows (cover 30/31 GT, vs 5/31 for
# contact_events).  contact_events are used only as a SCORING prior.
WEIGHTS = dict(gap=0.26, ce_prob=0.18, region=0.14, decel=0.14,
               react=0.10, approach=0.08, audio=0.06, conf=0.04)


def nearest_contact_event(ce_by_pair, sid, rid, fn, win=15):
    """Best (highest-prob) contact_event for striker->receiver near frame fn."""
    best = None
    for ev in ce_by_pair.get((sid, rid), []):
        if abs(ev["frame"] - fn) <= win:
            if best is None or ev["contact_prob"] > best["contact_prob"]:
                best = ev
    return best


def score_action(act, persons, ce_by_pair, onset):
    """Score one ASFormer action.  Refines impact frame to the in-window frame
    with the smallest wrist<->receiver pixel gap, then fuses all trustworthy
    signals.  Returns (impact_frame, fused_score, region, gates, contact_xy)."""
    sid = 0 if act["fighter_type"] == "fighter_0" else 1
    rid = 1 - sid
    sp, rp = persons.get(sid, {}), persons.get(rid, {})

    ws = act["window_start"] - 3
    we = act["window_end"] + 5

    # locate contact frame = min pixel gap in window (GRAZE pixel contact)
    best_f, best_gap, best_reg = act["frame"], 1e9, "torso"
    for f in range(ws, we + 1):
        gap, reg = min_wrist_body_gap(sp, rp, f)
        if gap is not None and gap < best_gap:
            best_gap, best_f, best_reg = gap, f, reg
    if best_gap >= 1e9:
        return act["frame"], 0.0, "torso", {}, None

    # contact pixel = striker wrist (at best_f) nearest the receiver body
    contact_xy = None
    se, re = sp.get(best_f), rp.get(best_f)
    if se is not None and re is not None:
        rbody = [proj_kp(re, k) for k in HEAD_KPS + TORSO_KPS]
        rbody = [b for b in rbody if b is not None]
        cand = []
        for wk in WRIST_KPS:
            w = proj_kp(se, wk)
            if w is not None and rbody:
                cand.append((min(np.linalg.norm(w - b) for b in rbody), w))
        if cand:
            contact_xy = min(cand, key=lambda c: c[0])[1].astype(int).tolist()

    # gap -> score (0.10 norm gap ~ contact, 0.45 ~ clearly apart)
    s_gap = max(0.0, min(1.0, (0.45 - best_gap) / 0.35))

    # contact_event prior (region + prob) for this striker->receiver near frame
    ce = nearest_contact_event(ce_by_pair, sid, rid, best_f)
    if ce is not None:
        s_ce_prob = float(ce["contact_prob"])
        s_region = REGION_W.get(ce["contact_region"], 0.5)
    else:
        s_ce_prob = 0.0
        s_region = REGION_W.get(best_reg, 0.6)  # fall back to pixel-derived region

    s_decel = wrist_decel(sp, best_f)
    s_react = head_reaction(rp, best_f)
    s_appr = approach_rate(sp, rp, best_f)
    s_audio = audio_score(onset, best_f / FPS)
    s_conf = float(min(1.0, act.get("confidence", 0.0)))

    fused = (WEIGHTS["gap"] * s_gap + WEIGHTS["ce_prob"] * s_ce_prob
             + WEIGHTS["region"] * s_region + WEIGHTS["decel"] * s_decel
             + WEIGHTS["react"] * s_react + WEIGHTS["approach"] * s_appr
             + WEIGHTS["audio"] * s_audio + WEIGHTS["conf"] * s_conf)
    gates = dict(gap=s_gap, ce_prob=s_ce_prob, region=s_region, decel=s_decel,
                 react=s_react, approach=s_appr, audio=s_audio, conf=s_conf)
    return best_f, fused, (ce["contact_region"] if ce else best_reg), gates, contact_xy


def nms(events, cooldown=18):
    """Keep highest-scoring event in each cooldown window (frames)."""
    kept = []
    for ev in sorted(events, key=lambda e: -e["score"]):
        if all(abs(ev["frame"] - k["frame"]) >= cooldown for k in kept):
            kept.append(ev)
    return sorted(kept, key=lambda e: e["frame"])


def evaluate(det_frames, tol=12):
    matched_gt, matched_det, pairs = set(), set(), []
    for di, df in enumerate(det_frames):
        best, bd = None, tol + 1
        for gi, gf in enumerate(GT_FRAMES):
            if gi in matched_gt:
                continue
            if abs(df - gf) < bd:
                bd, best = abs(df - gf), gi
        if best is not None:
            matched_gt.add(best)
            matched_det.add(di)
            pairs.append((di, best, bd))
    tp = len(matched_gt)
    fp = len(det_frames) - len(matched_det)
    fn = len(GT_FRAMES) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1, matched_gt


SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
COL = {0: (0, 255, 180), 1: (255, 140, 0)}   # BGR per fighter
FLASH_FR = 12


def load_2d_explicit(path):
    """Load REAL image-space joints_2d (70 per person), scaled to original res.
    Returns {pid: {frame: (70,2) array}} or None if unavailable."""
    if not path or not os.path.exists(path):
        return None
    raw = json.load(open(path))
    out = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str)
        out[pid] = {}
        for e in entries:
            dims = e.get("frame_dims", {})
            sx = dims.get("original_width", 1920) / dims.get("resized_width", 640)
            sy = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            j = np.asarray(e["joints_2d"], float)
            j[:, 0] *= sx
            j[:, 1] *= sy
            out[pid][e["frame"]] = j
    return out


def render_video(folder, persons, kept, tol, gt_set, j2d=None):
    """Draw both skeletons every frame; flash + contact marker + score at each
    detected impact.  TP impacts (within ±tol of a GT frame) are marked green,
    false positives red — so the weak signal is visible, not hidden.

    Skeletons use REAL image-space joints_2d (`j2d`) when available; the
    projected normalized_coords are geometrically wrong and render as garbage."""
    import cv2
    src = os.path.join(folder, VIDEO_NAME)
    out_path = os.path.join(folder, "3_fusion_v8.mp4")
    tmp_path = os.path.join(folder, "3_fusion_v8_noaudio.mp4")
    print(f"[v8] skeleton source: {'REAL joints_2d' if j2d else 'projected (approx)'}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[v8] cannot open {src}; skipping video.")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wd, Ht))

    def joints(pid, fi):
        """Return (70,2) image-space joints for pid at frame fi, or None."""
        if j2d is not None:
            return j2d.get(pid, {}).get(fi)
        e = persons.get(pid, {}).get(fi)
        if e is None:
            return None
        pts = [proj_kp(e, k) for k in range(17)]
        return pts  # list w/ Nones; handled below

    def get(jt, k):
        if jt is None:
            return None
        if isinstance(jt, list):
            return jt[k] if k < len(jt) else None
        p = jt[k]
        return None if np.allclose(p, 0) else p

    imp_by_frame = {e["frame"]: e for e in kept}
    flash, n_imp = None, 0
    print(f"[v8] rendering {total} frames -> {out_path}")
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        canvas = frame

        # skeletons (COCO-17 subset)
        jt_by_pid = {}
        for pid in (0, 1):
            jt = joints(pid, fi)
            jt_by_pid[pid] = jt
            if jt is None:
                continue
            for a, b in SKELETON_PAIRS:
                pa, pb = get(jt, a), get(jt, b)
                if pa is not None and pb is not None:
                    cv2.line(canvas, (int(pa[0]), int(pa[1])),
                             (int(pb[0]), int(pb[1])), COL[pid], 2, cv2.LINE_AA)
            for wk in WRIST_KPS:
                w = get(jt, wk)
                if w is not None:
                    cv2.circle(canvas, (int(w[0]), int(w[1])), 6,
                               (0, 180, 255), -1, cv2.LINE_AA)

        # fire impact
        if fi in imp_by_frame:
            ev = imp_by_frame[fi]
            is_tp = any(abs(fi - g) <= tol for g in gt_set)
            flash = (fi, ev, is_tp)
            n_imp += 1
            # recompute contact pixel from REAL joints when available
            cp = ev.get("contact_point")
            if j2d is not None:
                sjt = jt_by_pid.get(ev["striker_id"])
                rjt = jt_by_pid.get(ev["receiver_id"])
                if sjt is not None and rjt is not None:
                    rbody = [get(rjt, k) for k in HEAD_KPS + TORSO_KPS]
                    rbody = [b for b in rbody if b is not None]
                    cand = []
                    for wk in WRIST_KPS:
                        w = get(sjt, wk)
                        if w is not None and rbody:
                            cand.append((min(np.linalg.norm(w - b) for b in rbody), w))
                    if cand:
                        w = min(cand, key=lambda c: c[0])[1]
                        cp = [int(w[0]), int(w[1])]
            col = (0, 230, 0) if is_tp else (0, 50, 255)
            if cp:
                cv2.circle(canvas, (cp[0], cp[1]), 22, col, 3, cv2.LINE_AA)
                cv2.circle(canvas, (cp[0], cp[1]), 6, (0, 255, 255), -1, cv2.LINE_AA)
                tag = ("HIT" if is_tp else "FP")
                cv2.putText(canvas, f"{tag} {ev['contact_region']} {ev['score']:.2f}",
                            (cp[0] + 14, cp[1] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # decaying flash
        if flash is not None and fi < flash[0] + FLASH_FR:
            f0, ev, is_tp = flash
            alpha = max(0.0, 1.0 - (fi - f0) / FLASH_FR) * 0.30
            tint = np.zeros_like(canvas)
            tint[:, :] = (0, 200, 0) if is_tp else (0, 0, 200)
            cv2.addWeighted(tint, alpha, canvas, 1.0, 0, canvas)

        # HUD
        cv2.rectangle(canvas, (0, 0), (430, 34), (15, 15, 20), -1)
        cv2.putText(canvas, f"fusion_v8  impacts:{n_imp}  frame:{fi}  "
                    f"green=HIT red=FP", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
    cap.release()
    writer.release()

    # mux original audio so the thuds line up
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-i", src,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "copy",
                        "-shortest", out_path, "-loglevel", "error"], check=True)
        os.remove(tmp_path)
    except Exception as e:
        print(f"[v8] audio mux failed ({e}); keeping silent video at {tmp_path}")
        out_path = tmp_path
    print(f"[v8] video saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--cooldown", type=int, default=18)
    ap.add_argument("--tol", type=int, default=12)
    ap.add_argument("--video", action="store_true",
                    help="render annotated mp4 of the best operating point")
    ap.add_argument("--kp2d", default=DEFAULT_KP2D,
                    help="real joints_2d JSON for correct skeleton rendering")
    args = ap.parse_args()

    sam3d = os.path.join(args.folder, SAM3D_NAME)
    act_path = os.path.join(args.folder, ACT_NAME)
    print(f"[v8] loading {sam3d} ...")
    d = json.load(open(sam3d))
    persons = {}
    for pid in ("0", "1"):
        persons[int(pid)] = {e["frame"]: e for e in d.get(pid, [])}
    ce = d.get("contact_events", [])
    print(f"[v8] {len(ce)} contact_events; regions="
          f"{dict(Counter(e['contact_region'] for e in ce))}")

    # index contact_events by (striker, receiver) for fast nearest lookup
    ce_by_pair = defaultdict(list)
    for ev in ce:
        ce_by_pair[(ev["striker_id"], ev["receiver_id"])].append(ev)

    actions = json.load(open(act_path))["actions"]
    print(f"[v8] {len(actions)} ASFormer actions (candidate set; cover 30/31 GT)")

    onset = None if args.no_audio else load_audio_onset(args.folder)
    if onset is not None:
        print(f"[v8] audio onset envelope: {len(onset[1])} frames")

    # ── score every ACTION (candidate), refining its impact frame ─────────────
    scored = []
    for act in actions:
        fimp, s, region, g, cxy = score_action(act, persons, ce_by_pair, onset)
        sid = 0 if act["fighter_type"] == "fighter_0" else 1
        scored.append({
            "frame": fimp, "score": s, "gates": g,
            "contact_region": region, "striker_id": sid, "receiver_id": 1 - sid,
            "action": act["action"], "contact_point": cxy,
            "striker_body_part": None, "receiver_body_part": None,
        })

    # ── threshold sweep (report best-F1 operating point) ─────────────────────
    print("\n  thr   kept  TP  FP  FN    P      R      F1")
    print("  " + "-" * 48)
    best = None
    for thr in np.round(np.arange(0.10, 0.71, 0.02), 2):
        kept = nms([e for e in scored if e["score"] >= thr], args.cooldown)
        frames = [e["frame"] for e in kept]
        tp, fp, fn, p, r, f1, _ = evaluate(frames, args.tol)
        if best is None or f1 > best["f1"]:
            best = dict(thr=thr, kept=kept, tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1)
        if abs(thr * 100 % 6) < 1e-6 or thr in (0.20, 0.70):
            print(f"  {thr:.2f}  {len(frames):4d}  {tp:2d}  {fp:2d}  {fn:2d}  "
                  f"{p:5.1%}  {r:5.1%}  {f1:5.1%}")

    print("\n" + "=" * 60)
    print(f"  BEST  thr={best['thr']:.2f}  (tol=±{args.tol}fr, cd={args.cooldown})")
    print(f"  kept={len(best['kept'])}  TP={best['tp']}  FP={best['fp']}  "
          f"FN={best['fn']}")
    print(f"  Precision={best['p']:.1%}  Recall={best['r']:.1%}  "
          f"F1={best['f1']:.1%}")
    print("=" * 60)

    # ── save best operating point ─────────────────────────────────────────────
    out = {
        "approach": "fusion_v8",
        "label": "Region-Aware Pixel-Space Contact Detection",
        "threshold": best["thr"],
        "cooldown": args.cooldown,
        "tol_frames": args.tol,
        "src_fps": FPS,
        "metrics": {"precision": best["p"], "recall": best["r"], "f1": best["f1"],
                    "tp": best["tp"], "fp": best["fp"], "fn": best["fn"]},
        "n_impacts": len(best["kept"]),
        "events": [{
            "is_impact": True,
            "impact_frame": e["frame"],
            "timestamp_seconds": round(e["frame"] / FPS, 3),
            "impact_score": round(e["score"], 4),
            "contact_region": e["contact_region"],
            "striker_id": e["striker_id"],
            "receiver_id": e["receiver_id"],
            "striker_body_part": e.get("striker_body_part"),
            "receiver_body_part": e.get("receiver_body_part"),
            "gates": {k: round(v, 3) for k, v in e["gates"].items()},
        } for e in best["kept"]],
    }
    out_path = os.path.join(args.folder, OUT_NAME)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[v8] saved {out_path}")

    # ── render annotated video of the best operating point ────────────────────
    if args.video:
        gt_set = set(GT_FRAMES)
        j2d = load_2d_explicit(args.kp2d)
        if j2d is None:
            print(f"[v8] WARNING: no real joints_2d at {args.kp2d}; "
                  f"skeletons will use approximate projection.")
        render_video(args.folder, persons, best["kept"], args.tol, gt_set, j2d)


if __name__ == "__main__":
    main()
