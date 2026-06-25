#!/usr/bin/env python3
"""
Fusion v9 — MHR70-correct, world-XY contact detector for the 05-06-2026 export
==============================================================================
Why a new file (vs v8):
  * The "*_Fixed_05062026" export is MHR70, NOT COCO-17.  Wrists are at index
    41 (right) / 62 (left); indices 9/10 are HIPS.  v8/config used 9/10 — i.e.
    it measured hip-to-body, not wrist-to-body.  Fixed here.
  * `world_coords` Z is still per-person independent (cross-person head-Z gap
    median 0.5-0.7 m), so 3D distance is unusable.  BUT world_coords X and Y are
    shared-space metric (ground-plane normalised).  So we measure the
    striker-wrist -> receiver-body gap in the X-Y plane only (depth-free),
    normalised by the receiver's shoulder width (scale/zoom invariant).
  * `contact_events` now live in *_full_analysis.json (not sam3d.json) and use
    correct MHR70 indices — used here only as a SCORING PRIOR (prob + region).
  * Candidate set = ASFormer actions (the high-recall generator).  No training.

HONEST SCOPE: this is a static fusion scorer.  With a single broadcast camera
the landed-vs-missed depth cue is below resolution, so treat the output as
ranked impact *candidates*, not verified hits.  No GT exists for these fights,
so no P/R/F1 is reported — only scores.

Usage:
    python detectors/fusion/v9.py --folder "C:/Users/XRIG/Downloads/1st_Impact_detection_Fixed_05062026"
    python detectors/fusion/v9.py --folder "<dir>" --thr 0.45 --cooldown 18 --audio
"""
import os, sys, json, glob, argparse, subprocess, wave
from collections import Counter, defaultdict
import numpy as np

# ── MHR70 indices ───────────────────────────────────────────────────────────
RW, LW = 41, 62                 # right / left wrist
WRIST_KPS = [RW, LW]
HEAD_KPS  = [0, 1, 2, 3, 4]
TORSO_KPS = [5, 6, 9, 10]       # shoulders + hips (MHR70)
SHO_KPS   = [5, 6]
GUARD_KPS = [7, 8, 41, 62]   # defender guard: left-elbow, right-elbow, right-wrist, left-wrist

REGION_W = {"head": 1.00, "torso": 0.85, "left_arm": 0.20, "right_arm": 0.20,
            "left_leg": 0.30, "right_leg": 0.30}
WEIGHTS = dict(gap=0.26, ce_prob=0.08, region=0.10, decel=0.08, ext=0.10,
               speed=0.08, react=0.06, approach=0.06, audio=0.04, conf=0.04,
               guard=0.10)

# wrist (MHR70) -> (shoulder, elbow) for that arm
ARM_CHAIN = {62: (5, 7), 41: (6, 8)}

# ── Precision filters (tunable) ──────────────────────────────────────────────
# A detection is REJECTED outright when any of these fire:
GAP_MAX      = 0.95   # wrist never got within this many shoulder-widths of body
                      #   => not a landed punch (slip / squaring up / ref break)
IOU_REJECT   = 0.18   # bbox overlap this high AND no committed incoming punch => clinch
EXT_MIN      = 0.45   # striking-arm extension ratio below this == bent (guard)
APP_MIN      = 0.45   # wrist-approach fraction below this == not closing in
MIN_PEAK_SPD = 0.42   # striking wrist must reach this peak speed (shoulder-w/frame);
                      #   below it the wrist isn't punching (clinch/rest/keypoint error)
CLINCH_PEN   = 0.6    # soft: multiply score by (1 - CLINCH_PEN*iou) below reject line


# ── geometry helpers (X-Y plane of world_coords; depth dropped) ──────────────
def wc(entry):
    a = np.asarray(entry.get("world_coords", []), float)
    return a if a.ndim == 2 and a.shape[0] >= 63 else None

def xy(a, idx):
    return a[idx, 0:2]

def shoulder_w(a):
    return float(np.linalg.norm(xy(a, 5) - xy(a, 6))) + 1e-6

def body_pts(a, idxs):
    return [xy(a, k) for k in idxs if k < a.shape[0]]

def min_wrist_body_gap(se, re):
    """Min XY gap striker-wrist -> receiver head/torso, /shoulder-width."""
    if se is None or re is None:
        return None, None
    sw = shoulder_w(re)
    head = body_pts(re, HEAD_KPS)
    torso = body_pts(re, TORSO_KPS)
    wr = [xy(se, k) for k in WRIST_KPS]
    gh = min((np.linalg.norm(w - b) for w in wr for b in head), default=1e9)
    gt = min((np.linalg.norm(w - b) for w in wr for b in torso), default=1e9)
    return (gh / sw, "head") if gh <= gt else (gt / sw, "torso")

def arm_extension(e):
    """Max extension ratio over both arms: |wrist-shoulder| / (|sh-elbow|+|elbow-wr|).
    ~1.0 fully extended (committed punch), low when bent (guard/retracted)."""
    a = wc(e)
    if a is None:
        return 0.0
    best = 0.0
    for wk, (sh, el) in ARM_CHAIN.items():
        if wk >= a.shape[0]:
            continue
        w, s, l = xy(a, wk), xy(a, sh), xy(a, el)
        chain = np.linalg.norm(s - l) + np.linalg.norm(l - w)
        if chain > 1e-6:
            best = max(best, float(np.linalg.norm(w - s) / chain))
    return best

def bbox_iou(ea, eb):
    """IoU of the two fighters' bounding boxes (clinch indicator)."""
    if ea is None or eb is None:
        return 0.0
    ax1, ay1, ax2, ay2 = ea["bbox"]; bx1, by1, bx2, by2 = eb["bbox"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1); ab = (bx2 - bx1) * (by2 - by1)
    return float(inter / (aa + ab - inter + 1e-6))

def mean_iou(sp, rp, fn, half=2):
    vals = [bbox_iou(sp.get(f), rp.get(f))
            for f in range(fn - half, fn + half + 1)
            if sp.get(f) and rp.get(f)]
    return float(np.mean(vals)) if vals else 0.0

def wrist_peak_speed(frames, fn, half=6):
    """Max normalised XY speed of either wrist in the window (shoulder-w/frame).
    A real punch is a fast wrist motion; clinch/rest/keypoint-error are slow."""
    best = 0.0
    for wk in WRIST_KPS:
        traj = []
        for f in range(fn - half, fn + half + 1):
            e = frames.get(f); a = wc(e) if e else None
            if a is not None:
                traj.append((f, xy(a, wk), shoulder_w(a)))
        if len(traj) < 3:
            continue
        fs = np.array([t[0] for t in traj], float)
        ps = np.array([t[1] for t in traj])
        sw = np.mean([t[2] for t in traj])
        dt = np.diff(fs); dt[dt == 0] = 1.0
        vel = np.linalg.norm(np.diff(ps, axis=0), axis=1) / dt / sw
        if len(vel):
            best = max(best, float(vel.max()))
    return best

def wrist_decel(frames, fn, half=5):
    best = 0.0
    for wk in WRIST_KPS:
        traj = []
        for f in range(fn - half, fn + half + 1):
            e = frames.get(f); a = wc(e) if e else None
            if a is not None:
                traj.append((f, xy(a, wk), shoulder_w(a)))
        if len(traj) < 4:
            continue
        fs = np.array([t[0] for t in traj], float)
        ps = np.array([t[1] for t in traj])
        sw = np.mean([t[2] for t in traj])
        dt = np.diff(fs); dt[dt == 0] = 1.0
        vel = np.linalg.norm(np.diff(ps, axis=0), axis=1) / dt / sw
        if len(vel) < 2 or vel.max() < 1e-6:
            continue
        decel = max(0.0, -np.diff(vel).min())
        best = max(best, min(1.0, (decel / vel.max()) / 0.6))
    return best

def head_reaction(frames, fn, half=4):
    def hc(f):
        e = frames.get(f); a = wc(e) if e else None
        if a is None:
            return None, None
        pts = body_pts(a, HEAD_KPS)
        return (np.mean(pts, axis=0) if pts else None), shoulder_w(a)
    a, sw = hc(fn); b, _ = hc(fn - half); c, _ = hc(fn + half)
    if a is None or b is None or c is None:
        return 0.0
    acc = np.linalg.norm(c - 2 * a + b) / sw
    return min(1.0, acc / 1.5)

def approach_rate(sf, rf, fn, look=8):
    gaps = []
    for f in range(fn - look, fn + 1):
        se = wc(sf.get(f)) if sf.get(f) else None
        re = wc(rf.get(f)) if rf.get(f) else None
        if se is None or re is None:
            continue
        sw = shoulder_w(re)
        head = body_pts(re, HEAD_KPS)
        wr = [xy(se, k) for k in WRIST_KPS]
        g = min((np.linalg.norm(w - b) for w in wr for b in head), default=None)
        if g is not None:
            gaps.append(g / sw)
    if len(gaps) < 4:
        return 0.0
    return float((np.diff(gaps) < 0).mean())


def _pt_seg_dist(p, a, b):
    """Minimum 2-D distance from point p to segment a-b."""
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def guard_interception(sf, rf, fn, half=3):
    """Score how open the defender's guard is along the punch path.

    For frames [fn-half, fn] we draw the segment from the striker's punching
    wrist to the receiver's head centroid and find the minimum distance from
    any receiver guard keypoint (elbow / wrist) to that segment, normalised
    by shoulder width.

    Returns [0, 1]:  0 = guard arm is directly across the punch line (blocked),
                     1 = guard is wide open (path to head is clear, likely landed).
    """
    BLOCKED_DIST = 0.30   # normalised guard dist below which we call "interposed"
    OPEN_DIST    = 1.50   # above which the guard is clearly out of the way

    min_d = 1e9
    for f in range(fn - half, fn + 1):
        se_raw = sf.get(f)
        re_raw = rf.get(f)
        if se_raw is None or re_raw is None:
            continue
        sa = wc(se_raw)
        ra = wc(re_raw)
        if sa is None or ra is None:
            continue

        sw = shoulder_w(ra)
        head = body_pts(ra, HEAD_KPS)
        if not head:
            continue
        head_c = np.mean(head, axis=0)

        # whichever striker wrist is closer to the defender head is the punching hand
        wr_pts = [xy(sa, k) for k in WRIST_KPS]
        punch_wrist = min(wr_pts, key=lambda w: np.linalg.norm(w - head_c))

        for gk in GUARD_KPS:
            if gk >= ra.shape[0]:
                continue
            d = _pt_seg_dist(xy(ra, gk), punch_wrist, head_c) / sw
            if d < min_d:
                min_d = d

    if min_d >= 1e9:
        return 0.5   # no data — neutral
    return float(np.clip((min_d - BLOCKED_DIST) / (OPEN_DIST - BLOCKED_DIST), 0.0, 1.0))


# ── audio (optional, multi-band spectral flux; no librosa) ───────────────────
def load_audio_onset(video_path, folder):
    wavp = os.path.join(folder, "_v9_audio.wav")
    if not os.path.exists(wavp):
        try:
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ac", "1",
                            "-ar", "22050", wavp, "-loglevel", "error"], check=True)
        except Exception as e:
            print(f"  [audio] extraction failed ({e}); skipping audio gate.")
            return None
    with wave.open(wavp, "rb") as wf:
        sr, n = wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    win, hop = 1024, 512
    nfr = 1 + (len(x) - win) // hop
    if nfr < 2:
        return None
    window = np.hanning(win).astype(np.float32)
    mags = np.empty((nfr, win // 2 + 1), np.float32)
    for i in range(nfr):
        mags[i] = np.abs(np.fft.rfft(x[i * hop:i * hop + win] * window))
    flux = np.maximum(0.0, np.diff(mags, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    if flux.max() > 0:
        flux /= flux.max()
    times = (np.arange(nfr) * hop + win / 2) / sr
    return times, flux

def audio_score(onset, t_sec, win=0.30):
    if onset is None:
        return 0.0
    times, flux = onset
    m = (times >= t_sec - win) & (times <= t_sec + win)
    return float(flux[m].max()) if m.any() else 0.0


# ── scoring ─────────────────────────────────────────────────────────────────
def nearest_ce(ce_by_pair, sid, rid, fn, win=15):
    best = None
    for ev in ce_by_pair.get((sid, rid), []):
        if abs(ev["frame"] - fn) <= win and (best is None or ev["contact_prob"] > best["contact_prob"]):
            best = ev
    return best

def score_action(act, persons, ce_by_pair, onset, fps):
    sid = 0 if act["fighter_type"] == "fighter_0" else 1
    rid = 1 - sid
    sp, rp = persons.get(sid, {}), persons.get(rid, {})
    ws, we = act["window_start"] - 3, act["window_end"] + 5

    best_f, best_gap, best_reg = act["frame"], 1e9, "torso"
    for f in range(ws, we + 1):
        g, reg = min_wrist_body_gap(wc(sp.get(f)) if sp.get(f) else None,
                                    wc(rp.get(f)) if rp.get(f) else None)
        if g is not None and g < best_gap:
            best_gap, best_f, best_reg = g, f, reg
    if best_gap >= 1e9:
        return None

    # gap (shoulder-widths) -> score: ~0.3 sw == contact, ~1.3 sw == clearly apart
    s_gap = max(0.0, min(1.0, (1.3 - best_gap) / 1.0))
    ce = nearest_ce(ce_by_pair, sid, rid, best_f)
    if ce is not None:
        s_ce = float(ce["contact_prob"]); s_reg = REGION_W.get(ce["contact_region"], 0.5)
        region = ce["contact_region"]
    else:
        s_ce = 0.0; s_reg = REGION_W.get(best_reg, 0.6); region = best_reg
    s_dec = wrist_decel(sp, best_f)
    s_rea = head_reaction(rp, best_f)
    s_app = approach_rate(sp, rp, best_f)
    s_grd = guard_interception(sp, rp, best_f)
    s_aud = audio_score(onset, best_f / fps)
    s_con = float(min(1.0, act.get("confidence", 0.0)))

    # ── precision signals: arm extension + clinch (bbox overlap) ──────────────
    ext = max(arm_extension(sp.get(f)) for f in range(best_f - 1, best_f + 2)
              if sp.get(f)) if any(sp.get(f) for f in range(best_f - 1, best_f + 2)) else 0.0
    iou = mean_iou(sp, rp, best_f)
    peak = wrist_peak_speed(sp, best_f)
    s_ext = max(0.0, min(1.0, (ext - 0.45) / 0.40))
    s_spd = max(0.0, min(1.0, (peak - 0.40) / 0.60))

    # HARD FILTERS (drop clear non-impacts):
    #  1) wrist never near the body == slip / miss / squaring up / ref break
    #  2) striking wrist too slow == clinch / rest / keypoint error (e.g. f31)
    #  3) bbox overlap high AND no committed extended punch closing in == clinch
    #  4) arm bent AND not closing in == guard / retracting
    if best_gap > GAP_MAX:
        return None
    if peak < MIN_PEAK_SPD:
        return None
    if iou >= IOU_REJECT and s_app < 0.55 and ext < 0.90:
        return None
    if ext < EXT_MIN and s_app < APP_MIN:
        return None

    fused = (WEIGHTS["gap"]*s_gap + WEIGHTS["ce_prob"]*s_ce + WEIGHTS["region"]*s_reg
             + WEIGHTS["decel"]*s_dec + WEIGHTS["ext"]*s_ext + WEIGHTS["speed"]*s_spd
             + WEIGHTS["react"]*s_rea + WEIGHTS["approach"]*s_app
             + WEIGHTS["guard"]*s_grd
             + WEIGHTS["audio"]*s_aud + WEIGHTS["conf"]*s_con)
    # soft clinch penalty for partial overlap below the reject line
    fused *= (1.0 - CLINCH_PEN * iou)
    gates = dict(gap=s_gap, ce_prob=s_ce, region=s_reg, decel=s_dec, ext=s_ext,
                 speed=s_spd, react=s_rea, approach=s_app, guard=s_grd,
                 audio=s_aud, conf=s_con,
                 gap_shoulder_widths=round(best_gap, 3), arm_ext=round(ext, 3),
                 bbox_iou=round(iou, 3), wrist_peak=round(peak, 3))
    return dict(frame=best_f, score=fused, gates=gates, contact_region=region,
                striker_id=sid, receiver_id=rid, action=act["action"],
                confidence=s_con, window_start=act["window_start"], window_end=act["window_end"])

def nms(events, cooldown=18):
    kept = []
    for ev in sorted(events, key=lambda e: -e["score"]):
        if all(abs(ev["frame"] - k["frame"]) >= cooldown for k in kept):
            kept.append(ev)
    return sorted(kept, key=lambda e: e["frame"])


# ── annotated video ───────────────────────────────────────────────────────────
FLASH_FR = 12

def _region_marker(rec_entry, region):
    """Exact-ish marker from the receiver bbox + struck region (no garbage
    skeleton projection).  Returns (x, y) pixel or None."""
    if rec_entry is None:
        return None
    x1, y1, x2, y2 = rec_entry["bbox"]
    cx = (x1 + x2) / 2.0
    h = y2 - y1
    if region == "head":
        return int(cx), int(y1 + 0.13 * h)
    if region in ("left_arm", "right_arm"):
        return int(cx), int(y1 + 0.40 * h)
    if region in ("left_leg", "right_leg"):
        return int(cx), int(y1 + 0.80 * h)
    return int(cx), int(y1 + 0.45 * h)          # torso / default

def _draw_marker(frame, mk, score, frozen=False):
    """Draw the impact marker (circle + label) on frame in place."""
    import cv2
    if mk is None:
        return
    cv2.circle(frame, mk, 26, (0, 230, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, mk, 6, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, f"IMPACT {score:.2f}",
                (mk[0] + 16, mk[1] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if frozen:
        cv2.circle(frame, mk, 40, (0, 255, 255), 2, cv2.LINE_AA)


def render_video(folder, video_path, persons, kept, fps, freeze_sec=1.5, tag="v9"):
    import cv2
    base = os.path.splitext(os.path.basename(video_path))[0]
    suffix = f"_impacts_{tag}_frozen" if freeze_sec > 0 else f"_impacts_{tag}"
    tmp = os.path.join(folder, f"{base}_v9_tmp.mp4")
    out_path = os.path.join(folder, f"{base}{suffix}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[v9] cannot open {video_path}; skipping video.")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Wd, Ht = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wd, Ht))
    imp_by_frame = {e["frame"]: e for e in kept}
    freeze_n = int(round(freeze_sec * fps))
    print(f"[v9] rendering {total} frames @ {Wd}x{Ht} -> {os.path.basename(out_path)}"
          f"  (freeze {freeze_sec}s = {freeze_n} held frames per impact)")
    flash, n = None, 0
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi in imp_by_frame:
            ev = imp_by_frame[fi]
            flash = (fi, ev); n += 1
            # ── FREEZE: hold this impact frame for freeze_sec ──────────────
            if freeze_n > 0:
                hold = frame.copy()
                tint = np.zeros_like(hold); tint[:, :] = (0, 180, 0)
                cv2.addWeighted(tint, 0.35, hold, 1.0, 0, hold)
                cv2.rectangle(hold, (0, 0), (760, 34), (15, 15, 20), -1)
                cv2.putText(hold, f"IMPACT #{n}  frame:{fi}  score:{ev['score']:.2f}"
                            f"  (holding {freeze_sec}s for review)", (8, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                for _ in range(freeze_n):
                    writer.write(hold)
        if flash is not None and fi < flash[0] + FLASH_FR:
            f0, ev = flash
            a = max(0.0, 1.0 - (fi - f0) / FLASH_FR) * 0.40
            tint = np.zeros_like(frame); tint[:, :] = (0, 200, 0)
            cv2.addWeighted(tint, a, frame, 1.0, 0, frame)
        cv2.rectangle(frame, (0, 0), (470, 34), (15, 15, 20), -1)
        cv2.putText(frame, f"fusion_v9  impacts:{n}  frame:{fi}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    cap.release(); writer.release()
    try:
        subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", video_path,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "copy",
                        "-shortest", out_path, "-loglevel", "error"], check=True)
        os.remove(tmp)
    except Exception:
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(tmp, out_path)
        print("[v9] ffmpeg not available; saved silent video (no audio mux).")
    print(f"[v9] video saved -> {out_path}")
    return out_path


def save_impact_shots(folder, video_path, persons, kept, fps, max_w=820, tag="v9"):
    """Save one annotated still per impact (cropped around both fighters) so the
    contact can be visually verified.  Returns the shots directory."""
    import cv2
    base = os.path.splitext(os.path.basename(video_path))[0]
    shots_dir = os.path.join(folder, f"impacts_{tag}_shots")
    os.makedirs(shots_dir, exist_ok=True)
    for old in glob.glob(os.path.join(shots_dir, "*.jpg")):   # clear stale shots
        os.remove(old)
    want = {e["frame"]: e for e in kept}
    rank = {e["frame"]: i for i, e in enumerate(
        sorted(kept, key=lambda x: -x["score"]), 1)}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[v9] cannot open {video_path}; skipping shots.")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[v9] saving {len(want)} impact screenshots -> {shots_dir}")
    saved = 0
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi not in want:
            continue
        ev = want[fi]
        Hf, Wf = frame.shape[:2]
        # union of both fighters' bboxes (padded) so the contact is in view
        boxes = []
        for pid in (ev["striker_id"], ev["receiver_id"]):
            e = persons.get(pid, {}).get(fi)
            if e:
                boxes.append(e["bbox"])
        if boxes:
            x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
            pw, ph = (x2 - x1) * 0.25, (y2 - y1) * 0.18
            x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            x2, y2 = min(Wf, int(x2 + pw)), min(Hf, int(y2 + ph))
        else:
            x1, y1, x2, y2 = 0, 0, Wf, Hf
        mk = _region_marker(persons.get(ev["receiver_id"], {}).get(fi),
                            ev["contact_region"])
        crop = frame[y1:y2, x1:x2].copy()
        if mk is not None:
            cv2.circle(crop, (mk[0] - x1, mk[1] - y1), 26, (0, 230, 0), 3, cv2.LINE_AA)
            cv2.circle(crop, (mk[0] - x1, mk[1] - y1), 6, (0, 255, 255), -1, cv2.LINE_AA)
        ch, cw = crop.shape[:2]
        if cw > max_w:
            crop = cv2.resize(crop, (max_w, int(ch * max_w / cw)))
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 26), (15, 15, 20), -1)
        cv2.putText(crop, f"#{rank[fi]:02d}  frame {fi}  t={fi/fps:.1f}s  "
                    f"score {ev['score']:.2f}", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        fn = os.path.join(shots_dir, f"{rank[fi]:02d}_frame{fi}_score{ev['score']:.2f}.jpg")
        cv2.imwrite(fn, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1
    cap.release()
    print(f"[v9] saved {saved} screenshots -> {shots_dir}")
    return shots_dir


def save_impact_strips(folder, video_path, persons, kept, fps, off=6, panel_w=300, tag="v9"):
    """Save a 3-panel strip per impact: [t-off | t(impact) | t+off], same crop box
    so the REACTION is visible (head snap / body fold = real hit; none = slip/clinch).
    This gives the VLM the temporal context a single still lacks."""
    import cv2
    strips_dir = os.path.join(folder, f"impacts_{tag}_strips")
    os.makedirs(strips_dir, exist_ok=True)
    for old in glob.glob(os.path.join(strips_dir, "*.jpg")):
        os.remove(old)
    rank = {e["frame"]: i for i, e in enumerate(
        sorted(kept, key=lambda x: -x["score"]), 1)}
    # collect every frame index we need
    need = set()
    for e in kept:
        for o in (-off, 0, off):
            need.add(e["frame"] + o)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[v9] cannot open {video_path}; skipping strips.")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Wf = int(cap.get(3)); Hf = int(cap.get(4))
    store = {}
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi in need:
            store[fi] = frame.copy()
    cap.release()

    saved = 0
    for e in kept:
        f = e["frame"]
        # crop box = union of both fighters' bboxes at impact (padded)
        boxes = [persons.get(p, {}).get(f, {}).get("bbox")
                 for p in (e["striker_id"], e["receiver_id"])]
        boxes = [b for b in boxes if b]
        if boxes:
            x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
            pw, ph = (x2 - x1) * 0.25, (y2 - y1) * 0.18
            x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            x2, y2 = min(Wf, int(x2 + pw)), min(Hf, int(y2 + ph))
        else:
            x1, y1, x2, y2 = 0, 0, Wf, Hf
        mk = _region_marker(persons.get(e["receiver_id"], {}).get(f),
                            e["contact_region"])
        panels = []
        for j, o in enumerate((-off, 0, off)):
            fr = store.get(f + o)
            if fr is None:
                fr = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
            else:
                fr = fr[y1:y2, x1:x2].copy()
            if o == 0 and mk is not None:    # marker only on impact panel
                cv2.circle(fr, (mk[0] - x1, mk[1] - y1), 24, (0, 230, 0), 3, cv2.LINE_AA)
            ph_, pw_ = fr.shape[:2]
            scale = panel_w / max(1, pw_)
            fr = cv2.resize(fr, (panel_w, int(ph_ * scale)))
            lbl = {-off: f"t-{off}", 0: "IMPACT", off: f"t+{off}"}[o]
            cv2.rectangle(fr, (0, 0), (panel_w, 22), (15, 15, 20), -1)
            cv2.putText(fr, lbl, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(fr)
        h = min(p.shape[0] for p in panels)
        strip = np.hstack([p[:h] for p in panels])
        cv2.rectangle(strip, (0, 0), (strip.shape[1], 20), (15, 15, 20), -1)
        cv2.putText(strip, f"#{rank[f]:02d}  frame {f}  t={f/fps:.1f}s  "
                    f"score {e['score']:.2f}  (left=before  mid=impact  right=after)",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        fn = os.path.join(strips_dir, f"{rank[f]:02d}_frame{f}_score{e['score']:.2f}.jpg")
        cv2.imwrite(fn, strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1
    print(f"[v9] saved {saved} impact strips -> {strips_dir}")
    return strips_dir


# ── per-fight runner ─────────────────────────────────────────────────────────
def find_files(folder):
    sam3d = glob.glob(os.path.join(folder, "*_sam3d.json"))
    fa = glob.glob(os.path.join(folder, "*_full_analysis.json"))
    vids = [v for v in glob.glob(os.path.join(folder, "*.mp4"))
            if "_visualized" not in os.path.basename(v)]
    if not sam3d or not fa:
        raise SystemExit(f"[v9] missing *_sam3d.json / *_full_analysis.json in {folder}")
    return sam3d[0], fa[0], (vids[0] if vids else None)

def run(folder, thr, cooldown, use_audio, top, make_video=False, freeze_sec=1.5,
        make_shots=False, out_folder=None):
    out_folder = out_folder or folder
    sam3d_p, fa_p, vid = find_files(folder)
    name = os.path.basename(os.path.basename(sam3d_p)).replace("_sam3d.json", "")
    print(f"\n[v9] === {os.path.basename(folder)} ===")
    print(f"[v9] sam3d={os.path.basename(sam3d_p)}  full_analysis={os.path.basename(fa_p)}")

    s = json.load(open(sam3d_p))
    fa = json.load(open(fa_p))
    persons = {0: {e["frame"]: e for e in s.get("0", [])},
               1: {e["frame"]: e for e in s.get("1", [])}}
    actions = fa["actions"]
    ce = fa.get("contact_events", [])
    fps = float(fa.get("processing_stats", {}).get("total_frames", 0)) / \
          float(fa.get("processing_stats", {}).get("video_duration", 1)) or 30.0
    print(f"[v9] {len(actions)} actions, {len(ce)} contact_events, fps~{fps:.3f}")

    ce_by_pair = defaultdict(list)
    for ev in ce:
        ce_by_pair[(ev["striker_id"], ev["receiver_id"])].append(ev)

    onset = load_audio_onset(vid, folder) if (use_audio and vid) else None

    scored = [r for r in (score_action(a, persons, ce_by_pair, onset, fps)
                          for a in actions) if r]
    scored.sort(key=lambda e: -e["score"])
    kept = nms([e for e in scored if e["score"] >= thr], cooldown)

    print(f"[v9] scored {len(scored)} candidates; {len(kept)} pass thr={thr} (cd={cooldown})")
    print(f"[v9] kept region mix: {dict(Counter(e['contact_region'] for e in kept))}")
    print(f"\n  rank  frame   t(s)   score  region   action            gap(sw)")
    print("  " + "-" * 64)
    for i, e in enumerate(sorted(kept, key=lambda x: -x["score"])[:top], 1):
        print(f"  {i:3d}  {e['frame']:6d}  {e['frame']/fps:6.2f}  {e['score']:.3f}  "
              f"{e['contact_region']:7s}  {e['action']:16s}  "
              f"{e['gates']['gap_shoulder_widths']:.2f}")

    out = {
        "approach": "fusion_v9",
        "label": "MHR70-correct world-XY region-aware contact (static, untrained)",
        "source_sam3d": os.path.basename(sam3d_p),
        "source_full_analysis": os.path.basename(fa_p),
        "fps": round(fps, 3),
        "threshold": thr, "cooldown": cooldown, "audio_used": onset is not None,
        "note": "Ranked impact CANDIDATES, not verified hits. No GT for these "
                "fights. world_coords Z is unreliable; gap uses X-Y plane only.",
        "n_candidates_scored": len(scored),
        "n_impacts": len(kept),
        "impacts": [{
            "is_impact": True, "impact_frame": e["frame"],
            "timestamp_seconds": round(e["frame"] / fps, 3),
            "impact_score": round(e["score"], 4),
            "contact_region": e["contact_region"],
            "striker_id": e["striker_id"], "receiver_id": e["receiver_id"],
            "action": e["action"],
            "gates": {k: round(v, 3) if isinstance(v, float) else v
                      for k, v in e["gates"].items()},
        } for e in kept],
        "all_scored_candidates": [{
            "impact_frame": e["frame"],
            "timestamp_seconds": round(e["frame"] / fps, 3),
            "impact_score": round(e["score"], 4),
            "contact_region": e["contact_region"], "action": e["action"],
            "striker_id": e["striker_id"], "receiver_id": e["receiver_id"],
            "window_start": e["window_start"], "window_end": e["window_end"],
        } for e in scored],
    }
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{name}_impacts_v9.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[v9] saved -> {out_path}")
    # mirror into repo outputs/ too
    repo_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs")
    os.makedirs(repo_out, exist_ok=True)
    mirror = os.path.join(repo_out, f"{os.path.basename(folder)}_impacts_v9.json")
    json.dump(out, open(mirror, "w"), indent=2)
    print(f"[v9] mirror  -> {os.path.abspath(mirror)}")
    if make_shots and vid:
        save_impact_shots(out_folder, vid, persons, kept, fps)
        save_impact_strips(out_folder, vid, persons, kept, fps)
    elif make_shots:
        print("[v9] no raw video found in folder; cannot save shots.")
    if make_video and vid:
        render_video(out_folder, vid, persons, kept, fps, freeze_sec)
    elif make_video:
        print("[v9] no raw video found in folder; cannot render.")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="fight folder (or parent of several)")
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--cooldown", type=int, default=18)
    ap.add_argument("--audio", action="store_true", help="enable audio onset gate (slower)")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--video", action="store_true", help="render annotated mp4")
    ap.add_argument("--freeze", type=float, default=1.5,
                    help="seconds to freeze each impact frame for review (0=off)")
    ap.add_argument("--shots", action="store_true",
                    help="save one annotated screenshot per impact")
    ap.add_argument("--out-folder", default=None,
                    help="write json/video/shots here instead of --folder (still reads source files from --folder)")
    args = ap.parse_args()

    # accept either a single fight folder or a parent containing several
    folders = []
    if glob.glob(os.path.join(args.folder, "*_sam3d.json")):
        folders = [args.folder]
    else:
        for sub in sorted(glob.glob(os.path.join(args.folder, "*"))):
            if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*_sam3d.json")):
                folders.append(sub)
    if not folders:
        raise SystemExit(f"[v9] no fight folders found under {args.folder}")
    for fdr in folders:
        run(fdr, args.thr, args.cooldown, args.audio, args.top, args.video,
            args.freeze, args.shots, out_folder=args.out_folder)


if __name__ == "__main__":
    main()
