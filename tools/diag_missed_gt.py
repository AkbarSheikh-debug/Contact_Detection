#!/usr/bin/env python3
"""Diagnose gate scores for the 4 GT frames not caught at thr=0.42."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
from collections import defaultdict

# ── inline the scoring helpers from v8.py ───────────────────────────────────
import numpy as np

WRIST_KPS = [9, 10]
HEAD_KPS  = [0, 1, 2, 3, 4]
TORSO_KPS = [5, 6, 11, 12]
GUARD_KPS = [7, 8, 9, 10]
FPS = 24.995

REGION_W = {
    "head": 1.00, "torso": 0.85,
    "left_arm": 0.20, "right_arm": 0.20,
    "left_leg": 0.30, "right_leg": 0.30,
}
WEIGHTS = dict(gap=0.28, ce_prob=0.10, region=0.08, decel=0.12,
               react=0.04, bbox_react=0.04, approach=0.06,
               guard=0.10, audio=0.02, conf=0.10)


def proj_kp(entry, k):
    nc = entry.get("normalized_coords")
    if nc is None or k >= len(nc): return None
    nc = np.asarray(nc)
    x1, y1, x2, y2 = entry["bbox"]
    xmin, xmax = nc[:, 0].min(), nc[:, 0].max()
    ymin, ymax = nc[:, 1].min(), nc[:, 1].max()
    if xmax <= xmin or ymax <= ymin: return None
    u = x1 + (nc[k, 0] - xmin) / (xmax - xmin) * (x2 - x1)
    v = y1 + (nc[k, 1] - ymin) / (ymax - ymin) * (y2 - y1)
    return np.array([u, v])

def centroid_2d(entry, kps):
    pts = [proj_kp(entry, k) for k in kps]
    pts = [p for p in pts if p is not None]
    return np.mean(pts, axis=0) if pts else None

def bbox_diag(entry):
    x1, y1, x2, y2 = entry["bbox"]
    return float(np.hypot(x2-x1, y2-y1)) + 1e-6

def wrist_decel(person_frames, fn, half=5):
    best = 0.0
    for wk in WRIST_KPS:
        traj = []
        for f in range(fn - half, fn + half + 1):
            e = person_frames.get(f)
            if e is None: continue
            p = proj_kp(e, wk)
            if p is not None: traj.append((f, p, bbox_diag(e)))
        if len(traj) < 4: continue
        fs = np.array([t[0] for t in traj], float)
        ps = np.array([t[1] for t in traj])
        diag = np.mean([t[2] for t in traj])
        dt = np.diff(fs); dt[dt == 0] = 1.0
        vel = np.linalg.norm(np.diff(ps, axis=0), axis=1) / dt / diag
        if len(vel) < 2 or vel.max() < 1e-6: continue
        acc = np.diff(vel)
        decel = max(0.0, -acc.min())
        best = max(best, min(1.0, (decel / vel.max()) / 0.6))
    return best

def head_reaction(person_frames, fn, half=4):
    def hc(f):
        e = person_frames.get(f)
        return centroid_2d(e, HEAD_KPS) if e else None
    a, b, c = hc(fn - half), hc(fn), hc(fn + half)
    if a is None or b is None or c is None: return 0.0
    diag = bbox_diag(person_frames[fn])
    acc = np.linalg.norm(c - 2 * b + a) / diag
    return min(1.0, acc / 0.25)

def bbox_react_score(receiver_frames, fn, n_after=8):
    pre_vels = []
    for f in range(fn - 2, fn + 1):
        e0, e1 = receiver_frames.get(f - 1), receiver_frames.get(f)
        if e0 is None or e1 is None: continue
        x10, y10, x20, y20 = e0["bbox"]; x11, y11, x21, y21 = e1["bbox"]
        bbox_h = max(1.0, y21 - y11)
        tc0 = np.array([(x10 + x20) / 2.0, y10])
        tc1 = np.array([(x11 + x21) / 2.0, y11])
        pre_vels.append(float(np.linalg.norm(tc1 - tc0) / bbox_h))
    baseline_vel = float(np.mean(pre_vels)) if pre_vels else 0.02
    peak_vel = 0.0
    for f in range(fn + 1, fn + n_after + 1):
        e0, e1 = receiver_frames.get(f - 1), receiver_frames.get(f)
        if e0 is None or e1 is None: continue
        x10, y10, x20, y20 = e0["bbox"]; x11, y11, x21, y21 = e1["bbox"]
        bbox_h = max(1.0, y21 - y11)
        tc0 = np.array([(x10 + x20) / 2.0, y10])
        tc1 = np.array([(x11 + x21) / 2.0, y11])
        vel = float(np.linalg.norm(tc1 - tc0) / bbox_h)
        if vel > peak_vel: peak_vel = vel
    ratio = peak_vel / max(baseline_vel, 0.005)
    return float(np.clip((ratio - 1.0) / 4.0, 0.0, 1.0))

def approach_rate(striker_frames, receiver_frames, fn, look=8):
    gaps = []
    for f in range(fn - look, fn + 1):
        se, re = striker_frames.get(f), receiver_frames.get(f)
        if se is None or re is None: continue
        rhead = centroid_2d(re, HEAD_KPS)
        if rhead is None: continue
        diag = bbox_diag(re)
        wd = [np.linalg.norm(proj_kp(se, wk) - rhead)
              for wk in WRIST_KPS if proj_kp(se, wk) is not None]
        if wd: gaps.append(min(wd) / diag)
    if len(gaps) < 4: return 0.0
    deltas = np.diff(gaps)
    return float((deltas < 0).mean())

def min_wrist_body_gap(striker_frames, receiver_frames, fn):
    se, re = striker_frames.get(fn), receiver_frames.get(fn)
    if se is None or re is None: return None, None
    diag = bbox_diag(re)
    head = [proj_kp(re, k) for k in HEAD_KPS]
    torso = [proj_kp(re, k) for k in TORSO_KPS]
    head = [p for p in head if p is not None]
    torso = [p for p in torso if p is not None]
    wr = [proj_kp(se, k) for k in WRIST_KPS]
    wr = [w for w in wr if w is not None]
    if not wr or not (head or torso): return None, None
    gh = min((np.linalg.norm(w - b) for w in wr for b in head), default=1e9)
    gt_ = min((np.linalg.norm(w - b) for w in wr for b in torso), default=1e9)
    if gh <= gt_: return gh / diag, "head"
    return gt_ / diag, "torso"

def _pt_seg_dist(p, a, b):
    ab = b - a; denom = np.dot(ab, ab)
    if denom < 1e-9: return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))

def shoulder_w_2d(entry):
    ls, rs = proj_kp(entry, 5), proj_kp(entry, 6)
    if ls is not None and rs is not None:
        return float(np.linalg.norm(ls - rs)) + 1e-6
    return bbox_diag(entry) * 0.30

def guard_interception(striker_frames, receiver_frames, fn, half=3):
    BLOCKED_DIST = 0.30; OPEN_DIST = 1.50
    min_d = 1e9
    for f in range(fn - half, fn + 1):
        se = striker_frames.get(f); re = receiver_frames.get(f)
        if se is None or re is None: continue
        head = [proj_kp(re, k) for k in HEAD_KPS]
        head = [p for p in head if p is not None]
        if not head: continue
        head_c = np.mean(head, axis=0)
        sw = shoulder_w_2d(re)
        wr_pts = [proj_kp(se, k) for k in WRIST_KPS]
        wr_pts = [w for w in wr_pts if w is not None]
        if not wr_pts: continue
        punch_wrist = min(wr_pts, key=lambda w: np.linalg.norm(w - head_c))
        for gk in GUARD_KPS:
            gp = proj_kp(re, gk)
            if gp is None: continue
            d = _pt_seg_dist(gp, punch_wrist, head_c) / sw
            if d < min_d: min_d = d
    if min_d >= 1e9: return 0.5
    return float(np.clip((min_d - BLOCKED_DIST) / (OPEN_DIST - BLOCKED_DIST), 0.0, 1.0))

def nearest_contact_event(ce_by_pair, sid, rid, fn, win=15):
    best = None
    for ev in ce_by_pair.get((sid, rid), []):
        if abs(ev["frame"] - fn) <= win:
            if best is None or ev["contact_prob"] > best["contact_prob"]:
                best = ev
    return best

def _score_frame(f, sid, rid, sp, rp, act_conf, ce_by_pair, onset):
    gap, reg = min_wrist_body_gap(sp, rp, f)
    if gap is None: return None
    s_gap = max(0.0, min(1.0, (0.45 - gap) / 0.35))
    ce = nearest_contact_event(ce_by_pair, sid, rid, f)
    if ce is not None:
        s_ce_prob = float(ce["contact_prob"])
        s_region  = REGION_W.get(ce["contact_region"], 0.5)
        reg       = ce["contact_region"]
    else:
        s_ce_prob = 0.0
        s_region  = REGION_W.get(reg, 0.6)
    s_decel      = wrist_decel(sp, f)
    s_react      = head_reaction(rp, f)
    s_bbox_react = bbox_react_score(rp, f)
    s_appr       = approach_rate(sp, rp, f)
    s_grd        = guard_interception(sp, rp, f)
    s_audio      = 0.0
    s_conf       = float(min(1.0, act_conf))
    fused = (WEIGHTS["gap"]          * s_gap
             + WEIGHTS["ce_prob"]    * s_ce_prob
             + WEIGHTS["region"]     * s_region
             + WEIGHTS["decel"]      * s_decel
             + WEIGHTS["react"]      * s_react
             + WEIGHTS["bbox_react"] * s_bbox_react
             + WEIGHTS["approach"]   * s_appr
             + WEIGHTS["guard"]      * s_grd
             + WEIGHTS["audio"]      * s_audio
             + WEIGHTS["conf"]       * s_conf)
    gates = dict(gap=s_gap, ce_prob=s_ce_prob, region=s_region, decel=s_decel,
                 react=s_react, bbox_react=s_bbox_react, approach=s_appr,
                 guard=s_grd, audio=s_audio, conf=s_conf)
    return fused, gates, reg


# ── Load data ────────────────────────────────────────────────────────────────
folder = r'C:\Users\XRIG\Downloads\sam3d_with_world_coords'
d = json.load(open(os.path.join(folder, 'fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json')))
persons = {}
for pid in ('0', '1'):
    persons[int(pid)] = {e['frame']: e for e in d.get(pid, [])}
ce_by_pair = defaultdict(list)
for ev in d.get('contact_events', []):
    ce_by_pair[(ev['striker_id'], ev['receiver_id'])].append(ev)

TARGET_GT = [856, 858, 2192, 3736]
print('Gate breakdown for 4 missed GT frames (both striker roles, conf=0.5):')
print(f'  weights: gap=0.28 ce=0.10 reg=0.08 dec=0.12 rea=0.04 bxr=0.04 app=0.06 grd=0.10 cnf=0.10')
print()
for gf in TARGET_GT:
    print(f'  -- GT {gf} --')
    for sid in (0, 1):
        rid = 1 - sid
        sp, rp = persons.get(sid, {}), persons.get(rid, {})
        result = _score_frame(gf, sid, rid, sp, rp, 0.5, ce_by_pair, None)
        if result is not None:
            fused, gates, reg = result
            print(f'    sid={sid}: score={fused:.4f}  reg={reg}')
            for k, v in gates.items():
                contrib = WEIGHTS[k] * v
                print(f'      {k:10s}: {v:.3f}  (contrib={contrib:.4f}  w={WEIGHTS[k]})')
        else:
            print(f'    sid={sid}: NO VALID GAP (missing keypoints)')
    print()

# Also show distribution of FP scores for comparison
print('='*60)
print('FP score comparison: checking frames in range 840-870 and 2180-2200')
for frame_range, label in [((840, 870), 'near 856/858'), ((2180, 2200), 'near 2192'), ((3720, 3750), 'near 3736')]:
    print(f'\n  Region {label} (frames {frame_range[0]}-{frame_range[1]}):')
    for f in range(frame_range[0], frame_range[1]+1):
        for sid in (0, 1):
            rid = 1 - sid
            sp, rp = persons.get(sid, {}), persons.get(rid, {})
            result = _score_frame(f, sid, rid, sp, rp, 0.5, ce_by_pair, None)
            if result is not None:
                fused, gates, reg = result
                marker = ' <-- GT' if f in TARGET_GT else ''
                if fused > 0.35:
                    print(f'    f={f} sid={sid}: {fused:.4f} gap={gates["gap"]:.2f} dec={gates["decel"]:.2f} app={gates["approach"]:.2f} grd={gates["guard"]:.2f}{marker}')
