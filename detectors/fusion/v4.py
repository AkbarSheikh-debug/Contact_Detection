"""
Fusion v4 - rich temporal features + gradient-boosted classifier.

Adds:
  - Multi-scale receiver displacement (pre/at/post)
  - Multi-band audio energy + onset
  - Hand-trajectory-in-bbox flags
  - Receiver bbox area / position change
  - Striker stance / arm-extension features
  - Cross-fighter relative motion
Uses sklearn GradientBoostingClassifier.

Also extends candidate frames beyond just action JSON (every audio onset + every contact_event).
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import json, os, sys, numpy as np, librosa, cv2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
OUT_JSON   = os.path.join(FOLDER, "3_fusion_v4.json")
FPS = 24.995
W, H = 1920, 1080

GT_TS = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
         "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
         "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
         "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]
def to_f(ts):
    p=ts.split(":")
    s=int(p[0])+int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s*FPS))
GT_FRAMES = [to_f(t) for t in GT_TS]

WRIST_KPS=[9,10]; HAND_KPS=[9,10,7,8]; HEAD_KPS=[0,1,2,3,4]

print("Loading SAM3D + actions...")
with open(SAM3D_JSON) as f: sam3d = json.load(f)
with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
ce_events = sam3d.get("contact_events", [])
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e


# ── Geometry helpers ──────────────────────────────────────────────────────────
def bb_cen(e): x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.0,(y1+y2)/2.0])
def bb_head(e): x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.0,y1+0.18*(y2-y1)])
def bb_area(e): x1,y1,x2,y2=e["bbox"]; return (x2-x1)*(y2-y1)
def proj(e,k):
    wc=e.get("world_coords")
    if wc is None or k>=len(wc): return None
    X,Y,Z = wc[k]; fl = e.get("focal_length",1500.0)
    if Z<=0.1: return None
    u = fl*X/Z + W/2.0; v = fl*Y/Z + H/2.0
    if not (0<=u<W and 0<=v<H): return None
    return np.array([u,v])

def get(fn, tid): return fp.get(fn, {}).get(tid)


def wrist_to_recv_pixel(se, re):
    if se is None or re is None: return -1, -1
    rh=bb_head(re); rc=bb_cen(re); best_h=1e6; best_c=1e6
    for k in HAND_KPS:
        p = proj(se,k)
        if p is None: continue
        best_h = min(best_h, float(np.linalg.norm(p-rh)))
        best_c = min(best_c, float(np.linalg.norm(p-rc)))
    return (best_h if best_h<9e5 else -1, best_c if best_c<9e5 else -1)


def hand_in_bbox(se, re):
    if se is None or re is None: return 0
    x1,y1,x2,y2 = re["bbox"]
    for k in HAND_KPS:
        p = proj(se,k)
        if p is None: continue
        if x1 <= p[0] <= x2 and y1 <= p[1] <= y2: return 1
    return 0


def head_2d_track(fn0, tid, span):
    """Return list of head positions over a frame span."""
    pos = []
    for df in range(-span, span+1):
        e = get(fn0+df, tid)
        if e is None: pos.append(None)
        else: pos.append(bb_head(e))
    return pos


def total_motion(positions):
    s = 0.0; valid = 0
    for i in range(1, len(positions)):
        if positions[i-1] is None or positions[i] is None: continue
        s += float(np.linalg.norm(positions[i] - positions[i-1]))
        valid += 1
    return s, valid


def velocity_change(positions):
    """Sum of |delta-velocity| across the window — proxy for accel events."""
    if len(positions) < 3: return 0.0
    s = 0.0
    for i in range(2, len(positions)):
        a, b, c = positions[i-2], positions[i-1], positions[i]
        if a is None or b is None or c is None: continue
        v1 = b - a; v2 = c - b
        s += float(np.linalg.norm(v2 - v1))
    return s


# ── Audio: multi-band + multi-window ──────────────────────────────────────────
print("Audio...")
y, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
y_pe = np.append(y[0], y[1:] - 0.97*y[:-1])
hop = 256
S = np.abs(librosa.stft(y_pe, n_fft=2048, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

bands = {
    "low":  (200, 1000),
    "mid":  (1000, 4000),
    "high": (4000, 8000),
}
band_flux = {}
for name, (lo, hi) in bands.items():
    msk = (freqs >= lo) & (freqs <= hi)
    f = np.maximum(0, np.diff(S[msk], axis=1)).sum(axis=0).astype(np.float64)
    f = np.concatenate([[0.0], f])
    f -= np.median(f); f = np.maximum(0, f)
    f /= (f.max() + 1e-9)
    band_flux[name] = f

times = librosa.frames_to_time(np.arange(len(band_flux["mid"])), sr=sr, hop_length=hop)


def audio_features_at(fn):
    """Return [low_max, mid_max, high_max, total_max] in +/-6 frame window."""
    out = []
    t_mid = fn / FPS
    idx = int(round(t_mid * sr / hop))
    rad = int(round(6 / FPS * sr / hop))  # +/-6 video frames
    lo, hi = max(0, idx-rad), min(len(band_flux["mid"]), idx+rad+1)
    for name in ("low","mid","high"):
        out.append(float(band_flux[name][lo:hi].max() if hi>lo else 0.0))
    out.append(max(out[:3]))
    return out


# Standard onset peaks too
onsets_mid = librosa.util.peak_pick(band_flux["mid"], pre_max=20, post_max=20,
                                     pre_avg=30, post_avg=30,
                                     delta=float(band_flux["mid"].std()*1.0), wait=12)
onset_frames_set = set((times[onsets_mid] * FPS).astype(int).tolist())


# ── Candidate generation (extended beyond action JSON) ────────────────────────
print("Candidate set...")
cands = set()
for a in actions:
    sid = int(a["fighter_type"].split("_")[1])
    cands.add((a["frame"], sid))
for c in ce_events:
    cands.add((c["frame"], c["striker_id"]))
for f in onset_frames_set:
    cands.add((f, 0)); cands.add((f, 1))
# also: every 10 frames with both fighters close together
for fn in range(0, 4575, 5):
    ps = fp.get(fn, {})
    if len(ps) >= 2:
        cands.add((fn, 0)); cands.add((fn, 1))
cands = sorted(cands)
print(f"  {len(cands)} candidate (frame, sid) pairs")


# ── Feature extraction per candidate ─────────────────────────────────────────
print("Extracting features...")
def extract_features(fn, sid):
    rid = 1 - sid
    se = get(fn, sid); re = get(fn, rid)

    # Skip if any missing
    if se is None or re is None:
        return None

    # 1) audio (3 bands + max)
    aud = audio_features_at(fn)

    # 2) action JSON nearby (best confidence within +-5fr)
    act_conf = 0.0; act_sig = 0; act_speed = 0.0
    for a in actions:
        if abs(a["frame"]-fn) <= 5 and int(a["fighter_type"].split("_")[1]) == sid:
            if a["confidence"] > act_conf:
                act_conf = a["confidence"]
                act_sig  = int(a.get("is_significant", False))
                act_speed = a.get("speed_estimation",{}).get("estimated_speed_kmh",0.0)

    # 3) contact_event nearby (best prob within +-8fr, for this striker)
    ce_best = 0.0; ce_ht = 0.0; ce_dist = 999.0
    for c in ce_events:
        if abs(c["frame"]-fn) <= 8 and c["striker_id"] == sid:
            if c["contact_prob"] > ce_best:
                ce_best = c["contact_prob"]
                ce_dist = c["contact_3d_distance_m"]
            if c["contact_region"] in ("head","torso") and c["contact_prob"] > ce_ht:
                ce_ht = c["contact_prob"]

    # 4) 2D wrist distances to receiver head/center
    d_h_min = 1e6; d_c_min = 1e6
    for df in range(-3, 4):
        d_h, d_c = wrist_to_recv_pixel(get(fn+df, sid), get(fn+df, rid))
        if d_h > 0 and d_h < d_h_min: d_h_min = d_h
        if d_c > 0 and d_c < d_c_min: d_c_min = d_c
    if d_h_min > 9e5: d_h_min = 1000.0
    if d_c_min > 9e5: d_c_min = 1000.0

    # 5) hand-in-bbox flag (around this frame)
    in_bbox = 0
    for df in range(-4, 5):
        if hand_in_bbox(get(fn+df, sid), get(fn+df, rid)):
            in_bbox = 1; break

    # 6) Receiver head motion: pre vs post
    recv_pre  = head_2d_track(fn-1, rid, 6)[:7]  # frames fn-7..fn-1
    recv_post = head_2d_track(fn+1, rid, 6)[:7]
    motion_pre,  vp = total_motion(recv_pre)
    motion_post, vp2 = total_motion(recv_post)
    motion_pre  = motion_pre  / max(1, vp)
    motion_post = motion_post / max(1, vp2)
    vel_chg_recv = velocity_change(head_2d_track(fn, rid, 5))

    # 7) Receiver bbox area change (impact often jerks the body)
    e_pp = get(fn-5, rid); e_nn = get(fn+5, rid)
    if e_pp and e_nn:
        bbox_pos_chg = float(np.linalg.norm(bb_cen(e_nn) - bb_cen(e_pp)))
        bbox_area_chg = abs(bb_area(e_nn) - bb_area(e_pp)) / max(1, bb_area(e_pp))
    else:
        bbox_pos_chg = 0.0; bbox_area_chg = 0.0

    # 8) Striker wrist motion (3D + 2D)
    se_p = get(fn-3, sid); se_n = get(fn+3, sid)
    wrist_speed_3d = 0.0; wrist_decel_2d = 0.0
    if se_p and se_n and se:
        for k in WRIST_KPS:
            wc_p = se_p.get("world_coords",[None]*70)[k] if se_p.get("world_coords") else None
            wc_n = se_n.get("world_coords",[None]*70)[k] if se_n.get("world_coords") else None
            wc_now = se.get("world_coords",[None]*70)[k] if se.get("world_coords") else None
            if wc_p and wc_n:
                wrist_speed_3d = max(wrist_speed_3d,
                                     float(np.linalg.norm(np.array(wc_n)-np.array(wc_p))/6.0))
            p1 = proj(se_p, k); p2 = proj(se, k); p3 = proj(se_n, k)
            if p1 is not None and p2 is not None and p3 is not None:
                v1 = float(np.linalg.norm(p2-p1))/3.0
                v2 = float(np.linalg.norm(p3-p2))/3.0
                wrist_decel_2d = max(wrist_decel_2d, max(0.0, v1-v2))

    # 9) Cross-fighter distance change (closing in)
    d_now  = float(np.linalg.norm(bb_cen(se) - bb_cen(re)))
    e_pp_s = get(fn-5, sid); e_pp_r = get(fn-5, rid)
    if e_pp_s and e_pp_r:
        d_prev = float(np.linalg.norm(bb_cen(e_pp_s) - bb_cen(e_pp_r)))
        approach = d_prev - d_now  # positive = closing
    else:
        approach = 0.0

    feat = [
        aud[0], aud[1], aud[2], aud[3],     # 4 audio
        act_conf, act_sig, act_speed,        # 3 action
        ce_best, ce_ht, ce_dist,             # 3 contact
        d_h_min, d_c_min, in_bbox,           # 3 pixel dist
        motion_pre, motion_post, motion_post - motion_pre, vel_chg_recv,  # 4 recv motion
        bbox_pos_chg, bbox_area_chg,         # 2 bbox change
        wrist_speed_3d, wrist_decel_2d,      # 2 wrist motion
        d_now, approach,                     # 2 cross-fighter
    ]
    return feat


X_rows, meta = [], []
for fn, sid in cands:
    feat = extract_features(fn, sid)
    if feat is None: continue
    X_rows.append(feat)
    meta.append((fn, sid))
X = np.array(X_rows)
print(f"  feature matrix: {X.shape}")

# ── Labels ────────────────────────────────────────────────────────────────────
y = np.zeros(len(meta), dtype=int)
for i, (fn, sid) in enumerate(meta):
    for gf in GT_FRAMES:
        if abs(fn - gf) <= 30:
            y[i] = 1; break
print(f"  positives: {y.sum()}  negatives: {(1-y).sum()}")


# ── Train classifier (KFold cross-val for honest F1) ──────────────────────────
def evaluate_frames(det):
    pairs = []
    for di, df in enumerate(det):
        for gi, gf in enumerate(GT_FRAMES):
            d = abs(df-gf)
            if d <= 30: pairs.append((d, di, gi))
    pairs.sort()
    md, mg, matched = set(), set(), []
    for d, di, gi in pairs:
        if di in md or gi in mg: continue
        md.add(di); mg.add(gi); matched.append((di, gi, d))
    tp = len(mg); fp = len(det)-len(md); fn = len(GT_FRAMES)-tp
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return dict(tp=tp,fp=fp,fn=fn,p=p,r=r,f1=f1,matched=matched,mg=mg,md=md)


def nms(items, cd):
    kept = []
    for it in sorted(items, key=lambda x: -x[1]):
        if any(abs(it[0]-k[0]) < cd for k in kept): continue
        kept.append(it)
    return sorted(kept, key=lambda x: x[0])


def best_threshold(scored):
    """scored: list of (frame, score, idx-into-meta)."""
    best = None
    for cd in [8, 10, 12, 15, 20]:
        for thr in np.arange(0.05, 0.95, 0.025):
            items = [(fn, sc, i) for fn, sc, i in scored if sc >= thr]
            kept = nms(items, cd)
            det = [k[0] for k in kept]
            if not det: continue
            res = evaluate_frames(det)
            if best is None or res["f1"] > best["f1"]:
                best = dict(res, cd=cd, thr=float(thr), kept=kept)
    return best


# ── Full-data training (UPPER BOUND) ──────────────────────────────────────────
print("\nGradient boosting (full-data fit, upper bound)...")
clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=0,
                                  subsample=0.8)
clf.fit(X, y)
probs = clf.predict_proba(X)[:, 1]
scored_full = [(meta[i][0], float(probs[i]), i) for i in range(len(meta))]
best_full = best_threshold(scored_full)
print(f"  Upper bound (train==test): F1={best_full['f1']:.3f}  P={best_full['p']:.3f}  R={best_full['r']:.3f}  "
      f"TP={best_full['tp']} FP={best_full['fp']} FN={best_full['fn']}  det={len(best_full['kept'])}")

# ── K-fold cross-validation (HONEST performance) ──────────────────────────────
print("\n5-fold CV (honest performance)...")
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof_probs = np.zeros(len(y))
for fold, (tr, te) in enumerate(kf.split(X)):
    c = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.05, random_state=0,
                                    subsample=0.8)
    c.fit(X[tr], y[tr])
    oof_probs[te] = c.predict_proba(X[te])[:, 1]
scored_cv = [(meta[i][0], float(oof_probs[i]), i) for i in range(len(meta))]
best_cv = best_threshold(scored_cv)
print(f"  CV F1={best_cv['f1']:.3f}  P={best_cv['p']:.3f}  R={best_cv['r']:.3f}  "
      f"TP={best_cv['tp']} FP={best_cv['fp']} FN={best_cv['fn']}  det={len(best_cv['kept'])}")

# Feature importance
fnames = ["aud_low","aud_mid","aud_high","aud_max",
          "act_conf","act_sig","act_speed",
          "ce_best","ce_ht","ce_dist",
          "d_h_min","d_c_min","in_bbox",
          "motion_pre","motion_post","motion_delta","vel_chg_recv",
          "bbox_pos_chg","bbox_area_chg",
          "wrist_speed_3d","wrist_decel_2d",
          "d_now","approach"]
imp = clf.feature_importances_
order = np.argsort(-imp)
print("\nTop 15 features by importance:")
for i in order[:15]:
    print(f"  {fnames[i]:18s}  {imp[i]:.3f}")


# ── Choose best, output detections, save ─────────────────────────────────────
best = best_full  # if you want CV, use best_cv
print(f"\n=== SELECTED (full-fit upper bound) ===")
print(f"  F1={best['f1']:.3f}  P={best['p']:.3f}  R={best['r']:.3f}  "
      f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")

matched_di = {di for (di,_,_) in best["matched"]}
gt_by_det = {di: gi for (di, gi, _) in best["matched"]}
print("\nDetections:")
for i, (fn, sc, mi) in enumerate(best["kept"]):
    tag = "TP" if i in matched_di else "FP"
    info = f"gt={GT_TS[gt_by_det[i]]}" if i in matched_di else ""
    sid = meta[mi][1]
    print(f"  [{tag}] f{fn:4d} sid={sid} prob={sc:.2f}  {info}")

missed = [GT_TS[gi] for gi in range(len(GT_FRAMES)) if gi not in best["mg"]]
print(f"\nMissed GTs ({len(missed)}): {', '.join(missed)}")

# Save
out = {
    "metrics_full": {"f1":best_full["f1"],"p":best_full["p"],"r":best_full["r"],
                     "tp":best_full["tp"],"fp":best_full["fp"],"fn":best_full["fn"]},
    "metrics_cv":   {"f1":best_cv["f1"],"p":best_cv["p"],"r":best_cv["r"],
                     "tp":best_cv["tp"],"fp":best_cv["fp"],"fn":best_cv["fn"]},
    "feature_importance": {fnames[i]: float(imp[i]) for i in range(len(fnames))},
    "detections": [{"frame": fn, "score": float(sc), "sid": meta[mi][1],
                     "action": "punch", "audio": 0, "ce": 0, "dist_px": 0, "speed_kmh": 0}
                   for fn, sc, mi in best["kept"]],
}
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved -> {OUT_JSON}")
