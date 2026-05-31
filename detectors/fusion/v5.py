"""
Fusion v5 - uses relabeled GT (if available) + saves full contact metadata:
  - contact_frame : true impact frame (min wrist->receiver distance in +-12fr)
  - hit_xy        : 2D pixel [x,y] of the hit point on receiver
  - contact_region: head / torso / arm (from nearest contact_event)
  - action_type   : jab/cross/hook/uppercut from action JSON
  - speed_kmh     : from action JSON
  - audio_max     : multi-band audio peak
  - ce_prob       : best contact_event probability nearby

Everything else is identical to fusion_v4.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import json, os, sys, numpy as np, librosa
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
RELABEL_JSON = os.path.join(FOLDER, "relabeled_gt.json")
OUT_JSON   = os.path.join(FOLDER, "3_fusion_v5.json")
FPS = 24.995
W, H = 1920, 1080

# ── GT frames (prefer relabeled if available) ─────────────────────────────
GT_TS_ORIG = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
              "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
              "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
              "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]

def to_f(ts):
    p = ts.split(":")
    s = int(p[0])+int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s*FPS))

def to_ts(fn):
    t = fn / FPS
    m = int(t // 60); s = t - m*60
    si = int(s); fr = int(round((s-si)*FPS))
    return f"{m}:{si:02d}:{fr:02d}"

GT_TS_ORIG_FRAMES = [to_f(ts) for ts in GT_TS_ORIG]

# Load relabeled GT if it exists
if os.path.exists(RELABEL_JSON):
    with open(RELABEL_JSON) as f:
        relabeled = json.load(f)
    GT_FRAMES = [r["suggested_frame"] for r in relabeled]
    GT_TS     = [r["suggested_ts"]    for r in relabeled]
    print(f"Using RELABELED GT ({len(GT_FRAMES)} entries)")
    shifted = sum(1 for r in relabeled if abs(r["delta_frames"]) > 5)
    print(f"  {shifted} GTs shifted >5 frames from original label")
else:
    GT_FRAMES = GT_TS_ORIG_FRAMES
    GT_TS     = GT_TS_ORIG
    print("Using ORIGINAL GT")

WRIST_KPS = [9, 10]
ELBOW_KPS = [7, 8]
HAND_KPS  = [9, 10, 7, 8]
HEAD_KPS  = [0, 1, 2, 3, 4]
TORSO_KPS = [5, 6, 11, 12]
ALL_RECV_KPS = HEAD_KPS + TORSO_KPS

# NOTE: world_coords are in SMPL body-model space (NOT camera image space).
# proj() via fl*X/Z+W/2 is WRONG for this data — it places keypoints far
# outside the person's bbox. All distance/location computation uses bboxes.

print("Loading SAM3D + actions...")
with open(SAM3D_JSON) as f: sam3d = json.load(f)
with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
ce_events = sam3d.get("contact_events", [])
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e


# ── Geometry helpers ──────────────────────────────────────────────────────
def bb_cen(e):  x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.0,(y1+y2)/2.0])
def bb_head(e): x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.0,y1+0.18*(y2-y1)])
def bb_area(e): x1,y1,x2,y2=e["bbox"]; return (x2-x1)*(y2-y1)
def get(fn, tid): return fp.get(fn, {}).get(tid)

def proj_kp(e, k):
    """Map normalized_coords keypoint k to image (u,v) using bbox as linear reference.
    Verified: nose, shoulders, hips, ankles all land correctly inside the person's bbox."""
    nc = e.get("normalized_coords")
    if nc is None or k >= len(nc): return None
    nc_arr = np.array(nc)
    x1,y1,x2,y2 = e["bbox"]
    Y_min,Y_max = nc_arr[:,1].min(), nc_arr[:,1].max()
    X_min,X_max = nc_arr[:,0].min(), nc_arr[:,0].max()
    if Y_max <= Y_min or X_max <= X_min: return None
    u = x1 + (nc_arr[k,0]-X_min)/(X_max-X_min)*(x2-x1)
    v = y1 + (nc_arr[k,1]-Y_min)/(Y_max-Y_min)*(y2-y1)
    return np.array([u, v])

def fist_pos(se, re):
    """Striker's wrist/fist position in 2D. Uses proj_kp (normalized_coords)
    for accuracy; falls back to leading bbox edge if keypoints unavailable."""
    if se is None or re is None: return None
    # Try actual wrist keypoints first
    for k in WRIST_KPS:
        p = proj_kp(se, k)
        if p is not None: return p
    # Fallback: leading bbox edge
    sx1,sy1,sx2,sy2 = se["bbox"]
    rx1,ry1,rx2,ry2 = re["bbox"]
    sc_x = (sx1+sx2)/2; rc_x = (rx1+rx2)/2
    fx = sx2 if sc_x < rc_x else sx1
    fy = sy1 + 0.28*(sy2-sy1)
    return np.array([fx, fy])


# ── True contact frame + hit point using actual keypoints ─────────────────
def find_contact(fn, sid, scan=12):
    """
    Timing : frame in [fn-scan, fn+scan] where striker's wrist (proj_kp) is
             nearest to receiver's head keypoints (proj_kp).
    Location: the contact_events receiver_keypoint projected via proj_kp.
              Falls back to receiver nose/head area if no contact_event.
    Returns (contact_fn, hit_xy [x,y], dist_px).
    """
    rid = 1 - sid
    best_dist = 999.0; best_fn = fn

    for df in range(-scan, scan+1):
        cfn = fn + df
        se = get(cfn, sid); re = get(cfn, rid)
        if se is None or re is None: continue
        # Striker wrist positions (actual keypoints)
        for sk in WRIST_KPS:
            sp = proj_kp(se, sk)
            if sp is None: continue
            # Receiver head keypoints
            for rk in HEAD_KPS:
                rp = proj_kp(re, rk)
                if rp is None: continue
                d = float(np.linalg.norm(sp - rp))
                if d < best_dist:
                    best_dist = d; best_fn = cfn

    # Find nearest contact_event → get receiver_keypoint for exact hit location
    ce_region = None; recv_kp_idx = None; ce_best_prob = 0.0
    for c in ce_events:
        if abs(c["frame"]-fn) <= 15 and c["striker_id"] == sid:
            if c["contact_prob"] > ce_best_prob:
                ce_best_prob   = c["contact_prob"]
                ce_region      = c.get("contact_region")
                recv_kp_idx    = c.get("receiver_keypoint")

    re_c = get(best_fn, rid) or get(fn, rid)
    hit_xy = None
    if re_c is not None:
        # Try exact keypoint from contact_event
        if recv_kp_idx is not None:
            hit_xy = proj_kp(re_c, recv_kp_idx)
        # Fallback: project nose keypoint (kp 0) — always on the face
        if hit_xy is None:
            hit_xy = proj_kp(re_c, 0)
        # Last resort: bbox upper area
        if hit_xy is None:
            x1,y1,x2,y2 = re_c["bbox"]
            hit_xy = np.array([(x1+x2)/2, y1+0.18*(y2-y1)])

    return best_fn, hit_xy, best_dist, ce_region


# ── Wrist/hand proximity helpers (actual keypoints via proj_kp) ───────────
def wrist_to_recv_pixel(se, re):
    """Min distance from striker's projected wrist to receiver's head/center keypoints."""
    if se is None or re is None: return -1, -1
    rh_kp = proj_kp(re, 0)   # receiver nose
    rc    = bb_cen(re)
    rh    = rh_kp if rh_kp is not None else bb_head(re)
    best_h = 1e6; best_c = 1e6
    for k in WRIST_KPS:
        sp = proj_kp(se, k)
        if sp is None: continue
        best_h = min(best_h, float(np.linalg.norm(sp - rh)))
        best_c = min(best_c, float(np.linalg.norm(sp - rc)))
    return (best_h if best_h < 9e5 else -1, best_c if best_c < 9e5 else -1)

def hand_in_bbox(se, re):
    """Check if striker's projected wrist is inside receiver's bbox."""
    if se is None or re is None: return 0
    x1,y1,x2,y2 = re["bbox"]
    for k in WRIST_KPS:
        p = proj_kp(se, k)
        if p is None: continue
        if x1 <= p[0] <= x2 and y1 <= p[1] <= y2: return 1
    return 0

def head_2d_track(fn0, tid, span):
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
        s += float(np.linalg.norm(positions[i] - positions[i-1])); valid += 1
    return s, valid

def velocity_change(positions):
    if len(positions) < 3: return 0.0
    s = 0.0
    for i in range(2, len(positions)):
        a,b,c = positions[i-2],positions[i-1],positions[i]
        if a is None or b is None or c is None: continue
        s += float(np.linalg.norm((c-b)-(b-a)))
    return s


# ── Audio ──────────────────────────────────────────────────────────────────
print("Audio...")
y_a, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
y_pe = np.append(y_a[0], y_a[1:] - 0.97*y_a[:-1])
hop = 256
S = np.abs(librosa.stft(y_pe, n_fft=2048, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
bands = {"low":(200,1000), "mid":(1000,4000), "high":(4000,8000)}
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
    t_mid = fn / FPS
    idx = int(round(t_mid * sr / hop))
    rad = int(round(6 / FPS * sr / hop))
    lo, hi = max(0, idx-rad), min(len(band_flux["mid"]), idx+rad+1)
    out = [float(band_flux[n][lo:hi].max() if hi>lo else 0.0) for n in ("low","mid","high")]
    out.append(max(out[:3]))
    return out

onsets_mid = librosa.util.peak_pick(band_flux["mid"], pre_max=20, post_max=20,
                                     pre_avg=30, post_avg=30,
                                     delta=float(band_flux["mid"].std()*1.0), wait=12)
onset_frames_set = set((times[onsets_mid] * FPS).astype(int).tolist())


# ── Candidate generation ───────────────────────────────────────────────────
print("Candidate set...")
cands = set()
for a in actions:
    sid = int(a["fighter_type"].split("_")[1])
    cands.add((a["frame"], sid))
for c in ce_events:
    cands.add((c["frame"], c["striker_id"]))
for f in onset_frames_set:
    cands.add((f, 0)); cands.add((f, 1))
for fn in range(0, 4575, 5):
    ps = fp.get(fn, {})
    if len(ps) >= 2:
        cands.add((fn, 0)); cands.add((fn, 1))
cands = sorted(cands)
print(f"  {len(cands)} candidate (frame, sid) pairs")


# ── Feature extraction ─────────────────────────────────────────────────────
print("Extracting features...")
def extract_features(fn, sid):
    rid = 1 - sid
    se = get(fn, sid); re = get(fn, rid)
    if se is None or re is None: return None

    aud = audio_features_at(fn)

    act_conf = 0.0; act_sig = 0; act_speed = 0.0; act_type = "punch"
    for a in actions:
        if abs(a["frame"]-fn) <= 5 and int(a["fighter_type"].split("_")[1]) == sid:
            if a["confidence"] > act_conf:
                act_conf  = a["confidence"]
                act_sig   = int(a.get("is_significant", False))
                act_speed = a.get("speed_estimation",{}).get("estimated_speed_kmh",0.0)
                act_type  = a.get("action","punch")

    ce_best = 0.0; ce_ht = 0.0; ce_dist = 999.0
    for c in ce_events:
        if abs(c["frame"]-fn) <= 8 and c["striker_id"] == sid:
            if c["contact_prob"] > ce_best:
                ce_best = c["contact_prob"]; ce_dist = c["contact_3d_distance_m"]
            if c["contact_region"] in ("head","torso") and c["contact_prob"] > ce_ht:
                ce_ht = c["contact_prob"]

    d_h_min = 1e6; d_c_min = 1e6
    for df in range(-3, 4):
        d_h, d_c = wrist_to_recv_pixel(get(fn+df, sid), get(fn+df, rid))
        if d_h > 0 and d_h < d_h_min: d_h_min = d_h
        if d_c > 0 and d_c < d_c_min: d_c_min = d_c
    if d_h_min > 9e5: d_h_min = 1000.0
    if d_c_min > 9e5: d_c_min = 1000.0

    in_bbox = 0
    for df in range(-4, 5):
        if hand_in_bbox(get(fn+df, sid), get(fn+df, rid)):
            in_bbox = 1; break

    recv_pre  = head_2d_track(fn-1, rid, 6)[:7]
    recv_post = head_2d_track(fn+1, rid, 6)[:7]
    motion_pre,  vp  = total_motion(recv_pre)
    motion_post, vp2 = total_motion(recv_post)
    motion_pre  = motion_pre  / max(1, vp)
    motion_post = motion_post / max(1, vp2)
    vel_chg_recv = velocity_change(head_2d_track(fn, rid, 5))

    e_pp = get(fn-5, rid); e_nn = get(fn+5, rid)
    if e_pp and e_nn:
        bbox_pos_chg  = float(np.linalg.norm(bb_cen(e_nn) - bb_cen(e_pp)))
        bbox_area_chg = abs(bb_area(e_nn) - bb_area(e_pp)) / max(1, bb_area(e_pp))
    else:
        bbox_pos_chg = 0.0; bbox_area_chg = 0.0

    se_p = get(fn-3, sid); se_n = get(fn+3, sid)
    wrist_speed_3d = 0.0; wrist_decel_2d = 0.0
    if se_p and se_n and se:
        # 3D wrist speed from world_coords (same-person, so Z is reliable within one person)
        for k in WRIST_KPS:
            wc_p = se_p.get("world_coords",[None]*70)[k] if se_p.get("world_coords") else None
            wc_n = se_n.get("world_coords",[None]*70)[k] if se_n.get("world_coords") else None
            if wc_p and wc_n:
                wrist_speed_3d = max(wrist_speed_3d,
                                     float(np.linalg.norm(np.array(wc_n)-np.array(wc_p))/6.0))
        # 2D fist deceleration: striker fist approaching then stopping at contact
        fp_p = fist_pos(se_p, get(fn-3, rid))
        fp_0 = fist_pos(se,   get(fn,   rid))
        fp_n = fist_pos(se_n, get(fn+3, rid))
        if fp_p is not None and fp_0 is not None and fp_n is not None:
            v1 = float(np.linalg.norm(fp_0-fp_p))/3.0
            v2 = float(np.linalg.norm(fp_n-fp_0))/3.0
            wrist_decel_2d = max(0.0, v1-v2)

    d_now = float(np.linalg.norm(bb_cen(se) - bb_cen(re)))
    e_pp_s = get(fn-5, sid); e_pp_r = get(fn-5, rid)
    approach = (float(np.linalg.norm(bb_cen(e_pp_s) - bb_cen(e_pp_r))) - d_now
                if e_pp_s and e_pp_r else 0.0)

    feat = [
        aud[0], aud[1], aud[2], aud[3],
        act_conf, act_sig, act_speed,
        ce_best, ce_ht, ce_dist,
        d_h_min, d_c_min, in_bbox,
        motion_pre, motion_post, motion_post - motion_pre, vel_chg_recv,
        bbox_pos_chg, bbox_area_chg,
        wrist_speed_3d, wrist_decel_2d,
        d_now, approach,
    ]
    return feat


# Also store richer per-candidate metadata for saving to JSON
def candidate_meta(fn, sid):
    """Returns dict of rich metadata for a confirmed detection."""
    rid = 1 - sid
    aud = audio_features_at(fn)

    act_conf = 0.0; act_speed = 0.0; act_type = "punch"
    for a in actions:
        if abs(a["frame"]-fn) <= 8 and int(a["fighter_type"].split("_")[1]) == sid:
            if a["confidence"] > act_conf:
                act_conf  = a["confidence"]
                act_speed = a.get("speed_estimation",{}).get("estimated_speed_kmh",0.0)
                act_type  = a.get("action","punch")

    ce_best = 0.0; ce_region = "unknown"
    for c in ce_events:
        if abs(c["frame"]-fn) <= 10 and c["striker_id"] == sid:
            if c["contact_prob"] > ce_best:
                ce_best   = c["contact_prob"]
                ce_region = c.get("contact_region","unknown")

    # True contact timing + hit location (keypoint-accurate)
    contact_fn, hit_rp, dist_px, ce_region2 = find_contact(fn, sid, scan=12)
    if ce_region2: ce_region = ce_region2   # prefer find_contact's region

    return {
        "action":         act_type,
        "speed_kmh":      round(act_speed, 1),
        "audio":          round(float(max(aud[:3])), 3),
        "ce":             round(float(ce_best), 3),
        "contact_region": ce_region,
        "dist_px":        round(float(dist_px), 1),
        "contact_frame":  contact_fn,
        "hit_xy":         [round(float(hit_rp[0]),1), round(float(hit_rp[1]),1)] if hit_rp is not None else None,
    }


X_rows, meta = [], []
for fn, sid in cands:
    feat = extract_features(fn, sid)
    if feat is None: continue
    X_rows.append(feat)
    meta.append((fn, sid))
X = np.array(X_rows)
print(f"  feature matrix: {X.shape}")

y = np.zeros(len(meta), dtype=int)
for i, (fn, sid) in enumerate(meta):
    for gf in GT_FRAMES:
        if abs(fn - gf) <= 30:
            y[i] = 1; break
print(f"  positives: {y.sum()}  negatives: {(1-y).sum()}")


# ── Evaluation helpers ─────────────────────────────────────────────────────
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
    best = None
    for cd in [12, 15, 20, 25, 30]:
        for thr in np.arange(0.05, 0.95, 0.025):
            items = [(fn, sc, i) for fn, sc, i in scored if sc >= thr]
            kept  = nms(items, cd)
            det   = [k[0] for k in kept]
            if not det: continue
            res = evaluate_frames(det)
            if best is None or res["f1"] > best["f1"]:
                best = dict(res, cd=cd, thr=float(thr), kept=kept)
    return best


# ── Full-data training ─────────────────────────────────────────────────────
print("\nGradient boosting (full-data fit)...")
clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=0, subsample=0.8)
clf.fit(X, y)
probs = clf.predict_proba(X)[:, 1]
scored_full = [(meta[i][0], float(probs[i]), i) for i in range(len(meta))]
best_full = best_threshold(scored_full)
print(f"  Full-fit: F1={best_full['f1']:.3f}  P={best_full['p']:.3f}  R={best_full['r']:.3f}  "
      f"TP={best_full['tp']} FP={best_full['fp']} FN={best_full['fn']}")

print("\n5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof_probs = np.zeros(len(y))
for fold, (tr, te) in enumerate(kf.split(X)):
    c = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.05, random_state=0, subsample=0.8)
    c.fit(X[tr], y[tr])
    oof_probs[te] = c.predict_proba(X[te])[:, 1]
scored_cv = [(meta[i][0], float(oof_probs[i]), i) for i in range(len(meta))]
best_cv = best_threshold(scored_cv)
print(f"  CV:       F1={best_cv['f1']:.3f}  P={best_cv['p']:.3f}  R={best_cv['r']:.3f}  "
      f"TP={best_cv['tp']} FP={best_cv['fp']} FN={best_cv['fn']}")

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
print("\nTop 12 features:")
for i in order[:12]:
    print(f"  {fnames[i]:18s}  {imp[i]:.3f}")

best = best_full
print(f"\n=== SELECTED (full-fit) ===")
print(f"  F1={best['f1']:.3f}  P={best['p']:.3f}  R={best['r']:.3f}  "
      f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")

matched_di = {di for (di,_,_) in best["matched"]}
gt_by_det  = {di: gi for (di,gi,_) in best["matched"]}

print("\nDetections:")
dets_out = []
for i, (fn, sc, mi) in enumerate(best["kept"]):
    tag  = "TP" if i in matched_di else "FP"
    info = f"gt={GT_TS[gt_by_det[i]]}" if i in matched_di else ""
    sid  = meta[mi][1]
    print(f"  [{tag}] f{fn:4d} sid={sid} prob={sc:.2f}  {info}")
    m = candidate_meta(fn, sid)
    dets_out.append({
        "frame":          fn,
        "score":          round(float(sc), 3),
        "sid":            sid,
        "action":         m["action"],
        "speed_kmh":      m["speed_kmh"],
        "audio":          m["audio"],
        "ce":             m["ce"],
        "contact_region": m["contact_region"],
        "dist_px":        m["dist_px"],
        "contact_frame":  m["contact_frame"],
        "hit_xy":         m["hit_xy"],
    })

missed = [GT_TS[gi] for gi in range(len(GT_FRAMES)) if gi not in best["mg"]]
print(f"\nMissed GTs ({len(missed)}): {', '.join(missed)}")

out_data = {
    "metrics_full": {k: best_full[k] for k in ("f1","p","r","tp","fp","fn")},
    "metrics_cv":   {k: best_cv[k]   for k in ("f1","p","r","tp","fp","fn")},
    "feature_importance": {fnames[i]: float(imp[i]) for i in range(len(fnames))},
    "detections": dets_out,
}
with open(OUT_JSON, "w") as fh:
    json.dump(out_data, fh, indent=2)
print(f"\nSaved -> {OUT_JSON}")
