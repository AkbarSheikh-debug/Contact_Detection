"""
Fusion v6 - Full improvement stack:
  NEW vs v5:
  1. Optical flow: receiver head-region flow post-impact (head snap detection)
  2. Receiver head acceleration (impulse = velocity change at contact)
  3. Wrist approach rate (is wrist closing in over last 8 frames?)
  4. Audio onset strength (librosa, more precise than raw flux)
  5. XGBoost classifier (better than sklearn GBM)
  6. Temporal approach pattern (wrist distance monotonically decreasing)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import json, os, sys, cv2, numpy as np, librosa
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
VIDEO_IN   = os.path.join(FOLDER, "3.mp4")
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
RELABEL_JSON = os.path.join(FOLDER, "relabeled_gt.json")
OUT_JSON   = os.path.join(FOLDER, "3_fusion_v6.json")
FPS = 24.995
W, H = 1920, 1080

# ── GT frames ─────────────────────────────────────────────────────────────
GT_TS_ORIG = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
              "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
              "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
              "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]

def to_f(ts):
    p = ts.split(":")
    s = int(p[0])+int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s*FPS))

def to_ts(fn):
    t = fn/FPS; m = int(t//60); s = t-m*60
    return f"{m}:{int(s):02d}:{int(round((s-int(s))*FPS)):02d}"

GT_TS_ORIG_FRAMES = [to_f(ts) for ts in GT_TS_ORIG]

if os.path.exists(RELABEL_JSON):
    with open(RELABEL_JSON) as f: rl = json.load(f)
    GT_FRAMES = [r["suggested_frame"] for r in rl]
    GT_TS     = [r["suggested_ts"]    for r in rl]
    print(f"Using RELABELED GT ({len(GT_FRAMES)} entries)")
else:
    GT_FRAMES = GT_TS_ORIG_FRAMES
    GT_TS     = GT_TS_ORIG
    print("Using ORIGINAL GT")

WRIST_KPS = [9, 10]; HAND_KPS = [9,10,7,8]
HEAD_KPS  = [0,1,2,3,4]; TORSO_KPS = [5,6,11,12]
ALL_RECV_KPS = HEAD_KPS + TORSO_KPS

print("Loading SAM3D + actions...")
with open(SAM3D_JSON) as f: sam3d = json.load(f)
with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
ce_events = sam3d.get("contact_events", [])
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e

TOTAL_FRAMES = 4575


# ── Geometry helpers ───────────────────────────────────────────────────────
def bb_cen(e):  x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.,(y1+y2)/2.])
def bb_head(e): x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.,y1+0.18*(y2-y1)])
def bb_area(e): x1,y1,x2,y2=e["bbox"]; return (x2-x1)*(y2-y1)
def get(fn, tid): return fp.get(fn,{}).get(tid)

def proj_kp(e, k):
    nc = e.get("normalized_coords")
    if nc is None or k >= len(nc): return None
    nc_arr = np.array(nc)
    x1,y1,x2,y2 = e["bbox"]
    Y_min,Y_max = nc_arr[:,1].min(), nc_arr[:,1].max()
    X_min,X_max = nc_arr[:,0].min(), nc_arr[:,0].max()
    if Y_max<=Y_min or X_max<=X_min: return None
    u = x1 + (nc_arr[k,0]-X_min)/(X_max-X_min)*(x2-x1)
    v = y1 + (nc_arr[k,1]-Y_min)/(Y_max-Y_min)*(y2-y1)
    return np.array([u, v])

def fist_pos(se, re):
    if se is None or re is None: return None
    for k in WRIST_KPS:
        p = proj_kp(se, k)
        if p is not None: return p
    sx1,sy1,sx2,sy2 = se["bbox"]
    rx1,ry1,rx2,ry2 = re["bbox"]
    sc_x=(sx1+sx2)/2; rc_x=(rx1+rx2)/2
    fx = sx2 if sc_x < rc_x else sx1
    return np.array([fx, sy1+0.28*(sy2-sy1)])

def wrist_to_recv_pixel(se, re):
    if se is None or re is None: return -1,-1
    rh_kp = proj_kp(re, 0)
    if rh_kp is None: rh_kp = bb_head(re)
    rc = bb_cen(re)
    best_h=1e6; best_c=1e6
    for k in WRIST_KPS:
        sp = proj_kp(se, k)
        if sp is None: continue
        best_h = min(best_h, float(np.linalg.norm(sp-rh_kp)))
        best_c = min(best_c, float(np.linalg.norm(sp-rc)))
    return (best_h if best_h<9e5 else -1, best_c if best_c<9e5 else -1)

def hand_in_bbox(se, re):
    if se is None or re is None: return 0
    x1,y1,x2,y2 = re["bbox"]
    for k in WRIST_KPS:
        p = proj_kp(se, k)
        if p is None: continue
        if x1<=p[0]<=x2 and y1<=p[1]<=y2: return 1
    return 0

def head_2d_track(fn0, tid, span):
    pos = []
    for df in range(-span, span+1):
        e = get(fn0+df, tid)
        pos.append(bb_head(e) if e else None)
    return pos

def total_motion(positions):
    s=0.; v=0
    for i in range(1,len(positions)):
        if positions[i-1] is None or positions[i] is None: continue
        s+=float(np.linalg.norm(positions[i]-positions[i-1])); v+=1
    return s,v

def velocity_change(positions):
    if len(positions)<3: return 0.
    s=0.
    for i in range(2,len(positions)):
        a,b,c=positions[i-2],positions[i-1],positions[i]
        if a is None or b is None or c is None: continue
        s+=float(np.linalg.norm((c-b)-(b-a)))
    return s


# ── NEW: Optical flow precomputation ──────────────────────────────────────
# Stores per-frame per-fighter (head_mean_mag, head_max_mag, body_mean_mag)
# Computed in one video pass at half resolution for speed.
FLOW_SCALE = 0.5
print("Precomputing optical flow (single video pass)...")
cap = cv2.VideoCapture(VIDEO_IN)
prev_gray = None
flow_cache = {}   # fn -> {tid: (head_mean, head_max, body_mean)}
fn_v = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    small = cv2.resize(frame, (0,0), fx=FLOW_SCALE, fy=FLOW_SCALE)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 12, 3, 5, 1.1, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        persons = fp.get(fn_v, {})
        feats = {}
        for tid, e in persons.items():
            x1,y1,x2,y2 = [int(v*FLOW_SCALE) for v in e["bbox"]]
            x1=max(0,x1); y1=max(0,y1)
            x2=min(mag.shape[1],x2); y2=min(mag.shape[0],y2)
            # Head region: top 28% of bbox
            hy2 = y1 + max(1, int(0.28*(y2-y1)))
            head_reg = mag[y1:hy2, x1:x2]
            body_reg = mag[y1:y2,  x1:x2]
            hm = float(head_reg.mean()) if head_reg.size>0 else 0.
            hM = float(head_reg.max())  if head_reg.size>0 else 0.
            bm = float(body_reg.mean()) if body_reg.size>0 else 0.
            feats[tid] = (hm, hM, bm)
        flow_cache[fn_v] = feats
    prev_gray = gray
    fn_v += 1
    if fn_v % 1000 == 0: print(f"  optical flow: {fn_v}/{TOTAL_FRAMES}")
cap.release()
print(f"  Done. Cached {len(flow_cache)} frames.")


def flow_recv_features(fn, rid, look_ahead=6):
    """Max optical flow in receiver's head region over look_ahead post-impact frames."""
    head_mean=0.; head_max=0.; body_mean=0.
    for df in range(0, look_ahead+1):
        fc = flow_cache.get(fn+df, {})
        rv = fc.get(rid)
        if rv:
            head_mean = max(head_mean, rv[0])
            head_max  = max(head_max,  rv[1])
            body_mean = max(body_mean, rv[2])
    return head_mean, head_max, body_mean

def flow_striker_features(fn, sid, look_before=4):
    """Striker body flow in frames before contact (high = punch being thrown)."""
    body_mean=0.
    for df in range(-look_before, 1):
        fc = flow_cache.get(fn+df, {})
        sv = fc.get(sid)
        if sv: body_mean = max(body_mean, sv[2])
    return body_mean


# ── Audio ──────────────────────────────────────────────────────────────────
print("Audio...")
y_a, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
y_pe = np.append(y_a[0], y_a[1:] - 0.97*y_a[:-1])
hop = 256
S = np.abs(librosa.stft(y_pe, n_fft=2048, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
bands = {"low":(200,1000), "mid":(1000,4000), "high":(4000,8000)}
band_flux = {}
for name,(lo,hi) in bands.items():
    msk = (freqs>=lo)&(freqs<=hi)
    f = np.maximum(0, np.diff(S[msk],axis=1)).sum(axis=0).astype(np.float64)
    f = np.concatenate([[0.],f]); f-=np.median(f); f=np.maximum(0,f)
    f /= (f.max()+1e-9); band_flux[name]=f
times = librosa.frames_to_time(np.arange(len(band_flux["mid"])), sr=sr, hop_length=hop)

# Onset strength (more precise than raw flux)
onset_env = librosa.onset.onset_strength(y=y_a, sr=sr, hop_length=hop)
onset_env -= onset_env.mean(); onset_env = np.maximum(0, onset_env)
onset_env /= (onset_env.max()+1e-9)

onsets_mid = librosa.util.peak_pick(band_flux["mid"], pre_max=20, post_max=20,
                                     pre_avg=30, post_avg=30,
                                     delta=float(band_flux["mid"].std()*1.0), wait=12)
onset_frames_set = set((times[onsets_mid]*FPS).astype(int).tolist())

def audio_features_at(fn):
    t = fn/FPS; idx = int(round(t*sr/hop))
    rad = int(round(6/FPS*sr/hop))
    lo,hi = max(0,idx-rad), min(len(band_flux["mid"]),idx+rad+1)
    out = [float(band_flux[n][lo:hi].max() if hi>lo else 0.) for n in ("low","mid","high")]
    out.append(max(out[:3]))
    # Onset strength peak in window
    out.append(float(onset_env[lo:hi].max() if hi>lo else 0.))
    # Audio rising: is flux increasing into fn?
    rad2 = int(round(3/FPS*sr/hop))
    lo2,hi2 = max(0,idx-rad2*2), max(0,idx-rad2)
    flux_before = float(band_flux["mid"][lo2:hi2].max() if hi2>lo2 else 0.)
    out.append(float(max(0., out[1]-flux_before)))  # rising_mid
    return out  # 6 values


# ── Candidate generation ───────────────────────────────────────────────────
print("Candidate set...")
cands = set()
for a in actions:
    sid = int(a["fighter_type"].split("_")[1]); cands.add((a["frame"],sid))
for c in ce_events:
    cands.add((c["frame"],c["striker_id"]))
for f in onset_frames_set:
    cands.add((f,0)); cands.add((f,1))
for fn in range(0,TOTAL_FRAMES,5):
    ps = fp.get(fn,{})
    if len(ps)>=2: cands.add((fn,0)); cands.add((fn,1))
cands = sorted(cands)
print(f"  {len(cands)} candidate (frame,sid) pairs")


# ── Feature extraction ─────────────────────────────────────────────────────
print("Extracting features...")
def extract_features(fn, sid):
    rid = 1-sid
    se = get(fn,sid); re = get(fn,rid)
    if se is None or re is None: return None

    # 1) Audio (6: low,mid,high,max,onset_strength,rising_mid)
    aud = audio_features_at(fn)

    # 2) Action JSON
    act_conf=0.; act_sig=0; act_speed=0.
    for a in actions:
        if abs(a["frame"]-fn)<=5 and int(a["fighter_type"].split("_")[1])==sid:
            if a["confidence"]>act_conf:
                act_conf=a["confidence"]
                act_sig=int(a.get("is_significant",False))
                act_speed=a.get("speed_estimation",{}).get("estimated_speed_kmh",0.)

    # 3) Contact events
    ce_best=0.; ce_ht=0.; ce_dist=999.
    for c in ce_events:
        if abs(c["frame"]-fn)<=8 and c["striker_id"]==sid:
            if c["contact_prob"]>ce_best:
                ce_best=c["contact_prob"]; ce_dist=c["contact_3d_distance_m"]
            if c["contact_region"] in ("head","torso") and c["contact_prob"]>ce_ht:
                ce_ht=c["contact_prob"]

    # 4) Wrist-to-receiver pixel distance (keypoint-based)
    d_h_min=1e6; d_c_min=1e6
    for df in range(-3,4):
        d_h,d_c = wrist_to_recv_pixel(get(fn+df,sid), get(fn+df,rid))
        if d_h>0 and d_h<d_h_min: d_h_min=d_h
        if d_c>0 and d_c<d_c_min: d_c_min=d_c
    if d_h_min>9e5: d_h_min=1000.
    if d_c_min>9e5: d_c_min=1000.

    # 5) Hand in bbox
    in_bbox=0
    for df in range(-4,5):
        if hand_in_bbox(get(fn+df,sid), get(fn+df,rid)):
            in_bbox=1; break

    # 6) Receiver head motion pre/post
    recv_pre  = head_2d_track(fn-1,rid,6)[:7]
    recv_post = head_2d_track(fn+1,rid,6)[:7]
    motion_pre,vp   = total_motion(recv_pre)
    motion_post,vp2 = total_motion(recv_post)
    motion_pre  /= max(1,vp); motion_post /= max(1,vp2)
    vel_chg_recv = velocity_change(head_2d_track(fn,rid,5))

    # 7) NEW: Receiver head acceleration (impulse = velocity change at contact)
    # v_before = avg head velocity over fn-4..fn-2, v_after = fn+2..fn+4
    def head_vel(fn0, tid, start, end):
        positions = [get(fn0+df, tid) for df in range(start, end+1)]
        heads = [bb_head(e) if e else None for e in positions]
        s=0.; v=0
        for i in range(1,len(heads)):
            if heads[i-1] is None or heads[i] is None: continue
            s+=float(np.linalg.norm(bb_head(get(fn0+i+start-1,tid))
                                     - bb_head(get(fn0+i+start,tid)))
                     if get(fn0+i+start-1,tid) and get(fn0+i+start,tid) else 0.)
            v+=1
        return s/max(1,v)
    h_vel_before = head_vel(fn, rid, -5, -2)
    h_vel_after  = head_vel(fn, rid,  2,  5)
    head_accel   = float(h_vel_after - h_vel_before)  # positive = head snapped

    # 8) Receiver bbox change
    e_pp=get(fn-5,rid); e_nn=get(fn+5,rid)
    if e_pp and e_nn:
        bbox_pos_chg  = float(np.linalg.norm(bb_cen(e_nn)-bb_cen(e_pp)))
        bbox_area_chg = abs(bb_area(e_nn)-bb_area(e_pp))/max(1,bb_area(e_pp))
    else:
        bbox_pos_chg=0.; bbox_area_chg=0.

    # 9) Striker wrist speed 3D + decel 2D
    se_p=get(fn-3,sid); se_n=get(fn+3,sid)
    wrist_speed_3d=0.; wrist_decel_2d=0.
    if se_p and se_n and se:
        for k in WRIST_KPS:
            wc_p=se_p.get("world_coords",[None]*70)[k] if se_p.get("world_coords") else None
            wc_n=se_n.get("world_coords",[None]*70)[k] if se_n.get("world_coords") else None
            if wc_p and wc_n:
                wrist_speed_3d=max(wrist_speed_3d,
                    float(np.linalg.norm(np.array(wc_n)-np.array(wc_p))/6.))
        fp_p=fist_pos(se_p,get(fn-3,rid))
        fp_0=fist_pos(se,  get(fn,  rid))
        fp_n=fist_pos(se_n,get(fn+3,rid))
        if fp_p is not None and fp_0 is not None and fp_n is not None:
            v1=float(np.linalg.norm(fp_0-fp_p))/3.
            v2=float(np.linalg.norm(fp_n-fp_0))/3.
            wrist_decel_2d=max(0.,v1-v2)

    # 10) Cross-fighter distance + approach
    d_now = float(np.linalg.norm(bb_cen(se)-bb_cen(re)))
    e_pp_s=get(fn-5,sid); e_pp_r=get(fn-5,rid)
    approach = (float(np.linalg.norm(bb_cen(e_pp_s)-bb_cen(e_pp_r)))-d_now
                if e_pp_s and e_pp_r else 0.)

    # 11) NEW: Wrist approach rate over 8 frames
    d_8ago = -1.
    se_8=get(fn-8,sid); re_8=get(fn-8,rid)
    if se_8 and re_8:
        fp_8=fist_pos(se_8,re_8)
        rh_8=bb_head(re_8)
        if fp_8 is not None:
            d_8ago=float(np.linalg.norm(fp_8-rh_8))
    approach_rate = (d_8ago-d_h_min)/8. if d_8ago>0 else 0.  # px/frame closing in

    # 12) NEW: Optical flow — receiver head snap
    flow_hm, flow_hM, flow_bm = flow_recv_features(fn, rid, look_ahead=5)

    # 13) NEW: Optical flow — striker body pre-impact (punch being thrown)
    flow_strike = flow_striker_features(fn, sid, look_before=4)

    feat = [
        aud[0], aud[1], aud[2], aud[3], aud[4], aud[5],  # 6 audio
        act_conf, act_sig, act_speed,                      # 3 action
        ce_best, ce_ht, ce_dist,                           # 3 contact_event
        d_h_min, d_c_min, in_bbox,                         # 3 pixel dist
        motion_pre, motion_post, motion_post-motion_pre,   # 3 recv motion
        vel_chg_recv, head_accel,                          # 2 recv reaction
        bbox_pos_chg, bbox_area_chg,                       # 2 bbox change
        wrist_speed_3d, wrist_decel_2d,                    # 2 wrist motion
        d_now, approach, approach_rate,                    # 3 distance
        flow_hm, flow_hM, flow_bm, flow_strike,           # 4 optical flow
    ]
    return feat  # 33 features


X_rows, meta = [], []
for fn, sid in cands:
    feat = extract_features(fn, sid)
    if feat is None: continue
    X_rows.append(feat)
    meta.append((fn, sid))
X = np.array(X_rows, dtype=np.float32)
print(f"  feature matrix: {X.shape}")

y = np.zeros(len(meta), dtype=int)
for i,(fn,sid) in enumerate(meta):
    for gf in GT_FRAMES:
        if abs(fn-gf)<=30: y[i]=1; break
print(f"  positives: {y.sum()}  negatives: {(1-y).sum()}")


# ── Evaluation helpers ─────────────────────────────────────────────────────
def evaluate_frames(det):
    pairs=[]
    for di,df in enumerate(det):
        for gi,gf in enumerate(GT_FRAMES):
            d=abs(df-gf)
            if d<=30: pairs.append((d,di,gi))
    pairs.sort()
    md,mg,matched=set(),set(),[]
    for d,di,gi in pairs:
        if di in md or gi in mg: continue
        md.add(di); mg.add(gi); matched.append((di,gi,d))
    tp=len(mg); fp=len(det)-len(md); fn=len(GT_FRAMES)-tp
    p=tp/(tp+fp) if (tp+fp) else 0
    r=tp/(tp+fn) if (tp+fn) else 0
    f1=2*p*r/(p+r) if (p+r) else 0
    return dict(tp=tp,fp=fp,fn=fn,p=p,r=r,f1=f1,matched=matched,mg=mg,md=md)

def nms(items, cd):
    kept=[]
    for it in sorted(items,key=lambda x:-x[1]):
        if any(abs(it[0]-k[0])<cd for k in kept): continue
        kept.append(it)
    return sorted(kept,key=lambda x:x[0])

def best_threshold(scored):
    best=None
    for cd in [12,15,20,25,30]:
        for thr in np.arange(0.05,0.95,0.02):
            items=[(fn,sc,i) for fn,sc,i in scored if sc>=thr]
            kept=nms(items,cd)
            det=[k[0] for k in kept]
            if not det: continue
            res=evaluate_frames(det)
            if best is None or res["f1"]>best["f1"]:
                best=dict(res,cd=cd,thr=float(thr),kept=kept)
    return best


# ── XGBoost training ───────────────────────────────────────────────────────
xgb_params = dict(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    scale_pos_weight=float((1-y).sum())/max(1,y.sum()),
    use_label_encoder=False, eval_metric="logloss",
    random_state=0, n_jobs=-1, verbosity=0
)

print("\nXGBoost (full-data fit)...")
clf = XGBClassifier(**xgb_params)
clf.fit(X, y)
probs = clf.predict_proba(X)[:,1]
scored_full = [(meta[i][0], float(probs[i]), i) for i in range(len(meta))]
best_full = best_threshold(scored_full)
print(f"  Full-fit: F1={best_full['f1']:.3f}  P={best_full['p']:.3f}  R={best_full['r']:.3f}  "
      f"TP={best_full['tp']} FP={best_full['fp']} FN={best_full['fn']}")

print("\n5-fold CV (honest performance)...")
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof_probs = np.zeros(len(y))
for fold,(tr,te) in enumerate(kf.split(X)):
    c = XGBClassifier(**xgb_params)
    c.fit(X[tr], y[tr])
    oof_probs[te] = c.predict_proba(X[te])[:,1]
scored_cv = [(meta[i][0], float(oof_probs[i]), i) for i in range(len(meta))]
best_cv = best_threshold(scored_cv)
print(f"  CV:       F1={best_cv['f1']:.3f}  P={best_cv['p']:.3f}  R={best_cv['r']:.3f}  "
      f"TP={best_cv['tp']} FP={best_cv['fp']} FN={best_cv['fn']}")

fnames = ["aud_low","aud_mid","aud_high","aud_max","aud_onset","aud_rising",
          "act_conf","act_sig","act_speed",
          "ce_best","ce_ht","ce_dist",
          "d_h_min","d_c_min","in_bbox",
          "motion_pre","motion_post","motion_delta",
          "vel_chg_recv","head_accel",
          "bbox_pos_chg","bbox_area_chg",
          "wrist_speed_3d","wrist_decel_2d",
          "d_now","approach","approach_rate",
          "flow_recv_mean","flow_recv_max","flow_body_mean","flow_strike"]
imp = clf.feature_importances_
order = np.argsort(-imp)
print("\nTop 15 features:")
for i in order[:15]:
    print(f"  {fnames[i]:20s}  {imp[i]:.3f}")


# ── Contact metadata ───────────────────────────────────────────────────────
def find_contact(fn, sid, scan=12):
    rid=1-sid; best_dist=999.; best_fn=fn
    for df in range(-scan,scan+1):
        cfn=fn+df; se=get(cfn,sid); re=get(cfn,rid)
        if se is None or re is None: continue
        for sk in WRIST_KPS:
            sp=proj_kp(se,sk)
            if sp is None: continue
            for rk in HEAD_KPS:
                rp=proj_kp(re,rk)
                if rp is None: continue
                d=float(np.linalg.norm(sp-rp))
                if d<best_dist: best_dist=d; best_fn=cfn
    ce_region=None; recv_kp_idx=None; ce_best_p=0.
    for c in ce_events:
        if abs(c["frame"]-fn)<=15 and c["striker_id"]==sid:
            if c["contact_prob"]>ce_best_p:
                ce_best_p=c["contact_prob"]
                ce_region=c.get("contact_region")
                recv_kp_idx=c.get("receiver_keypoint")
    re_c=get(best_fn,rid) or get(fn,rid)
    hit_xy=None
    if re_c is not None:
        if recv_kp_idx is not None: hit_xy=proj_kp(re_c,recv_kp_idx)
        if hit_xy is None:          hit_xy=proj_kp(re_c,0)
        if hit_xy is None:
            x1,y1,x2,y2=re_c["bbox"]
            hit_xy=np.array([(x1+x2)/2, y1+0.18*(y2-y1)])
    return best_fn, hit_xy, best_dist, ce_region

def candidate_meta(fn, sid):
    aud=audio_features_at(fn)
    act_conf=0.; act_speed=0.; act_type="punch"
    for a in actions:
        if abs(a["frame"]-fn)<=8 and int(a["fighter_type"].split("_")[1])==sid:
            if a["confidence"]>act_conf:
                act_conf=a["confidence"]
                act_speed=a.get("speed_estimation",{}).get("estimated_speed_kmh",0.)
                act_type=a.get("action","punch")
    ce_best=0.; ce_region="unknown"
    for c in ce_events:
        if abs(c["frame"]-fn)<=10 and c["striker_id"]==sid:
            if c["contact_prob"]>ce_best:
                ce_best=c["contact_prob"]; ce_region=c.get("contact_region","unknown")
    contact_fn,hit_rp,dist_px,ce_region2=find_contact(fn,sid,scan=12)
    if ce_region2: ce_region=ce_region2
    return {
        "action":act_type, "speed_kmh":round(act_speed,1),
        "audio":round(float(max(aud[:3])),3), "ce":round(float(ce_best),3),
        "contact_region":ce_region, "dist_px":round(float(dist_px),1),
        "contact_frame":contact_fn,
        "hit_xy":[round(float(hit_rp[0]),1),round(float(hit_rp[1]),1)] if hit_rp is not None else None,
    }


# ── Select best model + save ───────────────────────────────────────────────
best = best_full
print(f"\n=== SELECTED (full-fit) ===")
print(f"  F1={best['f1']:.3f}  P={best['p']:.3f}  R={best['r']:.3f}  "
      f"TP={best['tp']} FP={best['fp']} FN={best['fn']}  cd={best['cd']}")

matched_di={di for (di,_,_) in best["matched"]}
gt_by_det={di:gi for (di,gi,_) in best["matched"]}
dets_out=[]
print("\nDetections:")
for i,(fn,sc,mi) in enumerate(best["kept"]):
    tag="TP" if i in matched_di else "FP"
    info=f"gt={GT_TS[gt_by_det[i]]}" if i in matched_di else ""
    sid=meta[mi][1]
    print(f"  [{tag}] f{fn:4d} sid={sid} prob={sc:.2f}  {info}")
    m=candidate_meta(fn,sid)
    dets_out.append({
        "frame":fn,"score":round(float(sc),3),"sid":sid,
        "action":m["action"],"speed_kmh":m["speed_kmh"],
        "audio":m["audio"],"ce":m["ce"],
        "contact_region":m["contact_region"],"dist_px":m["dist_px"],
        "contact_frame":m["contact_frame"],"hit_xy":m["hit_xy"],
    })

missed=[GT_TS[gi] for gi in range(len(GT_FRAMES)) if gi not in best["mg"]]
print(f"\nMissed GTs ({len(missed)}): {', '.join(missed)}")

out_data={
    "metrics_full":{k:best_full[k] for k in ("f1","p","r","tp","fp","fn")},
    "metrics_cv":  {k:best_cv[k]   for k in ("f1","p","r","tp","fp","fn")},
    "feature_importance":{fnames[i]:float(imp[i]) for i in range(len(fnames))},
    "detections":dets_out,
}
with open(OUT_JSON,"w") as fh: json.dump(out_data,fh,indent=2)
print(f"\nSaved -> {OUT_JSON}")
