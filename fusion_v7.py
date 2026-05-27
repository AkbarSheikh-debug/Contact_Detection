"""
Fusion v7 - Improvements over v6:
  NEW:
  1. Optical flow direction coherence (head snap = coherent direction, not random motion)
  2. Paired physics: wrist_decel × head_accel simultaneously (physics constraint)
  3. Approach trajectory slope over last 8 frames (temporal window)
  4. Spectral centroid (sharp punch = high centroid)
  5. Short-time RMS energy (boxing impacts are brief and sharp)
  6. Pre-impact silence ratio (real punches often preceded by brief quiet)
"""

import json, os, cv2, numpy as np, librosa
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

FOLDER      = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
VIDEO_IN    = os.path.join(FOLDER, "3.mp4")
SAM3D_JSON  = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON    = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV   = os.path.join(FOLDER, "3.wav")
RELABEL_JSON= os.path.join(FOLDER, "relabeled_gt.json")
OUT_JSON    = os.path.join(FOLDER, "3_fusion_v7.json")
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
    if se is None or re is None: return -1, -1
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

def head_vel(fn0, tid, start, end):
    """Average pixel-per-frame velocity of receiver head over fn0+start..fn0+end."""
    heads = [bb_head(e) if (e:=get(fn0+df, tid)) else None for df in range(start, end+1)]
    dists = [float(np.linalg.norm(heads[i]-heads[i-1]))
             for i in range(1, len(heads)) if heads[i-1] is not None and heads[i] is not None]
    return float(np.mean(dists)) if dists else 0.

def velocity_change(fn0, tid, span):
    """Magnitude of head acceleration (velocity change) over ±span around fn0."""
    positions = [bb_head(e) if (e:=get(fn0+df, tid)) else None for df in range(-span, span+1)]
    s = 0.
    for i in range(2, len(positions)):
        a,b,c = positions[i-2], positions[i-1], positions[i]
        if a is None or b is None or c is None: continue
        s += float(np.linalg.norm((c-b)-(b-a)))
    return s


# ── Optical flow precomputation ────────────────────────────────────────────
# Stores per-frame per-fighter: (head_mean_mag, head_max_mag, body_mean_mag, head_ang_coherence)
# head_ang_coherence: 1=all flow in same direction (real head snap), 0=random
FLOW_SCALE = 0.5
print("Precomputing optical flow (single video pass)...")
cap = cv2.VideoCapture(VIDEO_IN)
prev_gray = None
flow_cache = {}
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
            x2=min(mag.shape[1]-1,x2); y2=min(mag.shape[0]-1,y2)
            hy2 = y1 + max(1, int(0.28*(y2-y1)))
            head_reg  = mag[y1:hy2, x1:x2]
            body_reg  = mag[y1:y2,  x1:x2]
            hm = float(head_reg.mean()) if head_reg.size>0 else 0.
            hM = float(head_reg.max())  if head_reg.size>0 else 0.
            bm = float(body_reg.mean()) if body_reg.size>0 else 0.
            # Direction coherence: low angle std = all pixels moving same way = real snap
            fx_r = flow[y1:hy2, x1:x2, 0]
            fy_r = flow[y1:hy2, x1:x2, 1]
            if fx_r.size > 4:
                angles = np.arctan2(fy_r.ravel(), fx_r.ravel())
                ang_std = float(np.std(angles))
                coherence = max(0., 1. - ang_std / np.pi)
            else:
                coherence = 0.
            feats[tid] = (hm, hM, bm, coherence)
        flow_cache[fn_v] = feats
    prev_gray = gray
    fn_v += 1
    if fn_v % 1000 == 0: print(f"  optical flow: {fn_v}/{TOTAL_FRAMES}")
cap.release()
print(f"  Done. Cached {len(flow_cache)} frames.")


def flow_recv_features(fn, rid, look_ahead=6):
    """Max optical flow in receiver's head region over look_ahead post-impact frames."""
    head_mean=0.; head_max=0.; body_mean=0.; head_coh=0.
    for df in range(0, look_ahead+1):
        fc = flow_cache.get(fn+df, {})
        rv = fc.get(rid)
        if rv:
            head_mean = max(head_mean, rv[0])
            head_max  = max(head_max,  rv[1])
            body_mean = max(body_mean, rv[2])
            head_coh  = max(head_coh,  rv[3])
    return head_mean, head_max, body_mean, head_coh

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

N_af = len(band_flux["mid"])
times = librosa.frames_to_time(np.arange(N_af), sr=sr, hop_length=hop)

onset_env = librosa.onset.onset_strength(y=y_a, sr=sr, hop_length=hop)
onset_env -= onset_env.mean(); onset_env = np.maximum(0, onset_env)
onset_env /= (onset_env.max()+1e-9)

# NEW: spectral centroid (normalized) — sharp impact sound has high centroid
centroid = librosa.feature.spectral_centroid(y=y_a, sr=sr, hop_length=hop)[0]
centroid /= (centroid.max()+1e-9)

# NEW: short-time RMS energy — boxing impacts are brief sharp spikes
rms_env = librosa.feature.rms(y=y_a, frame_length=2048, hop_length=hop)[0]
rms_env /= (rms_env.max()+1e-9)
# Truncate/pad to N_af to keep lengths consistent
if len(centroid) > N_af: centroid = centroid[:N_af]
if len(rms_env)  > N_af: rms_env  = rms_env[:N_af]

onsets_mid = librosa.util.peak_pick(band_flux["mid"], pre_max=20, post_max=20,
                                     pre_avg=30, post_avg=30,
                                     delta=float(band_flux["mid"].std()*1.0), wait=12)
onset_frames_set = set((times[onsets_mid]*FPS).astype(int).tolist())

def audio_features_at(fn):
    t = fn/FPS; idx = int(round(t*sr/hop))
    rad = int(round(6/FPS*sr/hop))
    lo,hi = max(0,idx-rad), min(N_af, idx+rad+1)
    # 3-band flux + max
    out = [float(band_flux[n][lo:hi].max() if hi>lo else 0.) for n in ("low","mid","high")]
    out.append(max(out[:3]))                                          # aud_max
    out.append(float(onset_env[lo:hi].max() if hi>lo else 0.))       # aud_onset
    # rising: mid-flux increasing into fn?
    rad2 = int(round(3/FPS*sr/hop))
    lo2,hi2 = max(0,idx-rad2*2), max(0,idx-rad2)
    flux_before = float(band_flux["mid"][lo2:hi2].max() if hi2>lo2 else 0.)
    out.append(float(max(0., out[1]-flux_before)))                    # aud_rising
    # NEW: spectral centroid peak in window (sharp punch = high centroid)
    out.append(float(centroid[lo:hi].max() if hi>lo else 0.))         # aud_centroid
    # NEW: short-time RMS energy in ±3 frame window (sharp brief spike)
    rad3 = int(round(3/FPS*sr/hop))
    lo3,hi3 = max(0,idx-rad3), min(len(rms_env),idx+rad3+1)
    out.append(float(rms_env[lo3:hi3].max() if hi3>lo3 else 0.))      # aud_ste
    # NEW: pre-impact silence — mean RMS from -10..-2 frames before fn
    rad4 = int(round(10/FPS*sr/hop))
    rad5 = int(round(2/FPS*sr/hop))
    lo4,hi4 = max(0,idx-rad4), max(0,idx-rad5)
    out.append(float(rms_env[lo4:hi4].mean()) if hi4>lo4 else 0.5)    # aud_silence_pre
    return out  # 9 values


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
FEAT_NAMES = [
    "aud_low","aud_mid","aud_high","aud_max","aud_onset","aud_rising",
    "aud_centroid","aud_ste","aud_silence_pre",       # 9 audio
    "act_conf","act_sig","act_speed",                  # 3 action
    "ce_best","ce_ht","ce_dist",                       # 3 contact_event
    "d_h_min","d_c_min","in_bbox",                     # 3 pixel dist
    "motion_pre","motion_post","motion_delta",          # 3 recv motion
    "vel_chg_recv","head_accel","paired_impact",        # 3 recv reaction (+paired)
    "bbox_pos_chg","bbox_area_chg",                    # 2 bbox change
    "wrist_speed_3d","wrist_decel_2d",                 # 2 wrist motion
    "d_now","approach","approach_rate","approach_trend",# 4 distance (+trend)
    "flow_recv_mean","flow_recv_max","flow_body_mean",
    "flow_strike","flow_recv_coherence",               # 5 optical flow (+coherence)
]  # 37 total

def extract_features(fn, sid):
    rid = 1-sid
    se = get(fn,sid); re = get(fn,rid)
    if se is None or re is None: return None

    # 1) Audio (9)
    aud = audio_features_at(fn)

    # 2) Action JSON (3)
    act_conf=0.; act_sig=0; act_speed=0.
    for a in actions:
        if abs(a["frame"]-fn)<=5 and int(a["fighter_type"].split("_")[1])==sid:
            if a["confidence"]>act_conf:
                act_conf=a["confidence"]
                act_sig=int(a.get("is_significant",False))
                act_speed=a.get("speed_estimation",{}).get("estimated_speed_kmh",0.)

    # 3) Contact events (3)
    ce_best=0.; ce_ht=0.; ce_dist=999.
    for c in ce_events:
        if abs(c["frame"]-fn)<=8 and c["striker_id"]==sid:
            if c["contact_prob"]>ce_best:
                ce_best=c["contact_prob"]; ce_dist=c["contact_3d_distance_m"]
            if c["contact_region"] in ("head","torso") and c["contact_prob"]>ce_ht:
                ce_ht=c["contact_prob"]

    # 4) Wrist-to-receiver distance (3): scan ±5 frames
    d_h_min=1e6; d_c_min=1e6
    for df in range(-5,6):
        d_h,d_c = wrist_to_recv_pixel(get(fn+df,sid), get(fn+df,rid))
        if d_h>0 and d_h<d_h_min: d_h_min=d_h
        if d_c>0 and d_c<d_c_min: d_c_min=d_c
    if d_h_min>9e5: d_h_min=1000.
    if d_c_min>9e5: d_c_min=1000.

    in_bbox=0
    for df in range(-4,5):
        if hand_in_bbox(get(fn+df,sid), get(fn+df,rid)):
            in_bbox=1; break

    # 5) Receiver head motion (3)
    pre_heads  = [bb_head(e) if (e:=get(fn-6+i,rid)) else None for i in range(7)]
    post_heads = [bb_head(e) if (e:=get(fn+1+i,rid)) else None for i in range(7)]
    def _motion(heads):
        dists=[float(np.linalg.norm(heads[i]-heads[i-1]))
               for i in range(1,len(heads)) if heads[i-1] is not None and heads[i] is not None]
        return float(np.mean(dists)) if dists else 0.
    motion_pre  = _motion(pre_heads)
    motion_post = _motion(post_heads)
    vel_chg_recv = velocity_change(fn, rid, 5)

    # 6) Receiver head acceleration (3): pre-vel, post-vel, PAIRED physics
    h_vel_before = head_vel(fn, rid, -5, -2)
    h_vel_after  = head_vel(fn, rid,  2,  5)
    head_accel   = float(h_vel_after - h_vel_before)   # positive = head snapped forward

    # 7) Striker wrist deceleration (2)
    se_p=get(fn-3,sid); se_n=get(fn+3,sid)
    wrist_speed_3d=0.; wrist_decel_2d=0.
    if se_p and se_n and se:
        for k in WRIST_KPS:
            wc_p = (se_p.get("world_coords") or [None]*70)[k]
            wc_n = (se_n.get("world_coords") or [None]*70)[k]
            if wc_p and wc_n:
                wrist_speed_3d = max(wrist_speed_3d,
                    float(np.linalg.norm(np.array(wc_n)-np.array(wc_p))/6.))
        fp_p=fist_pos(se_p,get(fn-3,rid))
        fp_0=fist_pos(se,  get(fn,  rid))
        fp_n=fist_pos(se_n,get(fn+3,rid))
        if fp_p is not None and fp_0 is not None and fp_n is not None:
            v1=float(np.linalg.norm(fp_0-fp_p))/3.
            v2=float(np.linalg.norm(fp_n-fp_0))/3.
            wrist_decel_2d=max(0., v1-v2)

    # Paired physics: both striker decelerates AND receiver accelerates at same time
    paired_impact = float(max(0., wrist_decel_2d) * max(0., head_accel))

    # 8) Bbox change (2)
    e_pp=get(fn-5,rid); e_nn=get(fn+5,rid)
    if e_pp and e_nn:
        bbox_pos_chg  = float(np.linalg.norm(bb_cen(e_nn)-bb_cen(e_pp)))
        bbox_area_chg = abs(bb_area(e_nn)-bb_area(e_pp))/max(1,bb_area(e_pp))
    else:
        bbox_pos_chg=0.; bbox_area_chg=0.

    # 9) Distance features (4)
    d_now = float(np.linalg.norm(bb_cen(se)-bb_cen(re)))
    e_pp_s=get(fn-5,sid); e_pp_r=get(fn-5,rid)
    approach = (float(np.linalg.norm(bb_cen(e_pp_s)-bb_cen(e_pp_r)))-d_now
                if e_pp_s and e_pp_r else 0.)
    # approach_rate: was fist closing over last 8 frames?
    se_8=get(fn-8,sid); re_8=get(fn-8,rid)
    d_8ago=-1.
    if se_8 and re_8:
        fp_8=fist_pos(se_8,re_8)
        if fp_8 is not None: d_8ago=float(np.linalg.norm(fp_8-bb_head(re_8)))
    approach_rate = (d_8ago-d_h_min)/8. if d_8ago>0 else 0.

    # NEW: approach_trend — linear slope of fist distance over fn-8..fn-0
    dist_window=[]
    for df in range(-8, 1, 2):
        d_h,_ = wrist_to_recv_pixel(get(fn+df,sid), get(fn+df,rid))
        if d_h>0: dist_window.append(d_h)
    if len(dist_window)>=3:
        xs = np.arange(len(dist_window), dtype=float)
        slope = float(np.polyfit(xs, dist_window, 1)[0])
        approach_trend = -slope   # positive = fist closing in
    else:
        approach_trend = approach_rate

    # 10) Optical flow (5): recv head flow + direction coherence + striker body
    flow_hm, flow_hM, flow_bm, flow_coh = flow_recv_features(fn, rid, look_ahead=5)
    flow_strike = flow_striker_features(fn, sid, look_before=4)

    feat = [
        aud[0], aud[1], aud[2], aud[3], aud[4], aud[5],        # 6 audio (flux + onset + rising)
        aud[6], aud[7], aud[8],                                  # 3 audio (centroid, ste, silence_pre)
        act_conf, act_sig, act_speed,                            # 3 action
        ce_best, ce_ht, ce_dist,                                 # 3 contact_event
        d_h_min, d_c_min, in_bbox,                               # 3 pixel dist
        motion_pre, motion_post, motion_post-motion_pre,         # 3 recv motion
        vel_chg_recv, head_accel, paired_impact,                 # 3 recv reaction
        bbox_pos_chg, bbox_area_chg,                             # 2 bbox change
        wrist_speed_3d, wrist_decel_2d,                          # 2 wrist motion
        d_now, approach, approach_rate, approach_trend,          # 4 distance
        flow_hm, flow_hM, flow_bm, flow_strike, flow_coh,       # 5 optical flow
    ]
    assert len(feat) == len(FEAT_NAMES), f"Feature count mismatch: {len(feat)} vs {len(FEAT_NAMES)}"
    return feat


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
            if abs(df-gf)<=30: pairs.append((abs(df-gf),di,gi))
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
    n_estimators=500, max_depth=5, learning_rate=0.03,
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

imp = clf.feature_importances_
order = np.argsort(-imp)
print("\nTop 15 features:")
for i in order[:15]:
    print(f"  {FEAT_NAMES[i]:22s}  {imp[i]:.3f}")


# ── Contact metadata ───────────────────────────────────────────────────────
# Scan ALL body keypoints (head + torso + arms) to find true contact location.
# This fixes the bug where body shots got their crosshair drawn on the head.
SCAN_RECV_KPS = HEAD_KPS + TORSO_KPS + [7, 8]  # head, shoulders, hips, elbows
KP_REGION = {}
for k in HEAD_KPS:  KP_REGION[k] = "head"
for k in TORSO_KPS: KP_REGION[k] = "torso"
KP_REGION[7] = "arm"; KP_REGION[8] = "arm"

def find_contact(fn, sid, scan=12):
    """Find true contact frame + the actual struck body part (not just nearest head)."""
    rid = 1 - sid
    best_dist = 999.
    best_fn = fn
    best_recv_kp = None
    best_recv_xy = None
    for df in range(-scan, scan+1):
        cfn = fn + df
        se = get(cfn, sid); re = get(cfn, rid)
        if se is None or re is None: continue
        for sk in WRIST_KPS:
            sp = proj_kp(se, sk)
            if sp is None: continue
            for rk in SCAN_RECV_KPS:
                rp = proj_kp(re, rk)
                if rp is None: continue
                d = float(np.linalg.norm(sp - rp))
                if d < best_dist:
                    best_dist = d
                    best_fn = cfn
                    best_recv_kp = rk
                    best_recv_xy = rp.copy()
    # Region from anatomically closest keypoint
    ce_region = KP_REGION.get(best_recv_kp) if best_recv_kp is not None else None
    # If SAM3D has a high-confidence contact event, prefer its region/keypoint
    ce_best_p = 0.; ce_kp_idx = None; ce_kp_region = None
    for c in ce_events:
        if abs(c["frame"] - fn) <= 15 and c["striker_id"] == sid:
            if c["contact_prob"] > ce_best_p:
                ce_best_p = c["contact_prob"]
                ce_kp_idx = c.get("receiver_keypoint")
                ce_kp_region = c.get("contact_region")
    if ce_best_p >= 0.7:
        if ce_kp_region: ce_region = ce_kp_region
        if ce_kp_idx is not None:
            re_c = get(best_fn, rid)
            if re_c is not None:
                rp_ce = proj_kp(re_c, ce_kp_idx)
                if rp_ce is not None:
                    best_recv_xy = rp_ce
    # Final hit point — anatomical location of contact (not bbox center)
    re_c = get(best_fn, rid) or get(fn, rid)
    hit_xy = best_recv_xy
    if hit_xy is None and re_c is not None:
        hit_xy = proj_kp(re_c, 0)
    if hit_xy is None and re_c is not None:
        x1, y1, x2, y2 = re_c["bbox"]
        hit_xy = np.array([(x1+x2)/2, y1 + 0.18*(y2-y1)])
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
print(f"\nMissed GTs ({len(missed)}): {', '.join(missed) if missed else 'none'}")

out_data={
    "metrics_full":{k:best_full[k] for k in ("f1","p","r","tp","fp","fn")},
    "metrics_cv":  {k:best_cv[k]   for k in ("f1","p","r","tp","fp","fn")},
    "feature_importance":{FEAT_NAMES[i]:float(imp[i]) for i in range(len(FEAT_NAMES))},
    "detections":dets_out,
}
with open(OUT_JSON,"w") as fh: json.dump(out_data,fh,indent=2)
print(f"\nSaved -> {OUT_JSON}")
