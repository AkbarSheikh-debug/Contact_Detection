"""
Scan ±60 frames around each GT timestamp and suggest the true impact frame
using three signals:
  1. Minimum 2D wrist-to-receiver-head distance (geometry)
  2. Audio multi-band flux peak
  3. Receiver head sudden displacement (reaction)

Outputs a text report + saves corrected GT timestamps to relabeled_gt.json
"""
import json, os, numpy as np, librosa

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
OUT_JSON   = os.path.join(FOLDER, "relabeled_gt.json")

FPS  = 24.995
W, H = 1920, 1080

GT_TS = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
         "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
         "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
         "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]

WRIST_KPS = [9, 10]        # left/right wrist
ELBOW_KPS = [7, 8]         # left/right elbow
HEAD_KPS  = [0, 1, 2, 3, 4]   # nose, eyes, ears
TORSO_KPS = [5, 6, 11, 12]    # shoulders + hips
ALL_RECV_KPS = HEAD_KPS + TORSO_KPS  # receiver keypoints to check

SCAN_RADIUS = 60   # frames either side of GT to scan

def to_f(ts):
    p = ts.split(":")
    s = int(p[0]) + int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s * FPS))

def to_ts(fn):
    t = fn / FPS
    m = int(t // 60); s = t - m*60
    si = int(s); fr = int(round((s - si)*FPS))
    return f"{m}:{si:02d}:{fr:02d}"

GT_FRAMES = [to_f(ts) for ts in GT_TS]

print("Loading SAM3D...")
with open(SAM3D_JSON) as f: sam3d = json.load(f)
with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
ce_events = sam3d.get("contact_events", [])

fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e

def get(fn, tid): return fp.get(fn, {}).get(tid)

def proj(e, k):
    wc = e.get("world_coords")
    if wc is None or k >= len(wc): return None
    X, Y, Z = wc[k]
    fl = e.get("focal_length", 1500.0)
    if Z <= 0.1: return None
    u = fl*X/Z + W/2.0
    v = fl*Y/Z + H/2.0
    if not (0 <= u < W and 0 <= v < H): return None
    return np.array([u, v])

def bb_head(e):
    x1,y1,x2,y2 = e["bbox"]
    return np.array([(x1+x2)/2.0, y1 + 0.18*(y2-y1)])

def bb_cen(e):
    x1,y1,x2,y2 = e["bbox"]
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0])


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

def audio_at(fn, rad=4):
    t = fn / FPS
    idx = int(round(t * sr / hop))
    r   = int(round(rad / FPS * sr / hop))
    lo, hi = max(0, idx-r), min(len(band_flux["mid"]), idx+r+1)
    vals = [band_flux[n][lo:hi].max() if hi > lo else 0.0 for n in ("low","mid","high")]
    return max(vals)


# ── Per-frame signals (bbox-based, no broken world_coords projection) ─────
def wrist_to_recv_dist(fn, sid):
    """Striker fist (bbox edge) to receiver head distance."""
    rid = 1 - sid
    se = get(fn, sid); re = get(fn, rid)
    if se is None or re is None:
        return 999.0, None, None
    sx1,sy1,sx2,sy2 = se["bbox"]
    rx1,ry1,rx2,ry2 = re["bbox"]
    sc_x = (sx1+sx2)/2; rc_x = (rx1+rx2)/2
    fx = sx2 if sc_x < rc_x else sx1
    fy = sy1 + 0.28*(sy2-sy1)
    fist = np.array([fx, fy])
    rh   = bb_head(re)
    d    = float(np.linalg.norm(fist - rh))
    return d, fist, rh

def recv_head_displacement(fn, rid):
    """Displacement of receiver's head between fn-1 and fn+1."""
    e0 = get(fn-1, rid); e1 = get(fn+1, rid)
    if e0 is None or e1 is None: return 0.0
    return float(np.linalg.norm(bb_head(e1) - bb_head(e0)))


# ── Scan each GT ─────────────────────────────────────────────────────────
print("\nScanning GT frames...\n")
print(f"{'GT_TS':12s}  {'GT_F':5s}  {'Best_F':6s}  {'Best_TS':12s}  {'Delta':6s}  {'dist_px':8s}  {'audio':6s}  {'react':6s}  {'score':6s}")
print("-"*90)

results = []

for gi, (gt_ts, gf) in enumerate(zip(GT_TS, GT_FRAMES)):
    # We don't know which fighter is the striker a priori.
    # Try both and pick the better signal (smaller wrist dist).
    best_score = -1e9
    best_fn = gf
    best_sid = 0
    best_dist = 999.0
    best_audio = 0.0
    best_react = 0.0
    best_rp = None

    for df in range(-SCAN_RADIUS, SCAN_RADIUS+1):
        fn = gf + df
        if fn < 0: continue
        aud = audio_at(fn)
        for sid in (0, 1):
            rid = 1 - sid
            dist, sp, rp = wrist_to_recv_dist(fn, sid)
            if dist > 900: continue
            react = recv_head_displacement(fn, rid)

            # Normalised distance penalty: 0 at 0px, -1 at 500px
            dist_score = max(0.0, 1.0 - dist / 500.0)
            # Audio score already 0..1
            # React score: 0..1 (clip at 50px)
            react_score = min(react / 50.0, 1.0)

            score = 0.55 * dist_score + 0.30 * aud + 0.15 * react_score

            if score > best_score:
                best_score = score
                best_fn = fn
                best_sid = sid
                best_dist = dist
                best_audio = aud
                best_react = react
                best_rp = rp

    delta = best_fn - gf
    best_ts = to_ts(best_fn)
    print(f"{gt_ts:12s}  {gf:5d}  {best_fn:6d}  {best_ts:12s}  {delta:+6d}  {best_dist:8.1f}  {best_audio:.4f}  {best_react:6.1f}  {best_score:.4f}")

    results.append({
        "original_ts":  gt_ts,
        "original_frame": gf,
        "suggested_ts":  best_ts,
        "suggested_frame": best_fn,
        "delta_frames":  delta,
        "striker_id":    best_sid,
        "dist_px":       round(best_dist, 1),
        "audio_peak":    round(float(best_audio), 4),
        "recv_react_px": round(float(best_react), 1),
        "score":         round(float(best_score), 4),
        "hit_xy":        [round(float(best_rp[0]),1), round(float(best_rp[1]),1)] if best_rp is not None else None,
    })

with open(OUT_JSON, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nSaved -> {OUT_JSON}")

# Print summary: how many GTs shifted significantly
big = [r for r in results if abs(r["delta_frames"]) > 10]
print(f"\n{len(big)}/{len(results)} GTs shift >10 frames from original label:")
for r in big:
    print(f"  {r['original_ts']} -> {r['suggested_ts']}  (delta={r['delta_frames']:+d}  dist={r['dist_px']:.0f}px)")
