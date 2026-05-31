"""
Find a config that hits PRECISION >= 0.80, maximising recall under that constraint.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, os, numpy as np, librosa
from itertools import product

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
OUT_JSON   = os.path.join(FOLDER, "3_high_precision.json")
FPS = 24.995
W, H = 1920, 1080

GT_TS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]
def to_f(ts):
    p = ts.split(":")
    s = int(p[0]) + int(p[1])/FPS if len(p)==2 else int(p[0])*60 + int(p[1]) + int(p[2])/FPS
    return int(round(s*FPS))
GT_FRAMES = [to_f(t) for t in GT_TS]

WRIST_KPS = [9, 10]
HAND_KPS  = [9, 10, 7, 8]


# ── Load + features ──────────────────────────────────────────────────────────
with open(SAM3D_JSON) as f: sam3d = json.load(f)
with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
ce_events = sam3d.get("contact_events", [])
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e

def bbox_center(e):
    x1,y1,x2,y2 = e["bbox"]; return np.array([(x1+x2)/2.0,(y1+y2)/2.0])
def bbox_head(e):
    x1,y1,x2,y2 = e["bbox"]; return np.array([(x1+x2)/2.0, y1+0.18*(y2-y1)])
def proj2d(e,k):
    wc = e.get("world_coords");
    if wc is None or k>=len(wc): return None
    X,Y,Z = wc[k]; fl = e.get("focal_length",1500.0)
    if Z<=0.1: return None
    u = fl*X/Z + W/2.0; v = fl*Y/Z + H/2.0
    if not (0<=u<W and 0<=v<H): return None
    return np.array([u,v])

def wrist_dist(se, re):
    if se is None or re is None: return None
    rh = bbox_head(re); rc = bbox_center(re)
    best = float("inf")
    for k in HAND_KPS:
        p = proj2d(se,k)
        if p is None: continue
        best = min(best, float(np.linalg.norm(p-rh)), float(np.linalg.norm(p-rc)))
    return best if best<float("inf") else None

def head_accel(fn, rid, win=4):
    e_p=fp.get(fn-win,{}).get(rid); e_n=fp.get(fn,{}).get(rid); e_x=fp.get(fn+win,{}).get(rid)
    if not e_p or not e_n or not e_x: return 0.0
    h_p,h_n,h_x = bbox_head(e_p),bbox_head(e_n),bbox_head(e_x)
    v1 = (h_n-h_p)/win; v2 = (h_x-h_n)/win
    return float(np.linalg.norm(v2-v1))

def wrist_decel(fn, sid, win=3):
    e_p=fp.get(fn-win,{}).get(sid); e_n=fp.get(fn,{}).get(sid); e_x=fp.get(fn+win,{}).get(sid)
    if not e_p or not e_n or not e_x: return 0.0
    best = 0.0
    for k in WRIST_KPS:
        p1=proj2d(e_p,k); p2=proj2d(e_n,k); p3=proj2d(e_x,k)
        if p1 is None or p2 is None or p3 is None: continue
        v1=float(np.linalg.norm(p2-p1))/win; v2=float(np.linalg.norm(p3-p2))/win
        best = max(best, max(0.0, v1-v2))
    return best


# ── Audio ────────────────────────────────────────────────────────────────────
print("audio...")
y, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
y = np.append(y[0], y[1:]-0.97*y[:-1])
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
band = (fr>=1000)&(fr<=6000)
flux = np.maximum(0, np.diff(S[band], axis=1)).sum(axis=0).astype(np.float64)
flux = np.concatenate([[0.0], flux])
flux -= np.median(flux); flux = np.maximum(0, flux)
times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=256)
onsets = librosa.util.peak_pick(flux, pre_max=20, post_max=20, pre_avg=30, post_avg=30,
                                 delta=float(flux.std()*1.0), wait=12)
on_f = (times[onsets]*FPS).astype(int)
on_s = flux[onsets]/(flux.max()+1e-9)
keep = on_s>0.05
audio_onsets = list(zip(on_f[keep].tolist(), on_s[keep].tolist()))
on_by_f = {}
for f,s in audio_onsets:
    on_by_f.setdefault(f,0.0); on_by_f[f] = max(on_by_f[f], s)
print(f"  {len(audio_onsets)} onsets")


# ── Featurise once ───────────────────────────────────────────────────────────
print("featurising...")
feats = []
for a in actions:
    fn = a["frame"]; sid = int(a["fighter_type"].split("_")[1]); rid = 1-sid
    ce_best = 0.0; ce_ht = 0.0
    for c in ce_events:
        if abs(c["frame"]-fn) > 12 or c["striker_id"] != sid: continue
        if c["contact_prob"] > ce_best: ce_best = c["contact_prob"]
        if c["contact_region"] in ("head","torso") and c["contact_prob"] > ce_ht:
            ce_ht = c["contact_prob"]
    audio_best = 0.0
    for df in range(-10,11):
        if (fn+df) in on_by_f:
            audio_best = max(audio_best, on_by_f[fn+df]*(1.0-abs(df)/11.0))
    dist_best = 1e6
    for df in range(-3,4):
        d = wrist_dist(fp.get(fn+df,{}).get(sid), fp.get(fn+df,{}).get(rid))
        if d is not None and d < dist_best: dist_best = d
    feats.append({
        "frame": fn, "sid": sid, "conf": a["confidence"],
        "is_sig": int(a.get("is_significant", False)),
        "speed": a.get("speed_estimation",{}).get("estimated_speed_kmh",0.0),
        "action": a["action"],
        "audio": audio_best, "ce": ce_best, "ce_ht": ce_ht,
        "dist": dist_best if dist_best<9e5 else -1,
        "decel": wrist_decel(fn, sid),
        "hacc": head_accel(fn, rid),
        "_a": a,
    })


# ── Evaluation (optimal greedy by min distance) ──────────────────────────────
def evaluate(det):
    pairs = []
    for di, df in enumerate(det):
        for gi, gf in enumerate(GT_FRAMES):
            d = abs(df-gf)
            if d <= 30:
                pairs.append((d, di, gi))
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


# ── Sweep: HIGH PRECISION mode ───────────────────────────────────────────────
print("\nSearching for configs with P >= 0.80, maximising recall...")
results = []

# Strategy: require strong evidence - high conf AND (audio OR ce OR dist)
# Sweep many strict combinations
for conf_min in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for audio_min in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        for ce_min in [0.0, 0.20, 0.40, 0.60, 0.80]:
            for dist_max in [60, 100, 150, 250, 9999]:
                for require in [
                    "any",        # any of audio/ce/dist
                    "audio_or_ce",
                    "audio_and_dist",
                    "ce_or_dist",
                    "audio_only",
                    "ce_only",
                    "dist_only",
                ]:
                    cands = []
                    for f in feats:
                        if f["conf"] < conf_min: continue
                        ok_audio = f["audio"] >= audio_min
                        ok_ce    = f["ce"]    >= ce_min
                        ok_dist  = (0 < f["dist"] <= dist_max)

                        if require == "any":            keep = ok_audio or ok_ce or ok_dist
                        elif require == "audio_or_ce":  keep = ok_audio or ok_ce
                        elif require == "audio_and_dist": keep = ok_audio and ok_dist
                        elif require == "ce_or_dist":   keep = ok_ce or ok_dist
                        elif require == "audio_only":   keep = ok_audio
                        elif require == "ce_only":      keep = ok_ce
                        elif require == "dist_only":    keep = ok_dist
                        if not keep: continue
                        score = (f["conf"] + 0.5*f["audio"] + 0.5*f["ce"] +
                                 0.3*(1 if ok_dist else 0))
                        cands.append((f["frame"], score, f))

                    for cd in [8, 12, 15, 20, 25]:
                        kept = nms(cands, cd)
                        det = [k[0] for k in kept]
                        if not det: continue
                        res = evaluate(det)
                        results.append((res["p"], res["r"], res["f1"], len(det),
                                        dict(conf_min=conf_min, audio_min=audio_min,
                                             ce_min=ce_min, dist_max=dist_max,
                                             require=require, cd=cd), res, kept))

# Filter to P >= 0.80, sort by recall
hp = [r for r in results if r[0] >= 0.80]
print(f"\nConfigs with P>=0.80: {len(hp)} / {len(results)}")
if hp:
    hp.sort(key=lambda x: (-x[1], -x[2]))  # max recall, then F1
    print("\nTop 10 high-precision configs (by recall):")
    for p, r, f1, n, cfg, res, kept in hp[:10]:
        print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={n}  "
              f"conf>={cfg['conf_min']} aud>={cfg['audio_min']} ce>={cfg['ce_min']} "
              f"dist<={cfg['dist_max']} req={cfg['require']} cd={cfg['cd']}")
else:
    # No config achieves P>=0.80, find the highest P
    results.sort(key=lambda x: -x[0])
    print("\nNo P>=0.80 found. Top 10 by precision:")
    for p, r, f1, n, cfg, res, kept in results[:10]:
        print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={n}  "
              f"conf>={cfg['conf_min']} aud>={cfg['audio_min']} ce>={cfg['ce_min']} "
              f"dist<={cfg['dist_max']} req={cfg['require']} cd={cfg['cd']}")

# Also find best F1 overall
results.sort(key=lambda x: -x[2])
print("\nTop 5 by F1:")
for p, r, f1, n, cfg, res, kept in results[:5]:
    print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={n}  "
          f"conf>={cfg['conf_min']} aud>={cfg['audio_min']} ce>={cfg['ce_min']} "
          f"dist<={cfg['dist_max']} req={cfg['require']} cd={cfg['cd']}")

# Pick "best": P>=0.80 if available, else highest F1 with P>=0.70, else highest F1
if hp:
    best = hp[0]
    print(f"\nSELECTED (high precision): P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f}")
else:
    p70 = [r for r in results if r[0] >= 0.70]
    if p70:
        p70.sort(key=lambda x: -x[2])
        best = p70[0]
        print(f"\nSELECTED (P>=0.70 fallback): P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f}")
    else:
        best = results[0]
        print(f"\nSELECTED (best F1): P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f}")

p, r, f1, n, cfg, res, kept = best
print(f"\nDetections ({len(kept)}):")
matched_di = {di for (di,_,_) in res["matched"]}
gt_by_det = {di: gi for (di, gi, _) in res["matched"]}
for i, (fn, sc, f) in enumerate(kept):
    tag = "TP" if i in matched_di else "FP"
    info = f"gt={GT_TS[gt_by_det[i]]}" if i in matched_di else ""
    print(f"  [{tag}] f{fn:4d} sc={sc:.2f} {f['action']:14s} sid={f['sid']} "
          f"a={f['audio']:.2f} ce={f['ce']:.2f} d={f['dist']:6.0f} {info}")

# Save
out = {"config": cfg, "metrics": {"p":p,"r":r,"f1":f1,"tp":res["tp"],"fp":res["fp"],"fn":res["fn"]},
       "detections": [{"frame": fn, "score": float(sc), "sid": f["sid"],
                       "action": f["action"], "audio": f["audio"], "ce": f["ce"],
                       "dist_px": f["dist"], "speed_kmh": f["speed"]}
                      for fn, sc, f in kept]}
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved -> {OUT_JSON}")
