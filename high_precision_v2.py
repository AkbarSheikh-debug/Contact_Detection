"""
Push for P>=0.80 with MAXIMUM RECALL.
Strategy: combine 3+ corroborating signals per detection.
"""
import json, os, numpy as np, librosa

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
OUT_JSON   = os.path.join(FOLDER, "3_high_precision_v2.json")
FPS = 24.995; W, H = 1920, 1080

GT_TS = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
         "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
         "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
         "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]
def to_f(ts):
    p=ts.split(":")
    s=int(p[0])+int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s*FPS))
GT_FRAMES = [to_f(t) for t in GT_TS]

with open(SAM3D_JSON) as f: sam3d = json.load(f)
with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
ce_events = sam3d.get("contact_events", [])
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e

WRIST_KPS=[9,10]; HAND_KPS=[9,10,7,8]
def bbcen(e): x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.0,(y1+y2)/2.0])
def bbhead(e): x1,y1,x2,y2=e["bbox"]; return np.array([(x1+x2)/2.0,y1+0.18*(y2-y1)])
def proj(e,k):
    wc=e.get("world_coords")
    if wc is None or k>=len(wc): return None
    X,Y,Z=wc[k]; fl=e.get("focal_length",1500.0)
    if Z<=0.1: return None
    u=fl*X/Z+W/2.0; v=fl*Y/Z+H/2.0
    if not (0<=u<W and 0<=v<H): return None
    return np.array([u,v])
def wd(se,re):
    if se is None or re is None: return None
    rh,rc=bbhead(re),bbcen(re); best=float("inf")
    for k in HAND_KPS:
        p=proj(se,k)
        if p is None: continue
        best=min(best,float(np.linalg.norm(p-rh)),float(np.linalg.norm(p-rc)))
    return best if best<float("inf") else None

def hacc(fn, rid, win=4):
    e_p=fp.get(fn-win,{}).get(rid); e_n=fp.get(fn,{}).get(rid); e_x=fp.get(fn+win,{}).get(rid)
    if not e_p or not e_n or not e_x: return 0.0
    hp,hn,hx=bbhead(e_p),bbhead(e_n),bbhead(e_x)
    return float(np.linalg.norm((hx-hn)/win - (hn-hp)/win))

def wdecel(fn, sid, win=3):
    e_p=fp.get(fn-win,{}).get(sid); e_n=fp.get(fn,{}).get(sid); e_x=fp.get(fn+win,{}).get(sid)
    if not e_p or not e_n or not e_x: return 0.0
    best=0.0
    for k in WRIST_KPS:
        p1=proj(e_p,k); p2=proj(e_n,k); p3=proj(e_x,k)
        if p1 is None or p2 is None or p3 is None: continue
        v1=float(np.linalg.norm(p2-p1))/win; v2=float(np.linalg.norm(p3-p2))/win
        best=max(best,max(0.0,v1-v2))
    return best

print("audio...")
y, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
y = np.append(y[0], y[1:]-0.97*y[:-1])
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
fr = librosa.fft_frequencies(sr=sr, n_fft=2048); band = (fr>=1000)&(fr<=6000)
flux = np.maximum(0, np.diff(S[band], axis=1)).sum(axis=0).astype(np.float64)
flux = np.concatenate([[0.0], flux]); flux -= np.median(flux); flux = np.maximum(0, flux)
times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=256)
onsets = librosa.util.peak_pick(flux, pre_max=20, post_max=20, pre_avg=30, post_avg=30,
                                 delta=float(flux.std()*1.0), wait=12)
on_f = (times[onsets]*FPS).astype(int); on_s = flux[onsets]/(flux.max()+1e-9)
keep = on_s>0.05
on_by_f = {}
for f,s in zip(on_f[keep].tolist(), on_s[keep].tolist()):
    on_by_f.setdefault(f,0.0); on_by_f[f] = max(on_by_f[f], s)
print(f"  {len(on_by_f)} onsets")

print("featurising...")
feats=[]
for a in actions:
    fn=a["frame"]; sid=int(a["fighter_type"].split("_")[1]); rid=1-sid
    ce_best=0.0; ce_ht=0.0
    for c in ce_events:
        if abs(c["frame"]-fn)>12 or c["striker_id"]!=sid: continue
        if c["contact_prob"]>ce_best: ce_best=c["contact_prob"]
        if c["contact_region"] in ("head","torso") and c["contact_prob"]>ce_ht: ce_ht=c["contact_prob"]
    ab=0.0
    for df in range(-10,11):
        if (fn+df) in on_by_f: ab=max(ab, on_by_f[fn+df]*(1.0-abs(df)/11.0))
    db=1e6
    for df in range(-3,4):
        d=wd(fp.get(fn+df,{}).get(sid), fp.get(fn+df,{}).get(rid))
        if d is not None and d<db: db=d
    feats.append({
        "frame":fn,"sid":sid,"conf":a["confidence"],
        "is_sig":int(a.get("is_significant",False)),
        "speed":a.get("speed_estimation",{}).get("estimated_speed_kmh",0.0),
        "action":a["action"],"audio":ab,"ce":ce_best,"ce_ht":ce_ht,
        "dist":db if db<9e5 else -1,
        "decel":wdecel(fn,sid),"hacc":hacc(fn,rid),
    })

def evaluate(det):
    pairs=[(abs(df-gf),di,gi) for di,df in enumerate(det) for gi,gf in enumerate(GT_FRAMES) if abs(df-gf)<=30]
    pairs.sort(); md,mg,matched=set(),set(),[]
    for d,di,gi in pairs:
        if di in md or gi in mg: continue
        md.add(di); mg.add(gi); matched.append((di,gi,d))
    tp=len(mg); fp=len(det)-len(md); fn=len(GT_FRAMES)-tp
    p=tp/(tp+fp) if (tp+fp) else 0; r=tp/(tp+fn) if (tp+fn) else 0
    f1=2*p*r/(p+r) if (p+r) else 0
    return dict(tp=tp,fp=fp,fn=fn,p=p,r=r,f1=f1,matched=matched,md=md,mg=mg)

def nms(items,cd):
    kept=[]
    for it in sorted(items,key=lambda x:-x[1]):
        if any(abs(it[0]-k[0])<cd for k in kept): continue
        kept.append(it)
    return sorted(kept,key=lambda x:x[0])

# ── Many-signal voting with relaxed/strict modes ──────────────────────────────
def vote_score(f, t):
    """Return (votes, score) using thresholds dict `t`."""
    v = 0; s = 0.0
    if f["conf"]      >= t["c"]:  v += 1; s += f["conf"]
    if f["audio"]     >= t["a"]:  v += 1; s += f["audio"]
    if f["ce"]        >= t["e"]:  v += 1; s += f["ce"]
    if f["ce_ht"]     >= t["h"]:  v += 1; s += f["ce_ht"]
    if 0 < f["dist"] <= t["d"]:   v += 1; s += 1.0 - f["dist"]/t["d"]
    if f["decel"]     >= t["dc"]: v += 1; s += min(1.0, f["decel"]/30.0)
    if f["hacc"]      >= t["ha"]: v += 1; s += min(1.0, f["hacc"]/30.0)
    if f["is_sig"]:               v += 1; s += 0.5
    return v, s

print("\nSearching: P>=0.80, max recall (multi-vote sweep)...")
results = []
import itertools
# Sweep thresholds and min_votes
for cmin in [0.40, 0.50, 0.60, 0.70]:
    for amin in [0.05, 0.10, 0.20, 0.30]:
        for emin in [0.30, 0.50, 0.70]:
            for hmin in [0.30, 0.50, 0.70]:
                for dmax in [200, 300, 400]:
                    for dcmin in [3.0, 8.0]:
                        for hamin in [3.0, 8.0]:
                            t = dict(c=cmin, a=amin, e=emin, h=hmin, d=dmax, dc=dcmin, ha=hamin)
                            scored = [(f["frame"], *vote_score(f, t), f) for f in feats]
                            for minv in [2, 3, 4]:
                                for cd in [10, 15, 20]:
                                    items = [(fn, sc, f) for (fn, v, sc, f) in scored if v >= minv]
                                    kept = nms(items, cd)
                                    det = [k[0] for k in kept]
                                    if not det: continue
                                    res = evaluate(det)
                                    results.append((res["p"], res["r"], res["f1"], len(det),
                                                    dict(t=t, minv=minv, cd=cd), res, kept))

hp = [r for r in results if r[0] >= 0.80]
print(f"P>=0.80 configs: {len(hp)} / {len(results)}")
if hp:
    hp.sort(key=lambda x: (-x[1], -x[2]))
    print("\nTop 15 by recall (P>=0.80):")
    for p, r, f1, n, cfg, res, kept in hp[:15]:
        t = cfg["t"]
        print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} n={n}  "
              f"c>={t['c']} a>={t['a']} e>={t['e']} h>={t['h']} d<={t['d']} "
              f"dc>={t['dc']} ha>={t['ha']} v>={cfg['minv']} cd={cfg['cd']}")

# Also relax P slightly
print("\nBy F1 (no P constraint):")
results.sort(key=lambda x: -x[2])
for p,r,f1,n,cfg,res,kept in results[:5]:
    t = cfg["t"]
    print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} n={n}  "
          f"c>={t['c']} a>={t['a']} e>={t['e']} h>={t['h']} d<={t['d']} "
          f"dc>={t['dc']} ha>={t['ha']} v>={cfg['minv']} cd={cfg['cd']}")

# Pick best: highest recall among P>=0.80, else highest F1
if hp:
    best = hp[0]; mode = "P>=0.80"
else:
    best = results[0]; mode = "best F1"

p, r, f1, n, cfg, res, kept = best
print(f"\nSELECTED ({mode}): P={p:.3f} R={r:.3f} F1={f1:.3f} n_detections={n}")
matched_di = {di for (di,_,_) in res["matched"]}
gt_by_det = {di: gi for (di, gi, _) in res["matched"]}

print("\nDetections:")
for i, (fn, sc, f) in enumerate(kept):
    tag = "TP" if i in matched_di else "FP"
    info = f"gt={GT_TS[gt_by_det[i]]}" if i in matched_di else ""
    print(f"  [{tag}] f{fn:4d} sc={sc:.2f} {f['action']:14s} sid={f['sid']} "
          f"a={f['audio']:.2f} ce={f['ce']:.2f} d={f['dist']:6.0f} "
          f"dec={f['decel']:.1f} hacc={f['hacc']:.1f} {info}")

# Missed GTs
print("\nMissed GTs:")
matched_gi = {gi for (_,gi,_) in res["matched"]}
for gi, ts in enumerate(GT_TS):
    if gi not in matched_gi:
        print(f"  [MISS] {ts:8s} f{GT_FRAMES[gi]:4d}")

out = {"mode": mode, "config": cfg, "metrics": {"p":p,"r":r,"f1":f1,"tp":res["tp"],"fp":res["fp"],"fn":res["fn"]},
       "detections": [{"frame": fn, "score": float(sc), "sid": f["sid"],
                       "action": f["action"], "audio": f["audio"], "ce": f["ce"],
                       "dist_px": f["dist"], "speed_kmh": f["speed"]}
                      for fn, sc, f in kept]}
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved -> {OUT_JSON}")
