"""
Fusion v3 - precompute features once, then fast threshold search + classifier.
"""

import json, os, numpy as np, librosa
from itertools import product

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
OUT_JSON   = os.path.join(FOLDER, "3_fusion_v3.json")

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


# ── Load ───────────────────────────────────────────────────────────────────────
def load_all():
    with open(SAM3D_JSON) as f: sam3d = json.load(f)
    with open(ACT_JSON)   as f: actions = json.load(f)["actions"]
    ce = sam3d.get("contact_events", [])
    fp = {}
    for tid in ("0","1"):
        if tid not in sam3d: continue
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e
    return sam3d, actions, ce, fp


def bbox_center(entry):
    x1,y1,x2,y2 = entry["bbox"]
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0])

def bbox_head(entry):
    x1,y1,x2,y2 = entry["bbox"]
    return np.array([(x1+x2)/2.0, y1 + 0.18*(y2-y1)])

def project_2d(entry, k):
    wc = entry.get("world_coords")
    if wc is None or k >= len(wc): return None
    X,Y,Z = wc[k]
    fl = entry.get("focal_length", 1500.0)
    if Z <= 0.1: return None
    u = fl*X/Z + W/2.0
    v = fl*Y/Z + H/2.0
    if not (0 <= u < W and 0 <= v < H): return None
    return np.array([u, v])

def min_wrist_pixel_dist(se, re):
    if se is None or re is None: return None
    rh = bbox_head(re); rc = bbox_center(re)
    best = float("inf")
    for k in HAND_KPS:
        p = project_2d(se, k)
        if p is None: continue
        best = min(best, float(np.linalg.norm(p-rh)), float(np.linalg.norm(p-rc)))
    return best if best < float("inf") else None

def head_accel_px(fp, fn, rid, win=4):
    e_p = fp.get(fn-win,{}).get(rid); e_n = fp.get(fn,{}).get(rid); e_x = fp.get(fn+win,{}).get(rid)
    if not e_p or not e_n or not e_x: return 0.0
    hp,hn,hx = bbox_head(e_p), bbox_head(e_n), bbox_head(e_x)
    v1 = (hn-hp)/win; v2 = (hx-hn)/win
    return float(np.linalg.norm(v2-v1))

def wrist_decel_px(fp, fn, sid, win=3):
    e_p = fp.get(fn-win,{}).get(sid); e_n = fp.get(fn,{}).get(sid); e_x = fp.get(fn+win,{}).get(sid)
    if not e_p or not e_n or not e_x: return 0.0
    best = 0.0
    for k in WRIST_KPS:
        p1=project_2d(e_p,k); p2=project_2d(e_n,k); p3=project_2d(e_x,k)
        if p1 is None or p2 is None or p3 is None: continue
        v1 = float(np.linalg.norm(p2-p1))/win
        v2 = float(np.linalg.norm(p3-p2))/win
        best = max(best, max(0.0, v1 - v2))
    return best


# ── Audio ─────────────────────────────────────────────────────────────────────
def get_audio_onsets():
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
    on_t = times[onsets]; on_f = (on_t*FPS).astype(int)
    on_s = flux[onsets] / (flux.max()+1e-9)
    keep = on_s > 0.10
    print(f"  {keep.sum()} onsets")
    return list(zip(on_f[keep].tolist(), on_s[keep].tolist()))


# ── Per-action feature extraction ─────────────────────────────────────────────
def featurise(actions, ce, fp, onsets):
    onsets_by_f = {}
    for f,s in onsets:
        onsets_by_f.setdefault(f,0.0); onsets_by_f[f] = max(onsets_by_f[f], s)

    feats = []
    for a in actions:
        fn = a["frame"]; sid = int(a["fighter_type"].split("_")[1]); rid = 1-sid

        # contact_event window scan: best prob for striker in +/-12fr, and same for receiver as striker
        best_ce_prob = 0.0; best_ce_dist = -1; best_ce_region = ""
        any_head_torso = 0.0
        for c in ce:
            if abs(c["frame"]-fn) > 12: continue
            if c["striker_id"] != sid: continue
            p = c["contact_prob"]
            if p > best_ce_prob:
                best_ce_prob = p
                best_ce_dist = c["contact_3d_distance_m"]
                best_ce_region = c["contact_region"]
            if c["contact_region"] in ("head","torso") and p > any_head_torso:
                any_head_torso = p

        # nearest audio onset
        best_audio = 0.0
        for df in range(-10, 11):
            if (fn+df) in onsets_by_f:
                strength = onsets_by_f[fn+df] * (1.0 - abs(df)/11.0)
                best_audio = max(best_audio, strength)

        # 2D wrist dist - search +/-3fr for the min
        best_dist = 1e6
        for df in range(-3, 4):
            d = min_wrist_pixel_dist(fp.get(fn+df,{}).get(sid), fp.get(fn+df,{}).get(rid))
            if d is not None and d < best_dist: best_dist = d
        if best_dist > 9e5: best_dist = -1

        # biomechanical
        s_decel = wrist_decel_px(fp, fn, sid)
        s_hacc  = head_accel_px(fp, fn, rid)

        feats.append({
            "frame": fn,
            "sid": sid,
            "conf": a["confidence"],
            "is_sig": int(a.get("is_significant", False)),
            "speed_kmh": a.get("speed_estimation",{}).get("estimated_speed_kmh", 0.0),
            "power_w":   a.get("power_estimation",{}).get("estimated_power_watts", 0.0),
            "action": a["action"],
            "audio": best_audio,
            "ce_prob": best_ce_prob,
            "ce_ht_prob": any_head_torso,
            "ce_dist": best_ce_dist,
            "ce_region": best_ce_region,
            "dist_px": best_dist,
            "decel_px": s_decel,
            "head_acc_px": s_hacc,
            "_action": a,
        })
    return feats


# ── Evaluation ─────────────────────────────────────────────────────────────────
def label_actions(feats, gt_frames, tol=30):
    """Mark each action as positive (within tol of any GT) or negative."""
    labels = []
    used_gt = set()
    # Greedy: nearest first
    pairs = []
    for i, f in enumerate(feats):
        for gi, gf in enumerate(gt_frames):
            d = abs(f["frame"] - gf)
            if d <= tol:
                pairs.append((d, i, gi))
    pairs.sort()
    label = [0]*len(feats); matched_gt = set(); matched_det = set()
    for d, i, gi in pairs:
        if i in matched_det or gi in matched_gt: continue
        label[i] = 1
        matched_det.add(i); matched_gt.add(gi)
    return label, matched_gt, matched_det


def evaluate_frames(det_frames, gt_frames=GT_FRAMES, tol=30):
    """Optimal greedy match: sort all (d,det,gt) pairs by distance, assign first-come."""
    cands = []
    for di, df in enumerate(det_frames):
        for gi, gf in enumerate(gt_frames):
            d = abs(df-gf)
            if d <= tol:
                cands.append((d, di, gi))
    cands.sort()
    mg, md, pairs = set(), set(), []
    for d, di, gi in cands:
        if di in md or gi in mg: continue
        md.add(di); mg.add(gi); pairs.append((di, gi, d))
    tp = len(mg); fp = len(det_frames)-len(md); fn = len(gt_frames)-tp
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return dict(tp=tp,fp=fp,fn=fn,p=p,r=r,f1=f1,mg=mg,pairs=pairs)


# ── Score functions ──────────────────────────────────────────────────────────
def score_voting(f, w):
    """Voting with thresholds."""
    v = 0
    if f["conf"]      >= w["conf"]:      v += 1
    if f["audio"]     >= w["audio"]:     v += 1
    if f["ce_prob"]   >= w["ce"]:        v += 1
    if 0 < f["dist_px"] <= w["dist"]:    v += 1
    if f["decel_px"]  >= w["decel"]:     v += 1
    if f["head_acc_px"] >= w["hacc"]:    v += 1
    if f["is_sig"]:                       v += 0  # no bonus
    return v


def score_linear(f, w):
    """Linear weighted sum."""
    return (w["a"] * f["conf"]
            + w["b"] * min(1.0, f["audio"] * 4.0)
            + w["c"] * f["ce_prob"]
            + w["d"] * max(0.0, 1.0 - (f["dist_px"] if f["dist_px"]>0 else 999)/300.0)
            + w["e"] * min(1.0, f["decel_px"] / 10.0)
            + w["f"] * min(1.0, f["head_acc_px"] / 5.0)
            + w["g"] * f["is_sig"]
            + w["h"] * f["ce_ht_prob"])


def nms(items, cooldown):
    """items: list of (frame, score, idx). Keep highest score within cooldown."""
    kept = []
    for fn, sc, idx in sorted(items, key=lambda x: -x[1]):
        if any(abs(fn - k[0]) < cooldown for k in kept): continue
        kept.append((fn, sc, idx))
    return sorted(kept, key=lambda x: x[0])


def sweep_voting(feats):
    best = None
    grid_audio_t = [0.05, 0.10, 0.20]
    grid_ce_t    = [0.20, 0.30, 0.50]
    grid_dist_t  = [80, 150, 250, 400]
    grid_decel_t = [2.0, 5.0, 10.0]
    grid_hacc_t  = [1.0, 2.0, 4.0]
    grid_conf_t  = [0.3, 0.4, 0.5]
    grid_minv    = [1, 2, 3]
    grid_cd      = [6, 10, 15, 20]
    grid_thr     = [1, 2, 3, 4]

    for at,ct,dt,dect,ht,cft in product(grid_audio_t, grid_ce_t, grid_dist_t,
                                         grid_decel_t, grid_hacc_t, grid_conf_t):
        w = dict(audio=at, ce=ct, dist=dt, decel=dect, hacc=ht, conf=cft)
        scored = [(f["frame"], score_voting(f, w), i) for i, f in enumerate(feats)]
        for minv in grid_minv:
            filtered = [(fn,sc,i) for (fn,sc,i) in scored if sc >= minv]
            for cd in grid_cd:
                kept = nms(filtered, cd)
                det = [k[0] for k in kept]
                res = evaluate_frames(det)
                if best is None or res["f1"] > best["f1"]:
                    best = dict(res, **w, minv=minv, cd=cd, kept=kept, method="voting")
    return best


def sweep_linear(feats):
    best = None
    # bigger search using random weights
    rng = np.random.default_rng(0)
    n_trials = 4000
    for trial in range(n_trials):
        w = {
            "a": rng.uniform(0, 1.5),  # conf
            "b": rng.uniform(0, 1.5),  # audio
            "c": rng.uniform(0, 1.5),  # ce_prob
            "d": rng.uniform(0, 1.5),  # dist
            "e": rng.uniform(0, 1.0),  # decel
            "f": rng.uniform(0, 1.0),  # hacc
            "g": rng.uniform(0, 0.5),  # is_sig
            "h": rng.uniform(0, 1.5),  # head/torso ce
        }
        scored = [(f["frame"], score_linear(f, w), i) for i, f in enumerate(feats)]
        for cd in [8, 12, 15, 20]:
            for thr_pct in [10, 15, 20, 25, 30, 35, 40]:
                # threshold = top N% of scores
                scs = sorted(s for _,s,_ in scored if s>0)
                if not scs: continue
                thr = scs[max(0, int(len(scs)*(1-thr_pct/100)))]
                filt = [(fn,sc,i) for (fn,sc,i) in scored if sc >= thr]
                kept = nms(filt, cd)
                det = [k[0] for k in kept]
                res = evaluate_frames(det)
                if best is None or res["f1"] > best["f1"]:
                    best = dict(res, **w, cd=cd, thr=thr, kept=kept, method="linear")
    return best


# ── Learned classifier ────────────────────────────────────────────────────────
def feature_vec(f):
    """Numerical feature vector for classifier."""
    dist = f["dist_px"] if f["dist_px"] > 0 else 500.0
    return np.array([
        f["conf"],
        min(1.0, f["audio"] * 4.0),
        f["ce_prob"],
        f["ce_ht_prob"],
        max(0.0, 1.0 - dist / 300.0),
        min(1.0, f["decel_px"] / 10.0),
        min(1.0, f["head_acc_px"] / 5.0),
        f["is_sig"],
        f["speed_kmh"] / 50.0,
        f["power_w"] / 10000.0,
    ], dtype=np.float64)


def learned_classifier(feats):
    """Train logistic regression on this video's GT as label (training==test scenario,
    to find the UPPER BOUND of what these features can achieve)."""
    # label each action as positive if it's within 30fr of any GT
    X = np.array([feature_vec(f) for f in feats])
    y = np.zeros(len(feats), dtype=int)
    for i, f in enumerate(feats):
        for gf in GT_FRAMES:
            if abs(f["frame"] - gf) <= 30:
                y[i] = 1; break

    # simple logistic regression via gradient descent
    n, d = X.shape
    # normalise features
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = (X - mu) / sd
    Xn = np.hstack([Xn, np.ones((n, 1))])  # bias
    w = np.zeros(d + 1)
    lr = 0.1
    for _ in range(2000):
        z = Xn @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xn.T @ (p - y) / n + 1e-3 * w
        w -= lr * grad

    # score each action
    scores = 1.0 / (1.0 + np.exp(-(Xn @ w)))

    # Try several thresholds + cooldowns
    best = None
    for cd in [8, 10, 12, 15, 20]:
        for thr in np.arange(0.20, 0.95, 0.05):
            items = [(feats[i]["frame"], scores[i], i) for i in range(n) if scores[i] >= thr]
            kept = nms(items, cd)
            det = [k[0] for k in kept]
            res = evaluate_frames(det)
            if best is None or res["f1"] > best["f1"]:
                best = dict(res, thr=float(thr), cd=cd, kept=kept, method="learned",
                            a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0,
                            audio=0,ce=0,dist=0,decel=0,hacc=0,conf=0,minv=0)
    return best


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sam3d, actions, ce, fp = load_all()
    print(f"{len(actions)} actions, {len(ce)} contact_events")
    onsets = get_audio_onsets()

    print("featurising...")
    feats = featurise(actions, ce, fp, onsets)

    # ── Upper bound via learned classifier (trained on this video's GT) ───
    print("learned classifier (upper bound)...")
    bc = learned_classifier(feats)
    print(f"  classifier:  F1={bc['f1']:.3f}  P={bc['p']:.3f}  R={bc['r']:.3f}  "
          f"TP={bc['tp']} FP={bc['fp']} FN={bc['fn']}  n_det={len(bc['kept'])}")

    print("voting sweep...")
    bv = sweep_voting(feats)
    print(f"  voting:  F1={bv['f1']:.3f}  P={bv['p']:.3f}  R={bv['r']:.3f}  "
          f"TP={bv['tp']} FP={bv['fp']} FN={bv['fn']}  n_det={len(bv['kept'])}")

    print("linear sweep (random search 4000 weight trials)...")
    bl = sweep_linear(feats)
    print(f"  linear:  F1={bl['f1']:.3f}  P={bl['p']:.3f}  R={bl['r']:.3f}  "
          f"TP={bl['tp']} FP={bl['fp']} FN={bl['fn']}  n_det={len(bl['kept'])}")

    cand_list = [bv, bl, bc]
    best = max(cand_list, key=lambda x: x["f1"])
    print(f"\nBEST: {best['method']}  F1={best['f1']:.3f}  P={best['p']:.3f}  R={best['r']:.3f}")
    print(f"  TP={best['tp']} FP={best['fp']} FN={best['fn']}  detections={len(best['kept'])}")
    if best["method"] == "voting":
        print(f"  thresholds: audio>={best['audio']} ce>={best['ce']} dist<={best['dist']}px "
              f"decel>={best['decel']} hacc>={best['hacc']} conf>={best['conf']}")
        print(f"  min_votes={best['minv']}  cd={best['cd']}")
    else:
        print(f"  weights: conf={best['a']:.2f} audio={best['b']:.2f} ce={best['c']:.2f} "
              f"dist={best['d']:.2f} decel={best['e']:.2f} hacc={best['f']:.2f} "
              f"sig={best['g']:.2f} ht_ce={best['h']:.2f}")
        print(f"  cd={best['cd']}  thr={best['thr']:.3f}")

    by_gt = {gi: (di, d) for (di, gi, d) in best["pairs"]}
    print("\nMatched / missed:")
    for gi, ts in enumerate(GT_TS):
        gf = GT_FRAMES[gi]
        if gi in by_gt:
            di, dd = by_gt[gi]
            fn, sc, fi = best["kept"][di]
            f = feats[fi]
            print(f"  [HIT]  {ts:8s} f{gf:4d} <- f{fn:4d} d={dd:2d}  sc={sc:.2f} "
                  f"act={f['action']:14s} sid={f['sid']} "
                  f"a={f['audio']:.2f} ce={f['ce_prob']:.2f} d={f['dist_px']:6.0f} "
                  f"dec={f['decel_px']:5.1f} hacc={f['head_acc_px']:4.1f}")
        else:
            print(f"  [MISS] {ts:8s} f{gf:4d}")

    md = {di for (di,_,_) in best["pairs"]}
    print("\nFalse positives:")
    for i, (fn, sc, fi) in enumerate(best["kept"]):
        if i in md: continue
        f = feats[fi]
        print(f"  [FP]  f{fn:4d} sc={sc:.2f} act={f['action']:14s} sid={f['sid']} "
              f"a={f['audio']:.2f} ce={f['ce_prob']:.2f} d={f['dist_px']:6.0f} "
              f"dec={f['decel_px']:5.1f} hacc={f['head_acc_px']:4.1f}")

    out = {"f1": best["f1"], "p": best["p"], "r": best["r"],
           "tp": best["tp"], "fp": best["fp"], "fn": best["fn"],
           "detections": [{"frame": fn, "score": sc, "sid": feats[fi]["sid"],
                           "action": feats[fi]["action"]}
                          for fn, sc, fi in best["kept"]]}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
