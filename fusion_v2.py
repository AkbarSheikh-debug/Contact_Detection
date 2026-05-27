"""
Fusion v2 — voting-based impact detection.

Approach:
  1. CANDIDATES = action JSON entries (141)
  2. For each candidate, count evidence votes from independent signals:
       v_audio   = audio onset within +-N frames with strength > S
       v_ce      = contact_event within +-N frames with prob > P
       v_dist2d  = 2D wrist-to-receiver pixel distance < D
       v_decel   = striker wrist decel > T
       v_headacc = receiver head 2D accel > T
       v_conf    = action confidence > C
  3. Keep candidates with >= MIN_VOTES.
  4. NMS by score (votes + tiebreakers), cooldown CD.
  5. Evaluate vs 31 GT.
"""

import json, os, numpy as np, cv2, librosa
from itertools import product

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
AUDIO_WAV  = os.path.join(FOLDER, "3.wav")
VIDEO_IN   = os.path.join(FOLDER, "3.mp4")
OUT_JSON   = os.path.join(FOLDER, "3_fusion_v2.json")

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

HEAD_KPS = [0,1,2,3,4]
WRIST_KPS = [9,10]
HAND_KPS  = [9,10,7,8]


def load_all():
    with open(SAM3D_JSON) as f: sam3d = json.load(f)
    with open(ACT_JSON) as f:   actions = json.load(f)["actions"]
    ce = sam3d.get("contact_events", [])
    fp = {}
    for tid in ("0","1"):
        if tid not in sam3d: continue
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e
    return sam3d, actions, ce, fp


def bbox_center(entry):
    x1,y1,x2,y2 = entry["bbox"]
    return ((x1+x2)/2.0, (y1+y2)/2.0)


def bbox_head_center(entry):
    """Top-center of bbox = approximate head location (2D pixel)."""
    x1,y1,x2,y2 = entry["bbox"]
    return ((x1+x2)/2.0, y1 + 0.18*(y2-y1))


def project_kp_2d(entry, k):
    """Project 3D keypoint to 2D pixel using focal_length + 1920x1080 center."""
    wc = entry.get("world_coords")
    if wc is None or k >= len(wc): return None
    X,Y,Z = wc[k]
    fl = entry.get("focal_length", 1500.0)
    if Z <= 0.1: return None
    u = fl*X/Z + W/2.0
    v = fl*Y/Z + H/2.0
    if not (0 <= u < W and 0 <= v < H): return None
    return np.array([u, v])


def wrist_pixel_min_dist_to_receiver(striker_entry, receiver_entry):
    """Min pixel distance from any striker hand keypoint to receiver bbox center / head."""
    if striker_entry is None or receiver_entry is None: return None
    rh = np.array(bbox_head_center(receiver_entry))
    rc = np.array(bbox_center(receiver_entry))
    best = float("inf")
    for k in HAND_KPS:
        # try 2D projection of 3D kp first
        p = project_kp_2d(striker_entry, k)
        if p is None: continue
        d_head = float(np.linalg.norm(p - rh))
        d_ctr  = float(np.linalg.norm(p - rc))
        best = min(best, d_head, d_ctr)
    return best if best < float("inf") else None


def head_pixel_accel(frame_persons, fn, rid, win=4):
    """2D pixel acceleration of receiver head center (from bbox)."""
    e_p = frame_persons.get(fn-win, {}).get(rid)
    e_n = frame_persons.get(fn,    {}).get(rid)
    e_x = frame_persons.get(fn+win,{}).get(rid)
    if not e_p or not e_n or not e_x: return 0.0
    h_p = np.array(bbox_head_center(e_p))
    h_n = np.array(bbox_head_center(e_n))
    h_x = np.array(bbox_head_center(e_x))
    v1 = (h_n - h_p) / win
    v2 = (h_x - h_n) / win
    accel = float(np.linalg.norm(v2 - v1))  # px/frame^2
    return accel


def wrist_pixel_decel(frame_persons, fn, sid, win=3):
    """Striker wrist 2D pixel deceleration."""
    e_p = frame_persons.get(fn-win, {}).get(sid)
    e_n = frame_persons.get(fn,    {}).get(sid)
    e_x = frame_persons.get(fn+win,{}).get(sid)
    if not e_p or not e_n or not e_x: return 0.0
    best = 0.0
    for k in WRIST_KPS:
        p1 = project_kp_2d(e_p, k); p2 = project_kp_2d(e_n, k); p3 = project_kp_2d(e_x, k)
        if p1 is None or p2 is None or p3 is None: continue
        v1 = float(np.linalg.norm(p2 - p1)) / win
        v2 = float(np.linalg.norm(p3 - p2)) / win
        decel = max(0.0, v1 - v2)
        best = max(best, decel)
    return best  # px/frame


def extract_audio_onsets():
    print("Loading audio...")
    y, sr = librosa.load(AUDIO_WAV, sr=22050, mono=True)
    print(f"  audio: {len(y)/sr:.1f}s @ {sr}Hz")
    y = np.append(y[0], y[1:] - 0.97*y[:-1])
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= 1000) & (freqs <= 6000)
    flux = np.maximum(0, np.diff(S[band], axis=1)).sum(axis=0).astype(np.float64)
    flux = np.concatenate([[0.0], flux])
    flux -= np.median(flux); flux = np.maximum(0, flux)
    times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=256)
    onsets = librosa.util.peak_pick(
        flux, pre_max=20, post_max=20, pre_avg=30, post_avg=30,
        delta=float(flux.std()*1.0), wait=12)
    on_t = times[onsets]
    on_f = (on_t*FPS).astype(int)
    on_s = flux[onsets] / (flux.max()+1e-9)
    keep = on_s > 0.10
    print(f"  onsets: {keep.sum()}")
    return list(zip(on_f[keep].tolist(), on_s[keep].tolist()))


def evaluate(det, gt=GT_FRAMES, tol=30):
    mg, md, pairs = set(), set(), []
    for di, df in enumerate(det):
        best, bd = None, tol+1
        for gi, gf in enumerate(gt):
            if gi in mg: continue
            d = abs(df-gf)
            if d < bd: bd, best = d, gi
        if best is not None:
            mg.add(best); md.add(di); pairs.append((di, best, bd))
    tp = len(mg); fp = len(det)-len(md); fn = len(gt)-tp
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, mg=mg, pairs=pairs)


def build_candidates(actions, ce, fp, onsets,
                     audio_win, audio_min,
                     ce_win, ce_prob_min,
                     dist_max_px, decel_min, headacc_min, conf_min,
                     min_votes):
    onsets_by_f = {}
    for f, s in onsets:
        onsets_by_f.setdefault(f, 0.0)
        onsets_by_f[f] = max(onsets_by_f[f], s)

    cands = []
    for a in actions:
        fn  = a["frame"]
        sid = int(a["fighter_type"].split("_")[1])
        rid = 1 - sid

        # -- Evidence flags --
        v_conf = 1 if a["confidence"] >= conf_min else 0

        # audio
        v_audio = 0; s_audio = 0.0
        for df in range(-audio_win, audio_win+1):
            if (fn+df) in onsets_by_f:
                s_audio = max(s_audio, onsets_by_f[fn+df])
        if s_audio >= audio_min: v_audio = 1

        # contact_event
        v_ce = 0; s_ce = 0.0; ce_d = None; ce_r = None
        for c in ce:
            if abs(c["frame"]-fn) <= ce_win and c["striker_id"] == sid:
                if c["contact_prob"] >= ce_prob_min:
                    s_ce = max(s_ce, c["contact_prob"])
                    ce_d, ce_r = c["contact_3d_distance_m"], c["contact_region"]
        if s_ce >= ce_prob_min: v_ce = 1

        # 2D pixel wrist-to-receiver distance
        v_dist = 0; s_dist = None
        for df in range(-3, 4):
            se = fp.get(fn+df, {}).get(sid)
            re = fp.get(fn+df, {}).get(rid)
            d = wrist_pixel_min_dist_to_receiver(se, re)
            if d is not None and (s_dist is None or d < s_dist):
                s_dist = d
        if s_dist is not None and s_dist <= dist_max_px: v_dist = 1

        # wrist decel (px)
        s_decel = wrist_pixel_decel(fp, fn, sid)
        v_decel = 1 if s_decel >= decel_min else 0

        # receiver head 2D accel (px)
        s_hacc = head_pixel_accel(fp, fn, rid)
        v_hacc = 1 if s_hacc >= headacc_min else 0

        votes = v_conf + v_audio + v_ce + v_dist + v_decel + v_hacc
        if votes < min_votes: continue

        # Score: votes (primary) + confidence tiebreaker
        score = votes + 0.1*a["confidence"] + 0.02*s_audio + 0.01*s_ce
        cands.append(dict(
            frame=fn, score=score, votes=votes, action=a, sid=sid,
            v_audio=v_audio, v_ce=v_ce, v_dist=v_dist,
            v_decel=v_decel, v_hacc=v_hacc, v_conf=v_conf,
            s_audio=s_audio, s_ce=s_ce, s_dist=s_dist or -1,
            s_decel=s_decel, s_hacc=s_hacc, ce_dist=ce_d, ce_region=ce_r,
        ))
    return cands


def nms(cands, cd):
    kept = []
    for c in sorted(cands, key=lambda x: -x["score"]):
        if any(abs(c["frame"]-k["frame"]) < cd for k in kept): continue
        kept.append(c)
    return sorted(kept, key=lambda x: x["frame"])


def main():
    sam3d, actions, ce, fp = load_all()
    print(f"{len(actions)} actions, {len(ce)} contact_events, {len(fp)} frames")
    onsets = extract_audio_onsets()

    print("\nGrid search v2 (voting-based)...")
    best = None
    n = 0
    for audio_win in [5, 8, 12]:
        for audio_min in [0.10, 0.20, 0.30]:
            for ce_win in [5, 8, 12]:
                for ce_prob_min in [0.20, 0.30, 0.50]:
                    for dist_max in [80, 120, 180, 250]:
                        for decel_min in [3.0, 6.0, 10.0]:
                            for headacc_min in [1.5, 3.0, 5.0]:
                                for conf_min in [0.3, 0.4, 0.5, 0.6]:
                                    for min_votes in [2, 3, 4]:
                                        for cd in [6, 10, 15]:
                                            cands = build_candidates(
                                                actions, ce, fp, onsets,
                                                audio_win, audio_min,
                                                ce_win, ce_prob_min,
                                                dist_max, decel_min, headacc_min,
                                                conf_min, min_votes)
                                            kept = nms(cands, cd)
                                            res = evaluate([k["frame"] for k in kept])
                                            n += 1
                                            if best is None or res["f1"] > best["f1"]:
                                                best = dict(
                                                    res,
                                                    audio_win=audio_win, audio_min=audio_min,
                                                    ce_win=ce_win, ce_prob_min=ce_prob_min,
                                                    dist_max=dist_max, decel_min=decel_min,
                                                    headacc_min=headacc_min, conf_min=conf_min,
                                                    min_votes=min_votes, cd=cd, kept=kept)
    print(f"  evaluated {n} configs")
    print(f"\nBest: F1={best['f1']:.3f}  P={best['p']:.3f}  R={best['r']:.3f}  "
          f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    print(f"  audio_win={best['audio_win']} audio_min={best['audio_min']}")
    print(f"  ce_win={best['ce_win']} ce_prob_min={best['ce_prob_min']}")
    print(f"  dist_max={best['dist_max']}px decel_min={best['decel_min']} headacc_min={best['headacc_min']}")
    print(f"  conf_min={best['conf_min']} min_votes={best['min_votes']} cd={best['cd']}")
    print(f"  detections: {len(best['kept'])}")

    by_gt = {gi: (di, d) for (di, gi, d) in best["pairs"]}
    print("\nMatched / missed:")
    for gi, ts in enumerate(GT_TS):
        gf = GT_FRAMES[gi]
        if gi in by_gt:
            di, dd = by_gt[gi]
            k = best["kept"][di]
            print(f"  [HIT]  {ts:8s} f{gf:4d} <- f{k['frame']:4d} d={dd:2d}  votes={k['votes']} "
                  f"(conf={k['v_conf']} aud={k['v_audio']} ce={k['v_ce']} dist={k['v_dist']} "
                  f"dec={k['v_decel']} hacc={k['v_hacc']})  s_dist={k['s_dist']:.0f}")
        else:
            print(f"  [MISS] {ts:8s} f{gf:4d}")

    md = {di for (di,_,_) in best["pairs"]}
    print("\nFalse positives:")
    for i, k in enumerate(best["kept"]):
        if i in md: continue
        print(f"  [FP] f{k['frame']:4d} votes={k['votes']} act={k['action']['action']:14s} "
              f"(conf={k['v_conf']} aud={k['v_audio']} ce={k['v_ce']} dist={k['v_dist']} "
              f"dec={k['v_decel']} hacc={k['v_hacc']})  s_dist={k['s_dist']:.0f}")

    out = {"config": {k: best[k] for k in ["audio_win","audio_min","ce_win","ce_prob_min",
                                            "dist_max","decel_min","headacc_min","conf_min",
                                            "min_votes","cd","f1","p","r","tp","fp","fn"]},
           "detections": [{"frame": k["frame"], "votes": k["votes"],
                           "striker_id": k["sid"],
                           "action": k["action"]["action"],
                           "ce_region": k["ce_region"]}
                          for k in best["kept"]]}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
