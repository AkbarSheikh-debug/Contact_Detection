#!/usr/bin/env python3
"""
SAM + Depth + Head-Reaction Punch Detector
==========================================
Attacks the proven root cause of the ~47% ceiling: 2D mask overlap can't tell
"fist touching body" from "fist in front of body".  We add a monocular DEPTH
check (Depth Anything V2) and a receiver HEAD-REACTION check, and we evaluate
against the USER's 27 video-verified landings (not the questionable 31 GT).

Per ASFormer action window:
  contact frame = in-window frame with min striker-wrist→receiver-body 2D gap
  Gate SAM    : striker wrist inside receiver body silhouette (SAM ViT-B)
  Gate DEPTH  : wrist depth ≈ receiver body depth at contact
                (a punch that falls short is CLOSER than the body → mismatch)
  Gate REACT  : receiver head moves in the frames just after contact

The script first prints each gate's separation on the verified labels (does it
actually help?), then evaluates a combined detector.

Usage:
    python sam_depth_detect.py                 # full run + video
    python sam_depth_detect.py --no-video
    python sam_depth_detect.py --combine and   # AND-gate instead of weighted
"""
import os
import re
import glob
import json
import argparse
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

DATA = r"/home/jake/Downloads/sam3d_with_world_coords"
KP2D = r"/home/jake/Downloads/for_impact_detection_experiment_2/2d_points.json"
VIDEO = os.path.join(DATA, "3.mp4")
ACT_JSON = os.path.join(DATA, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
CKPT = r"/home/jake/Desktop/HITAI/Contact_Detection/checkpoints/sam_vit_b_01ec64.pth"
VERIFIED_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs/sound_samples_av/punch"
OUT_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs"
OUT_MP4 = os.path.join(OUT_DIR, "sam_depth_detect.mp4")

FPS = 24.995
WRIST_KPS = [9, 10]
ELBOW_KPS = [7, 8]
HEAD_KPS = [0, 1, 2, 3, 4]
TORSO_KPS = [5, 6, 11, 12]
BODY_KPS = HEAD_KPS + TORSO_KPS
FALLOFF_PX = 40.0

# 31 original GT (for comparison only)
GT31 = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
        "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
        "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
        "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]


def ts_to_frame(ts):
    p = ts.split(":")
    s = (int(p[0]) + int(p[1]) / FPS) if len(p) == 2 \
        else (int(p[0]) * 60 + int(p[1]) + int(p[2]) / FPS)
    return int(round(s * FPS))


GT31_FRAMES = [ts_to_frame(t) for t in GT31]


def verified_frames():
    """The user's video-verified landings (frame numbers from punch/ filenames)."""
    fs = []
    for f in glob.glob(os.path.join(VERIFIED_DIR, "*.mp4")):
        m = re.search(r"_f(\d+)\.", os.path.basename(f))
        if m:
            fs.append(int(m.group(1)))
    return sorted(fs)


def load_2d(path):
    raw = json.load(open(path))
    joints, bbox = {}, {}
    for pid_s, entries in raw.items():
        pid = int(pid_s); joints[pid], bbox[pid] = {}, {}
        for e in entries:
            d = e.get("frame_dims", {})
            sx = d.get("original_width", 1920) / d.get("resized_width", 640)
            sy = d.get("original_height", 1080) / d.get("resized_height", 360)
            j = np.asarray(e["joints_2d"], float); j[:, 0] *= sx; j[:, 1] *= sy
            joints[pid][e["frame"]] = j
            bbox[pid][e["frame"]] = np.asarray(e["bbox"], float)
    return joints, bbox


def min_gap_frame(joints, sid, ws, we):
    rid = 1 - sid
    best = (None, 1e18, None)
    for f in range(ws, we + 1):
        sj = joints.get(sid, {}).get(f); rj = joints.get(rid, {}).get(f)
        if sj is None or rj is None:
            continue
        body = [rj[k] for k in BODY_KPS if not np.allclose(rj[k], 0)]
        if not body:
            continue
        for wk in WRIST_KPS:
            w = sj[wk]
            if np.allclose(w, 0):
                continue
            g = min(np.linalg.norm(w - b) for b in body)
            if g < best[1]:
                best = (f, g, wk)
    return best


def head_reaction(joints, rid, f, half=6):
    """Max receiver head-centroid displacement in frames after f (norm by bbox)."""
    def hc(ff):
        j = joints.get(rid, {}).get(ff)
        if j is None:
            return None
        pts = [j[k] for k in HEAD_KPS if not np.allclose(j[k], 0)]
        return np.mean(pts, axis=0) if pts else None
    base = hc(f)
    if base is None:
        return 0.0
    j0 = joints.get(rid, {}).get(f)
    diag = np.hypot(np.ptp(j0[:, 0]), np.ptp(j0[:, 1])) + 1e-6
    mx = 0.0
    for ff in range(f + 1, f + half + 1):
        h = hc(ff)
        if h is not None:
            mx = max(mx, np.linalg.norm(h - base) / diag)
    return float(min(1.0, mx / 0.30))


class SAM:
    def __init__(self, ckpt):
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sam] vit_b on {dev}")
        sam = sam_model_registry["vit_b"](checkpoint=ckpt); sam.to(dev)
        self.pred = SamPredictor(sam)

    def run(self, frame_bgr, rec_bbox, probe_pts):
        self.pred.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x1, y1, x2, y2 = [float(v) for v in rec_bbox]
        masks, _, _ = self.pred.predict(
            box=np.array([[x1, y1, x2, y2]], np.float32), multimask_output=False)
        mask = masks[0]
        H, W = mask.shape
        dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        best_s, best_pt = 0.0, None
        for p in probe_pts:
            if p is None or np.allclose(p, 0):
                continue
            px = int(np.clip(p[0], 0, W - 1)); py = int(np.clip(p[1], 0, H - 1))
            s = 1.0 if mask[py, px] else max(0.0, 1.0 - float(dt[py, px]) / FALLOFF_PX)
            if s > best_s:
                best_s, best_pt = s, [px, py]
        return best_s, best_pt, mask


class Depth:
    def __init__(self):
        from transformers import pipeline
        print("[depth] Depth-Anything-V2-Small on cpu")
        self.pipe = pipeline("depth-estimation",
                             model="depth-anything/Depth-Anything-V2-Small-hf",
                             device=-1)

    def map(self, frame_bgr):
        from PIL import Image
        out = self.pipe(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        d = out["predicted_depth"]
        d = d.squeeze().cpu().numpy().astype(np.float32)
        if d.shape != frame_bgr.shape[:2]:
            d = cv2.resize(d, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return d   # larger = closer (inverse depth)

    @staticmethod
    def patch(D, x, y, r=3):
        H, W = D.shape
        x = int(np.clip(x, 0, W - 1)); y = int(np.clip(y, 0, H - 1))
        return float(np.median(D[max(0, y-r):y+r+1, max(0, x-r):x+r+1]))


def depth_gate(D, wrist_xy, rj):
    """1.0 if wrist depth ≈ receiver body depth; →0 if wrist is in front."""
    body_pts = [rj[k] for k in BODY_KPS if not np.allclose(rj[k], 0)]
    if not body_pts:
        return 0.0, 0.0
    wd = Depth.patch(D, wrist_xy[0], wrist_xy[1])
    bvals = [Depth.patch(D, p[0], p[1]) for p in body_pts]
    bd = float(np.median(bvals))
    scale = float(np.std(bvals)) + 1e-3
    # positive delta = wrist closer than body (fist in front → a short punch)
    delta = (wd - bd) / scale
    score = float(np.exp(-(delta ** 2) / 2.0))   # 1 at delta=0, falls off
    return score, delta


def evaluate(det_frames, gt_frames, tol):
    mg, md = set(), set()
    for di, df in enumerate(det_frames):
        best, bd = None, tol + 1
        for gi, gf in enumerate(gt_frames):
            if gi in mg:
                continue
            if abs(df - gf) < bd:
                bd, best = abs(df - gf), gi
        if best is not None:
            mg.add(best); md.add(di)
    tp = len(mg); fp = len(det_frames) - len(md); fn = len(gt_frames) - tp
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return tp, fp, fn, p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=15)
    ap.add_argument("--cooldown", type=int, default=12)
    ap.add_argument("--combine", choices=["weighted", "and"], default="weighted")
    ap.add_argument("--sam-thr", type=float, default=0.5)
    ap.add_argument("--depth-thr", type=float, default=0.5)
    ap.add_argument("--react-thr", type=float, default=0.15)
    ap.add_argument("--score-thr", type=float, default=0.55)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=OUT_MP4)
    args = ap.parse_args()

    VF = verified_frames()
    print(f"[eval] {len(VF)} user-verified landings; {len(GT31_FRAMES)} legacy GT")
    joints, bbox = load_2d(KP2D)
    actions = json.load(open(ACT_JSON))["actions"]
    sam = SAM(CKPT); depth = Depth()
    cap = cv2.VideoCapture(VIDEO)

    rows = []
    print(f"[run] scoring {len(actions)} action windows (SAM + depth + react)…")
    for i, a in enumerate(actions):
        sid = 0 if a["fighter_type"] == "fighter_0" else 1
        f, gap, wk = min_gap_frame(joints, sid, a["window_start"]-3, a["window_end"]+5)
        if f is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, frame = cap.read()
        if not ok:
            continue
        rid = 1 - sid
        rj = joints.get(rid, {}).get(f); sj = joints.get(sid, {}).get(f)
        rb = bbox.get(rid, {}).get(f)
        if rj is None or sj is None or rb is None:
            continue
        probes = [sj[wk]] + [sj[k] for k in ELBOW_KPS]
        sam_s, pt, _ = sam.run(frame, rb, probes)
        D = depth.map(frame)
        dep_s, dep_delta = depth_gate(D, sj[wk], rj)
        rea_s = head_reaction(joints, rid, f)
        rows.append(dict(frame=f, sid=sid, rid=rid, action=a["action"], pt=pt,
                         sam=sam_s, depth=dep_s, react=rea_s, ddelta=dep_delta))
        if (i+1) % 30 == 0:
            print(f"    {i+1}/{len(actions)} …")
    cap.release()

    # ── diagnostic: does each gate separate verified landings? ───────────────
    def is_landed(fr, tol=15):
        return any(abs(fr - v) <= tol for v in VF)
    pos = [r for r in rows if is_landed(r["frame"])]
    neg = [r for r in rows if not is_landed(r["frame"])]
    print(f"\n[diag] candidates near a verified landing: {len(pos)}  "
          f"others: {len(neg)}")
    print(f"  {'gate':>8}  {'landed':>7}  {'other':>7}  {'separation':>10}")
    for g in ["sam", "depth", "react"]:
        pm = np.mean([r[g] for r in pos]) if pos else 0
        nm = np.mean([r[g] for r in neg]) if neg else 0
        print(f"  {g:>8}  {pm:7.3f}  {nm:7.3f}  {pm-nm:+10.3f}")

    # ── combined detector ─────────────────────────────────────────────────────
    for r in rows:
        if args.combine == "and":
            r["score"] = 1.0 if (r["sam"] >= args.sam_thr and
                                 r["depth"] >= args.depth_thr and
                                 r["react"] >= args.react_thr) else 0.0
        else:
            r["score"] = 0.45*r["sam"] + 0.35*r["depth"] + 0.20*r["react"]
    acc = (lambda r: r["score"] >= 1.0) if args.combine == "and" \
        else (lambda r: r["score"] >= args.score_thr)
    dets = [r for r in rows if acc(r)]
    dets.sort(key=lambda r: -r["score"])
    kept = []
    for r in dets:
        if all(abs(r["frame"] - k["frame"]) >= args.cooldown for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: r["frame"])
    det_f = [r["frame"] for r in kept]

    print(f"\n{'='*60}")
    print(f"  Combined ({args.combine})  kept={len(kept)}")
    for name, gts in [("USER-verified (27)", VF), ("legacy GT (31)", GT31_FRAMES)]:
        tp, fp, fn, p, r_, f1 = evaluate(det_f, gts, args.tol)
        print(f"  vs {name:<20} TP={tp:2d} FP={fp:2d} FN={fn:2d}  "
              f"P={p:.1%} R={r_:.1%} F1={f1:.1%}")
    print(f"{'='*60}")

    out_json = args.out.replace(".mp4", ".json")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"method": "sam+depth+react", "combine": args.combine,
               "n": len(kept),
               "events": [{"frame": r["frame"], "time_sec": round(r["frame"]/FPS, 3),
                           "sam": round(r["sam"], 3), "depth": round(r["depth"], 3),
                           "react": round(r["react"], 3), "score": round(r["score"], 3),
                           "contact_point": r["pt"], "action": r["action"],
                           "striker": r["sid"]} for r in kept]},
              open(out_json, "w"), indent=2)
    print(f"[out] {out_json}")

    if not args.no_video:
        render(kept, VF, args.tol, args.out, joints)


def render(kept, VF, tol, out_path, joints):
    SKEL = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    COL = {0:(0,255,180), 1:(255,140,0)}
    imp = {r["frame"]: r for r in kept}
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out_path.replace(".mp4", "_noaudio.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    flash = None; n = 0
    print(f"[render] {total} frames…")
    for fi in range(total):
        ok, fr = cap.read()
        if not ok:
            break
        for pid in (0, 1):
            j = joints.get(pid, {}).get(fi)
            if j is None:
                continue
            for a, b in SKEL:
                if not np.allclose(j[a], 0) and not np.allclose(j[b], 0):
                    cv2.line(fr, tuple(j[a].astype(int)), tuple(j[b].astype(int)), COL[pid], 2, cv2.LINE_AA)
            for wk in WRIST_KPS:
                if not np.allclose(j[wk], 0):
                    cv2.circle(fr, tuple(j[wk].astype(int)), 6, (0,180,255), -1)
        if fi in imp:
            r = imp[fi]; n += 1
            is_tp = any(abs(fi - v) <= tol for v in VF)
            flash = (fi, is_tp); col = (0,230,0) if is_tp else (0,50,255)
            if r["pt"]:
                cv2.circle(fr, tuple(r["pt"]), 24, col, 3, cv2.LINE_AA)
                cv2.putText(fr, f"{'HIT' if is_tp else 'FP'} s{r['sam']:.1f}/d{r['depth']:.1f}/r{r['react']:.1f}",
                            (r["pt"][0]+14, r["pt"][1]-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        if flash and fi < flash[0]+12:
            al = max(0.0, 1.0-(fi-flash[0])/12)*0.30
            t = np.zeros_like(fr); t[:, :] = (0,200,0) if flash[1] else (0,0,200)
            cv2.addWeighted(t, al, fr, 1.0, 0, fr)
        cv2.rectangle(fr, (0,0), (470,30), (15,15,20), -1)
        cv2.putText(fr, f"SAM+Depth+React  impacts:{n}  f:{fi}  green=verified",
                    (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 1, cv2.LINE_AA)
        wr.write(fr)
    cap.release(); wr.release()
    import subprocess
    try:
        subprocess.run(["ffmpeg","-y","-i",tmp,"-i",VIDEO,"-map","0:v","-map","1:a?",
                        "-c:v","copy","-shortest",out_path,"-loglevel","error"], check=True)
        os.remove(tmp); print(f"[render] saved {out_path}")
    except Exception as e:
        print(f"[render] mux failed ({e}); silent at {tmp}")


if __name__ == "__main__":
    main()
