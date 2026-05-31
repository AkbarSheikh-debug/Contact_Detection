#!/usr/bin/env python3
"""
SAM Mask-Overlap Punch Detector  (the strong signal, finally runnable)
======================================================================
This is the one method the project's own learned weights rated 3.25x above
every other gate — and it had never actually run here because no checkpoint
existed.  It answers the only question that really separates a landed punch
from a near-miss: *is the striker's fist pixel actually inside the opponent's
body silhouette?*

Pipeline (CPU-friendly: ONE SAM encode per candidate)
-----------------------------------------------------
  1. Candidates = ASFormer action windows (thrown punches).
  2. Real joints_2d (from for_impact_detection_experiment_2/2d_points.json)
     give accurate pixel wrists/elbows + the receiver bbox.
  3. For each action, find the in-window frame where the striker's wrist is
     closest (in pixels) to the receiver body — the likely contact frame.
  4. Run SAM ONCE on that frame: segment the receiver (bbox prompt) → body
     mask.  Probe the striker wrist AND elbow:
         inside mask           -> 1.0
         outside               -> falloff to 0 at FALLOFF_PX from the edge
     Take the max of wrist/elbow.
  5. score = SAM overlap.  Keep if >= --thr.  Cooldown NMS.
  6. Evaluate vs the 31 GT timestamps; render annotated video.

Honest limitation: a punch stopped by the GUARD still has the wrist inside the
receiver silhouette, so SAM alone cannot separate blocked-on-glove from
landed-on-head.  That needs contact-region data (Pi-HOC / VolumetricSMPL).
But SAM removes the huge class of misses where the fist never reaches the body.

Usage:
    python sam_detect.py
    python sam_detect.py --thr 0.5 --no-video
"""
import os
import json
import argparse
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = r"/home/jake/Downloads/sam3d_with_world_coords"
KP2D = r"/home/jake/Downloads/for_impact_detection_experiment_2/2d_points.json"
VIDEO = os.path.join(DATA, "3.mp4")
ACT_JSON = os.path.join(DATA, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
CKPT = r"/home/jake/Desktop/HITAI/Contact_Detection/checkpoints/sam_vit_b_01ec64.pth"
OUT_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs"
OUT_MP4 = os.path.join(OUT_DIR, "sam_detect.mp4")

FPS = 24.995
WRIST_KPS = [9, 10]
ELBOW_KPS = [7, 8]
BODY_KPS = [0, 1, 2, 3, 4, 5, 6, 11, 12]   # head + shoulders + hips
FALLOFF_PX = 40.0

GT_TS = ["7:11", "18:01", "24:04", "30:19", "31:07", "34:07", "37:08", "53:02",
         "55:17", "1:05:22", "1:06:09", "1:06:20", "1:20:14", "1:25:15", "1:26:05",
         "1:27:18", "1:42:16", "1:42:19", "1:48:22", "1:51:23", "1:53:24", "2:03:19",
         "2:15:22", "2:17:11", "2:25:17", "2:27:24", "2:28:24", "2:34:13", "2:46:12",
         "2:49:24", "2:52:16"]


def ts_to_frame(ts):
    p = ts.split(":")
    sec = (int(p[0]) + int(p[1]) / FPS) if len(p) == 2 \
        else (int(p[0]) * 60 + int(p[1]) + int(p[2]) / FPS)
    return int(round(sec * FPS))


GT_FRAMES = [ts_to_frame(t) for t in GT_TS]


def load_2d(path):
    """{pid: {frame: (70,2) px}}, scaled to original resolution, + bbox."""
    raw = json.load(open(path))
    joints, bbox = {}, {}
    for pid_s, entries in raw.items():
        pid = int(pid_s)
        joints[pid], bbox[pid] = {}, {}
        for e in entries:
            d = e.get("frame_dims", {})
            sx = d.get("original_width", 1920) / d.get("resized_width", 640)
            sy = d.get("original_height", 1080) / d.get("resized_height", 360)
            j = np.asarray(e["joints_2d"], float)
            j[:, 0] *= sx
            j[:, 1] *= sy
            joints[pid][e["frame"]] = j
            bbox[pid][e["frame"]] = np.asarray(e["bbox"], float)
    return joints, bbox


def min_gap_frame(joints, sid, ws, we):
    """In-window frame minimising striker-wrist→receiver-body pixel gap."""
    rid = 1 - sid
    best = (None, 1e18, None)
    for f in range(ws, we + 1):
        sj = joints.get(sid, {}).get(f)
        rj = joints.get(rid, {}).get(f)
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
    return best   # (frame, gap, wrist_idx)


class SAM:
    def __init__(self, ckpt):
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sam] loading vit_b on {dev} …")
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        sam.to(dev)
        self.pred = SamPredictor(sam)
        self.dev = dev

    def overlap(self, frame_bgr, rec_bbox, probe_pts):
        """Return (max overlap score, best probe point, mask)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.pred.set_image(rgb)
        x1, y1, x2, y2 = [float(v) for v in rec_bbox]
        masks, _, _ = self.pred.predict(
            box=np.array([[x1, y1, x2, y2]], np.float32), multimask_output=False)
        mask = masks[0]
        H, W = mask.shape
        inv = (~mask).astype(np.uint8)
        dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

        best_s, best_pt = 0.0, None
        for p in probe_pts:
            if p is None or np.allclose(p, 0):
                continue
            px = int(np.clip(p[0], 0, W - 1))
            py = int(np.clip(p[1], 0, H - 1))
            s = 1.0 if mask[py, px] else max(0.0, 1.0 - float(dt[py, px]) / FALLOFF_PX)
            if s > best_s:
                best_s, best_pt = s, [px, py]
        return best_s, best_pt, mask


def evaluate(det_frames, tol=12):
    mg, md = set(), set()
    for di, df in enumerate(det_frames):
        best, bd = None, tol + 1
        for gi, gf in enumerate(GT_FRAMES):
            if gi in mg:
                continue
            if abs(df - gf) < bd:
                bd, best = abs(df - gf), gi
        if best is not None:
            mg.add(best)
            md.add(di)
    tp = len(mg)
    fp = len(det_frames) - len(md)
    fn = len(GT_FRAMES) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5,
                    help="SAM overlap to accept as a landed punch")
    ap.add_argument("--cooldown", type=int, default=12)
    ap.add_argument("--tol", type=int, default=12)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=OUT_MP4)
    args = ap.parse_args()

    joints, bbox = load_2d(KP2D)
    actions = json.load(open(ACT_JSON))["actions"]
    print(f"[sam] {len(actions)} action windows; real joints_2d loaded")
    sam = SAM(CKPT)

    cap = cv2.VideoCapture(VIDEO)

    # 1) pick the contact frame per action (cheap 2D), collect unique frames
    cands = []
    for a in actions:
        sid = 0 if a["fighter_type"] == "fighter_0" else 1
        f, gap, wk = min_gap_frame(joints, sid, a["window_start"] - 3,
                                   a["window_end"] + 5)
        if f is None:
            continue
        cands.append(dict(frame=f, sid=sid, rid=1 - sid, wk=wk,
                          action=a["action"]))

    # 2) run SAM once per candidate frame
    print(f"[sam] scoring {len(cands)} candidate frames (one SAM encode each)…")
    dets = []
    for i, c in enumerate(cands):
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        rj_bbox = bbox.get(c["rid"], {}).get(c["frame"])
        sj = joints.get(c["sid"], {}).get(c["frame"])
        if rj_bbox is None or sj is None:
            continue
        probes = [sj[c["wk"]]] + [sj[k] for k in ELBOW_KPS]
        s, pt, _ = sam.overlap(frame, rj_bbox, probes)
        c["sam"] = s
        c["pt"] = pt
        if s >= args.thr:
            dets.append(c)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(cands)} …")

    # 3) cooldown NMS (keep higher SAM)
    dets.sort(key=lambda c: -c["sam"])
    kept = []
    for c in dets:
        if all(abs(c["frame"] - k["frame"]) >= args.cooldown for k in kept):
            kept.append(c)
    kept.sort(key=lambda c: c["frame"])

    tp, fp, fn, p, r, f1 = evaluate([c["frame"] for c in kept], args.tol)
    print(f"\n{'='*56}")
    print(f"  SAM mask-overlap  (thr={args.thr}, cd={args.cooldown}, "
          f"tol=±{args.tol}fr)")
    print(f"  kept={len(kept)}  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision={p:.1%}  Recall={r:.1%}  F1={f1:.1%}")
    print(f"{'='*56}\n")
    for c in kept:
        m, s = int(c["frame"] / FPS // 60), (c["frame"] / FPS) % 60
        hit = "HIT" if any(abs(c["frame"] - g) <= args.tol for g in GT_FRAMES) else "FP "
        print(f"  {hit}  {m}:{s:05.2f}  f{c['frame']:<5d} sam={c['sam']:.2f} "
              f"F{c['sid']}->F{c['rid']} {c['action']}")

    out_json = args.out.replace(".mp4", ".json")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"method": "sam_mask_overlap", "thr": args.thr,
               "metrics": dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1),
               "n": len(kept),
               "events": [{"frame": c["frame"], "time_sec": round(c["frame"]/FPS, 3),
                           "sam": round(c["sam"], 3), "striker": c["sid"],
                           "contact_point": c["pt"], "action": c["action"]}
                          for c in kept]}, open(out_json, "w"), indent=2)
    print(f"[sam] JSON saved: {out_json}")

    cap.release()
    if not args.no_video:
        render(kept, args.tol, args.out)


def render(kept, tol, out_path):
    SKEL = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    COL = {0: (0,255,180), 1: (255,140,0)}
    joints, _ = load_2d(KP2D)
    imp = {c["frame"]: c for c in kept}
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out_path.replace(".mp4", "_noaudio.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    flash = None; n = 0
    print(f"[sam] rendering {total} frames …")
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
                    cv2.line(fr, tuple(j[a].astype(int)), tuple(j[b].astype(int)),
                             COL[pid], 2, cv2.LINE_AA)
            for wk in WRIST_KPS:
                if not np.allclose(j[wk], 0):
                    cv2.circle(fr, tuple(j[wk].astype(int)), 6, (0,180,255), -1)
        if fi in imp:
            c = imp[fi]; n += 1
            is_tp = any(abs(fi - g) <= tol for g in GT_FRAMES)
            flash = (fi, is_tp)
            col = (0,230,0) if is_tp else (0,50,255)
            if c["pt"]:
                cv2.circle(fr, tuple(c["pt"]), 24, col, 3, cv2.LINE_AA)
                cv2.putText(fr, f"{'HIT' if is_tp else 'FP'} sam={c['sam']:.2f}",
                            (c["pt"][0]+14, c["pt"][1]-14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2, cv2.LINE_AA)
        if flash and fi < flash[0] + 12:
            a = max(0.0, 1.0-(fi-flash[0])/12)*0.30
            t = np.zeros_like(fr); t[:, :] = (0,200,0) if flash[1] else (0,0,200)
            cv2.addWeighted(t, a, fr, 1.0, 0, fr)
        cv2.rectangle(fr, (0,0), (440,30), (15,15,20), -1)
        cv2.putText(fr, f"SAM overlap  impacts:{n}  f:{fi}  green=HIT red=FP",
                    (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,215,255), 1, cv2.LINE_AA)
        wr.write(fr)
    cap.release(); wr.release()
    import subprocess
    try:
        subprocess.run(["ffmpeg","-y","-i",tmp,"-i",VIDEO,"-map","0:v","-map","1:a?",
                        "-c:v","copy","-shortest",out_path,"-loglevel","error"], check=True)
        os.remove(tmp); print(f"[sam] saved with audio: {out_path}")
    except Exception as e:
        print(f"[sam] mux failed ({e}); silent at {tmp}")


if __name__ == "__main__":
    main()
