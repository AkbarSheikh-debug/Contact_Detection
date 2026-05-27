#!/usr/bin/env python3
"""
Depth + Segmentation Contact Filter  --  Approach H v4
=======================================================
Uses Depth Anything V2 (monocular depth estimation) to verify that the
striker's wrist is at the same 3D depth as the receiver's head at impact.

Idea:
  - Real punch landing: wrist depth ≈ receiver head depth (same 3D plane)
  - Near-miss in 3D: wrist at different depth (arm extended past head, or
    still approaching), causing a large normalized depth difference
  - Clinch: both at same depth (will NOT help with clinch FPs)

Depth model: Depth Anything V2 Small (HuggingFace)
  - Outputs inverse depth (disparity): larger value = closer to camera
  - Normalized per-frame before comparison

Full pipeline:
  raw(157) -> cd=50 -> IoU<=0.35 -> flow>=1.3 -> arm_ratio>=0.30 -> depth_delta<=thr

Usage:
    python depth_filter.py               # eval + threshold sweep
    python depth_filter.py --video       # also render output video
    python depth_filter.py --depth-thr 0.15 --video
"""

import os, sys, json, argparse
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, PROCESS_EVERY_N_FRAMES, KEYPOINTS_2D_PATH, \
                   KP70_LEFT_WRIST, KP70_RIGHT_WRIST

VIDEO_IN   = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)
JSON_ARM   = os.path.join(OUTPUT_DIR, "fullscan", "results_H_cd50_iou35_flow13_arm30.json")
OUTPUT_SUB = os.path.join(OUTPUT_DIR, "fullscan")
STRIDE     = PROCESS_EVERY_N_FRAMES

GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]

HEAD_KPS = [0, 1, 2, 3, 4]
PATCH_PX = 20   # pixel patch radius for depth sampling

COL_F0    = (0, 255, 180); COL_F1 = (255, 140, 0); COL_WRIST = (0, 180, 255)
COL_IMP   = (0, 50, 255);  COL_HUD = (15, 15, 20); FLASH_FR  = 15
SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def parse_ts(ts, fps):
    p = ts.split(":")
    if len(p) == 2: return int(p[0]) + int(p[1]) / fps
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / fps


def load_2d(path):
    with open(path) as f: raw = json.load(f)
    persons = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str); persons[pid] = {}
        for e in entries:
            fn   = e["frame"]
            dims = e.get("frame_dims", {})
            sx   = dims.get("original_width",  1920) / dims.get("resized_width",  640)
            sy   = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            j    = np.array(e["joints_2d"], dtype=np.float32)
            j[:, 0] *= sx; j[:, 1] *= sy
            b    = e["bbox"]
            persons[pid][fn] = {
                "joints": j,
                "bbox":   [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
            }
    return persons


def load_depth_model():
    print("  Loading Depth Anything V2 Small ...")
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    print(f"  Depth model loaded on {'GPU' if device == 0 else 'CPU'}")
    return pipe


def get_depth_map(pipe, frame_bgr):
    """Run Depth Anything V2 on a BGR frame. Returns float32 depth map, same H x W."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    result = pipe(pil)
    depth = np.array(result["depth"], dtype=np.float32)
    # Normalize to [0, 1] per frame so values are comparable across scenes
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def sample_patch(depth_map, x, y, patch=PATCH_PX):
    """Mean depth in a square patch around (x, y)."""
    H, W = depth_map.shape
    x1, y1 = max(0, x - patch), max(0, y - patch)
    x2, y2 = min(W, x + patch), min(H, y + patch)
    region = depth_map[y1:y2, x1:x2]
    return float(region.mean()) if region.size > 0 else None


def depth_delta_for_event(ev, f2d, depth_map):
    """
    Compute normalized depth difference between striker's wrist and
    receiver's head at the impact frame.

    Returns: (delta, wrist_d, head_d) or (None, None, None) if no data.
    """
    fn  = ev["impact_frame"]
    sid = ev["striker_id"]; rid = ev["receiver_id"]
    fd_s = f2d.get(sid, {}).get(fn)
    fd_r = f2d.get(rid, {}).get(fn)
    if fd_s is None or fd_r is None:
        return None, None, None

    js = fd_s["joints"]; jr = fd_r["joints"]

    # Best wrist: whichever is closer to receiver's head
    wrist_d = None
    for wk in [KP70_LEFT_WRIST, KP70_RIGHT_WRIST]:
        if wk >= len(js) or np.allclose(js[wk], 0): continue
        x, y = int(js[wk][0]), int(js[wk][1])
        d = sample_patch(depth_map, x, y)
        if d is not None:
            if wrist_d is None or d is not None:
                wrist_d = d
                break  # take first valid wrist

    # Receiver head: mean of all valid head keypoints
    head_vals = []
    for k in HEAD_KPS:
        if k >= len(jr) or np.allclose(jr[k], 0): continue
        x, y = int(jr[k][0]), int(jr[k][1])
        d = sample_patch(depth_map, x, y)
        if d is not None:
            head_vals.append(d)
    head_d = float(np.mean(head_vals)) if head_vals else None

    if wrist_d is None or head_d is None:
        return None, wrist_d, head_d

    delta = abs(wrist_d - head_d)
    return delta, wrist_d, head_d


def evaluate(events, gt_frames, tol=30):
    det = [e["impact_frame"] for e in events]
    mgt = set(); mdet = set()
    for di, df in enumerate(det):
        for gi, gf in enumerate(gt_frames):
            if gi in mgt: continue
            if abs(df - gf) <= tol:
                mgt.add(gi); mdet.add(di); break
    tp = len(mgt); fp = len(det) - len(mdet); fn = 31 - tp
    p  = tp/(tp+fp) if (tp+fp) else 0.0
    r  = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return tp, fp, fn, p, r, f1, mgt, mdet


def seek_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def render_video(events, f2d, video_path, out_path, src_fps):
    impact_map = {e["impact_frame"]: e for e in events}
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = src_fps / STRIDE

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    flash_q = deque(); event_log = deque(maxlen=6); n_impacts = 0

    with tqdm(total=total // STRIDE, unit="fr", desc="  Rendering") as pbar:
        for fi in range(total):
            ret, frame = cap.read()
            if not ret: break
            if fi % STRIDE != 0: continue
            pbar.update(1)
            canvas = frame.copy()

            for pid, col in [(0, COL_F0), (1, COL_F1)]:
                fd = f2d.get(pid, {}).get(fi)
                if fd is None: continue
                j = fd["joints"]
                for a, b in SKELETON_PAIRS:
                    if (a < len(j) and b < len(j)
                            and not np.allclose(j[a], 0)
                            and not np.allclose(j[b], 0)):
                        cv2.line(canvas, tuple(j[a].astype(int)), tuple(j[b].astype(int)),
                                 col, 1, cv2.LINE_AA)
                for wi in [KP70_LEFT_WRIST, KP70_RIGHT_WRIST]:
                    if wi < len(j) and not np.allclose(j[wi], 0):
                        cv2.circle(canvas, tuple(j[wi].astype(int)), 5, COL_WRIST, -1, cv2.LINE_AA)

            if fi in impact_map:
                e = impact_map[fi]
                flash_q.clear(); flash_q.append(fi); n_impacts += 1
                flow_r = e.get("flow_ratio", 0.0)
                arm_r  = e.get("arm_ratio", 0.0)
                dep_d  = e.get("depth_delta", 0.0)
                event_log.appendleft(
                    f"F{e['striker_id']}->F{e['receiver_id']} "
                    f"{e['contact_region']} s={e['impact_score']:.2f} "
                    f"fl={flow_r:.1f} ar={arm_r:.1f} dd={dep_d:.2f}"
                )
                cp = e.get("contact_point", [])
                if len(cp) >= 2:
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])), 18, COL_IMP, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])),  8, (0,255,255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"d={dep_d:.2f}",
                                (int(cp[0])+12, int(cp[1])-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)

            if flash_q and fi < flash_q[-1] + FLASH_FR:
                alpha = max(0.0, 1.0 - (fi - flash_q[-1]) / FLASH_FR)
                red = np.zeros_like(canvas); red[:, :] = (0, 0, 200)
                cv2.addWeighted(red, alpha * 0.35, canvas, 1.0, 0, canvas)

            hud_h = 36 + 22 * len(event_log)
            cv2.rectangle(canvas, (0, 0), (560, hud_h), COL_HUD, -1)
            cv2.putText(canvas,
                        f"Approach H v4 (Depth+Seg) | Impacts: {n_impacts} | Frame: {fi}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,215,255), 1, cv2.LINE_AA)
            for k, ev_txt in enumerate(event_log):
                cv2.putText(canvas, ev_txt, (12, 42 + k*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1, cv2.LINE_AA)
            writer.write(canvas)

    cap.release(); writer.release()
    print(f"  Video saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-thr", type=float, default=None,
                    help="Max normalized depth delta to accept (default: auto sweep)")
    ap.add_argument("--video", action="store_true", help="Render output video")
    args = ap.parse_args()

    print()
    print("=" * 70)
    print("  Approach H v4  --  Depth Estimation + Segmentation Filter")
    print("=" * 70)
    print("  Model: Depth Anything V2 Small (monocular, normalized per-frame)")
    print("  Filter: keep events where |wrist_depth - head_depth| < threshold")
    print("  Near-miss: wrist at different depth than receiver head -> reject")
    print()

    with open(JSON_ARM) as f: data = json.load(f)
    src_fps   = data["src_fps"]
    events    = data["events"]
    gt_frames = [int(parse_ts(ts, src_fps) * src_fps) for ts in GT_TIMESTAMPS]

    print("  Loading 2D keypoints ...")
    f2d = load_2d(KEYPOINTS_2D_PATH)

    print(f"  Input events (after cd+iou+flow+arm): {len(events)}")
    tp0, fp0, fn0, p0, r0, f10, _, _ = evaluate(events, gt_frames)
    print(f"  Baseline: TP={tp0}  FP={fp0}  P={p0:.1%}  R={r0:.1%}  F1={f10:.1%}")
    print()

    # Load depth model
    depth_pipe = load_depth_model()
    print()

    # Open video and run depth inference on each event frame
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {VIDEO_IN}")

    print(f"  Running Depth Anything V2 on {len(events)} event frames ...")
    for ev in tqdm(events, unit="ev", desc="  Depth inference"):
        fi = ev["impact_frame"]
        frame = seek_frame(cap, fi)
        if frame is None:
            ev["depth_delta"] = None
            ev["wrist_depth"] = None
            ev["head_depth"]  = None
            continue
        depth_map = get_depth_map(depth_pipe, frame)
        delta, wd, hd = depth_delta_for_event(ev, f2d, depth_map)
        ev["depth_delta"] = float(delta) if delta is not None else None
        ev["wrist_depth"] = float(wd)    if wd    is not None else None
        ev["head_depth"]  = float(hd)    if hd    is not None else None

    cap.release()
    print()

    # Separate TP / FP deltas for analysis
    _, _, _, _, _, _, _, mdet0 = evaluate(events, gt_frames)
    valid_evs = [ev for ev in events if ev["depth_delta"] is not None]
    tp_deltas = [ev["depth_delta"] for di, ev in enumerate(events)
                 if di in mdet0 and ev["depth_delta"] is not None]
    fp_deltas = [ev["depth_delta"] for di, ev in enumerate(events)
                 if di not in mdet0 and ev["depth_delta"] is not None]

    if tp_deltas:
        print(f"  TP depth_delta: min={min(tp_deltas):.3f}  "
              f"mean={np.mean(tp_deltas):.3f}  max={max(tp_deltas):.3f}")
    if fp_deltas:
        print(f"  FP depth_delta: min={min(fp_deltas):.3f}  "
              f"mean={np.mean(fp_deltas):.3f}  max={max(fp_deltas):.3f}")
    print()

    # Threshold sweep (lower delta = same depth = keep)
    print("  Depth delta threshold sweep (depth_delta = |wrist_depth - head_depth|):")
    print("  Keep events where delta < threshold  (small delta = wrist at same depth as head)")
    best_f1 = 0.0; best_thr = None; best_events = events
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
    for thr in thresholds:
        # Keep: delta < thr, OR no depth data (neutral -> keep)
        filtered = [ev for ev in events
                    if ev["depth_delta"] is None or ev["depth_delta"] < thr]
        if not filtered: continue
        tp, fp, fn, p, r, f1, mg, _ = evaluate(filtered, gt_frames)
        mark = " <--" if f1 > best_f1 else ""
        lbl = "all" if thr > 1.0 else f"{thr:.2f}"
        print(f"  delta<{lbl:5s}: {len(filtered):3d} events  "
              f"TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}{mark}")
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr; best_events = filtered

    if args.depth_thr is not None:
        thr = args.depth_thr
        best_events = [ev for ev in events
                       if ev["depth_delta"] is None or ev["depth_delta"] < thr]
        best_thr = thr
        tp, fp, fn, p, r, f1, mg, md = evaluate(best_events, gt_frames)
        print(f"\n  Manual threshold: delta<{thr:.2f} -> {len(best_events)} events  "
              f"TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}")
    else:
        tp, fp, fn, p, r, f1, mg, md = evaluate(best_events, gt_frames)

    print(f"\n  BEST: delta<{best_thr:.2f} -> {len(best_events)} events  "
          f"TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}")
    missed = [GT_TIMESTAMPS[gi] for gi in range(31) if gi not in mg]
    if missed: print(f"  Missed: {missed}")

    print()
    print("  === FULL PIPELINE SUMMARY ===")
    print(f"  raw(157) -> cd=50 -> IoU<=0.35 -> flow>=1.3 -> arm>=0.30 -> depth<{best_thr:.2f}")
    print(f"  Result: {len(best_events)} events | TP={tp} | FP={fp} | P={p:.1%} | R={r:.1%} | F1={f1:.1%}")
    print(f"  vs. previous best (v3 Pi-HOC arm filter): 45 events | F1=65.8%")
    print()
    print("  === DEPTH SIGNAL INTERPRETATION ===")
    print("  Depth Anything V2 outputs inverse depth (larger = closer to camera)")
    print("  delta = |wrist_depth - head_depth| normalized [0,1] per frame")
    print("  Small delta: wrist and head at same 3D depth -> likely contact")
    print("  Large delta: wrist at different depth -> likely near-miss or extended arm")

    # Save JSON
    thr_tag = int(best_thr * 100) if best_thr is not None else 100
    tag = f"cd50_iou35_flow13_arm30_dep{thr_tag:02d}"
    json_out = os.path.join(OUTPUT_SUB, f"results_H_{tag}.json")
    out_data = dict(data)
    out_data["events"]    = best_events
    out_data["n_impacts"] = len(best_events)
    out_data["filters"]   = {
        "cooldown": 50, "max_iou": 0.35,
        "min_flow_ratio": 1.3, "min_arm_ratio": 0.30,
        "max_depth_delta": best_thr,
        "depth_model": "Depth-Anything-V2-Small-hf",
        "depth_normalization": "per-frame [0,1]",
    }
    os.makedirs(OUTPUT_SUB, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  JSON saved: {json_out}")

    if args.video:
        vid_out = os.path.join(OUTPUT_SUB, f"1_H_{tag}_real.mp4")
        render_video(best_events, f2d, VIDEO_IN, vid_out, src_fps)

    print()


if __name__ == "__main__":
    main()
