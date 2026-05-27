#!/usr/bin/env python3
"""
Optical Flow Impact Gate  —  Approach H v2
==========================================
Research basis:
  - "Boxing Punch Detection with Single Static Camera" (MDPI Entropy 2024)
  - "Optical Flow Divergence for Collision Detection" (visual momentum transfer)
  - DECO / Pi-HOC: body deformation as contact signal

Core idea (zero-shot, no retraining):
  A landed punch causes the RECEIVER's head/body to suddenly accelerate.
  This is visible as a spike in dense optical flow magnitude in the receiver's
  head region at the impact frame vs. the baseline motion of that region.

  - Real punch:  flow_ratio = flow_at_impact / baseline_flow >> 1 (head snaps)
  - Clinch:      flow_ratio ≈ 1  (both bodies moving together, no sudden change)
  - Near-miss:   flow_ratio ≈ 1  (receiver's head doesn't react)

Usage:
    python optflow_gate.py                          # full pipeline
    python optflow_gate.py --no-video               # eval only, no video render
    python optflow_gate.py --min-ratio 1.5          # tune sensitivity
"""

import os, sys, json, argparse
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUTPUT_DIR, PROCESS_EVERY_N_FRAMES, KEYPOINTS_2D_PATH,
    KP70_LEFT_WRIST, KP70_RIGHT_WRIST,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_IN = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)
JSON_RAW    = os.path.join(OUTPUT_DIR, "fullscan", "results_H_fullscan.json")
OUTPUT_SUB  = os.path.join(OUTPUT_DIR, "fullscan")

STRIDE      = PROCESS_EVERY_N_FRAMES  # 2

# Cooldown applied before optical flow gate
GLOBAL_CD   = 50   # source frames (2s @ 25fps) — same as best result

# Optical flow gate parameters
FLOW_BASELINE_WINDOW = (-30, -10)   # frames [t-30, t-10] for baseline measurement
FLOW_IMPACT_WINDOW   = (-4, 0)     # frames [t-4, t] for impact measurement
HEAD_KPS = [0, 1, 2, 3, 4]         # nose + eyes + ears
TORSO_KPS = [5, 6, 11, 12]         # shoulders + hips (fallback if head not visible)
HEAD_PAD = 30                       # px padding around head bbox

# GT timestamps for evaluation
GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]

# Rendering colours
COL_F0 = (0, 255, 180); COL_F1 = (255, 140, 0); COL_WRIST = (0, 180, 255)
COL_IMP = (0, 50, 255); COL_HUD = (15, 15, 20); FLASH_FR = 15
SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
            fn  = e["frame"]
            dims = e.get("frame_dims", {})
            sx = dims.get("original_width",  1920) / dims.get("resized_width",  640)
            sy = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            j  = np.array(e["joints_2d"], dtype=np.float32)
            j[:, 0] *= sx; j[:, 1] *= sy
            b  = e["bbox"]
            persons[pid][fn] = {
                "joints": j,
                "bbox": [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
            }
    return persons


def bbox_iou(b0, b1):
    if b0 is None or b1 is None: return 0.0
    x1 = max(b0[0], b1[0]); y1 = max(b0[1], b1[1])
    x2 = min(b0[2], b1[2]); y2 = min(b0[3], b1[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a0 = (b0[2]-b0[0]) * (b0[3]-b0[1])
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    return inter / (a0 + a1 - inter + 1e-6)


def apply_cooldown(events, cd):
    kept = []; last = -9999
    for e in events:
        if e["impact_frame"] - last >= cd:
            kept.append(e); last = e["impact_frame"]
    return kept


def head_roi(f2d_pid, fn, H, W):
    """Returns (x1,y1,x2,y2) of receiver's head region, or None."""
    fd = f2d_pid.get(fn)
    if fd is None: return None
    j  = fd["joints"]
    pts = [j[k] for k in HEAD_KPS if k < len(j) and not np.allclose(j[k], 0)]
    if not pts:
        pts = [j[k] for k in TORSO_KPS if k < len(j) and not np.allclose(j[k], 0)]
    if not pts: return None
    pts = np.array(pts)
    x1 = max(0, int(pts[:, 0].min()) - HEAD_PAD)
    y1 = max(0, int(pts[:, 1].min()) - HEAD_PAD)
    x2 = min(W,  int(pts[:, 0].max()) + HEAD_PAD)
    y2 = min(H,  int(pts[:, 1].max()) + HEAD_PAD)
    if x2 <= x1 or y2 <= y1: return None
    return x1, y1, x2, y2


def flow_magnitude_in_roi(frame_a, frame_b, roi):
    """Dense Farneback optical flow magnitude mean within ROI."""
    if roi is None: return 0.0
    x1, y1, x2, y2 = roi
    ga = cv2.cvtColor(frame_a[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if ga.size == 0 or gb.size == 0: return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        ga, gb, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(mag.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Frame cache builder (reads video once, caches needed frames)
# ─────────────────────────────────────────────────────────────────────────────

def build_frame_cache(video_path, events, baseline_win, impact_win):
    """Read only the frames needed for optical flow computation."""
    needed = set()
    for e in events:
        fn = e["impact_frame"]
        for delta in range(baseline_win[0] - 4, impact_win[1] + 2, STRIDE):
            needed.add(fn + delta)
    needed = {f for f in needed if f >= 0}

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    needed = {f for f in needed if f < total}
    needed_sorted = sorted(needed)

    cache = {}
    ptr = 0
    with tqdm(total=len(needed_sorted), unit="fr", desc="  Caching frames") as pbar:
        for fn in needed_sorted:
            if fn < ptr:
                continue
            if fn > ptr:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ptr = fn
            ret, frame = cap.read()
            if ret:
                cache[fn] = frame
                pbar.update(1)
            ptr += 1
    cap.release()
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Optical flow gate
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_ratio(events, frame_cache, f2d, video_h, video_w, baseline_win, impact_win):
    """For each event, compute optical flow ratio at receiver's head.

    ratio = mean_flow_at_impact / mean_flow_at_baseline

    High ratio (>>1): receiver's head accelerated suddenly → real impact
    Ratio ≈ 1:        head motion same as baseline → clinch or near-miss
    """
    results = []
    for e in events:
        fn  = e["impact_frame"]
        rid = e["receiver_id"]
        roi = head_roi(f2d.get(rid, {}), fn, video_h, video_w)

        # Impact window: optical flow magnitude summed over frames [fn+impact_win[0], fn]
        impact_flows = []
        for d in range(impact_win[0], impact_win[1], STRIDE):
            fa = frame_cache.get(fn + d)
            fb = frame_cache.get(fn + d + STRIDE)
            if fa is not None and fb is not None:
                mag = flow_magnitude_in_roi(fa, fb, roi)
                impact_flows.append(mag)

        # Baseline window: optical flow magnitude [fn+baseline_win[0], fn+baseline_win[1]]
        baseline_flows = []
        for d in range(baseline_win[0], baseline_win[1], STRIDE):
            fa = frame_cache.get(fn + d)
            fb = frame_cache.get(fn + d + STRIDE)
            if fa is not None and fb is not None:
                mag = flow_magnitude_in_roi(fa, fb, roi)
                baseline_flows.append(mag)

        impact_mean   = float(np.mean(impact_flows))   if impact_flows   else 0.0
        baseline_mean = float(np.mean(baseline_flows)) if baseline_flows else 0.0
        ratio = impact_mean / (baseline_mean + 1e-4)

        results.append({
            "impact_mean": impact_mean,
            "baseline_mean": baseline_mean,
            "flow_ratio": ratio,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(events, gt_frames, tol=30):
    det = [e["impact_frame"] for e in events]
    matched_det = set(); matched_gt = set()
    for di, df in enumerate(det):
        for gi, gf in enumerate(gt_frames):
            if gi in matched_gt: continue
            if abs(df - gf) <= tol:
                matched_gt.add(gi); matched_det.add(di); break
    tp = len(matched_gt); fp = len(det) - len(matched_det); fn = 31 - tp
    p  = tp/(tp+fp) if (tp+fp) else 0.0
    r  = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return tp, fp, fn, p, r, f1, matched_gt, matched_det


# ─────────────────────────────────────────────────────────────────────────────
# Video renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_video(events, f2d, video_path, out_path, src_fps):
    impact_map = {e["impact_frame"]: e for e in events}
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                ratio = e.get("flow_ratio", 0.0)
                event_log.appendleft(
                    f"F{e['striker_id']}->F{e['receiver_id']} "
                    f"{e['contact_region']} {e['impact_score']:.2f} r={ratio:.2f}"
                )
                cp = e.get("contact_point", [])
                if len(cp) >= 2:
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])), 18, COL_IMP, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])),  8, (0,255,255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"{e['impact_score']:.2f} r{ratio:.1f}",
                                (int(cp[0])+12, int(cp[1])-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

            if flash_q and fi < flash_q[-1] + FLASH_FR:
                alpha = max(0.0, 1.0 - (fi - flash_q[-1]) / FLASH_FR)
                red = np.zeros_like(canvas); red[:, :] = (0, 0, 200)
                cv2.addWeighted(red, alpha * 0.35, canvas, 1.0, 0, canvas)

            hud_h = 36 + 22 * len(event_log)
            cv2.rectangle(canvas, (0, 0), (480, hud_h), COL_HUD, -1)
            cv2.putText(canvas,
                        f"Approach H v2 | Impacts: {n_impacts} | Frame: {fi}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,215,255), 1, cv2.LINE_AA)
            for k, ev_txt in enumerate(event_log):
                cv2.putText(canvas, ev_txt, (12, 42 + k*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200,200,200), 1, cv2.LINE_AA)
            writer.write(canvas)

    cap.release(); writer.release()
    print(f"  Video saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratio",  type=float, default=1.2,
                    help="Min optical flow ratio (impact/baseline) to accept as real impact")
    ap.add_argument("--cooldown",   type=int,   default=GLOBAL_CD)
    ap.add_argument("--max-iou",    type=float, default=0.35)
    ap.add_argument("--no-video",   action="store_true")
    args = ap.parse_args()

    print()
    print("=" * 70)
    print("  Approach H v2  —  Optical Flow Impact Gate")
    print("=" * 70)
    print(f"  Cooldown   : {args.cooldown}fr  Max-IoU: {args.max_iou}  Min-ratio: {args.min_ratio}")

    # Load data
    with open(JSON_RAW) as f: raw_data = json.load(f)
    src_fps = raw_data["src_fps"]
    gt_frames = [int(parse_ts(ts, src_fps) * src_fps) for ts in GT_TIMESTAMPS]

    print("  Loading 2D keypoints...")
    f2d = load_2d(KEYPOINTS_2D_PATH)

    raw_events = raw_data["events"]

    # Step 1: global cooldown
    after_cd = apply_cooldown(raw_events, args.cooldown)
    print(f"  After cooldown ({args.cooldown}fr): {len(after_cd)} events")

    # Step 2: IoU filter (clinch removal)
    def _iou(e):
        fn = e["impact_frame"]
        fd_s = f2d.get(e["striker_id"], {}).get(fn)
        fd_r = f2d.get(e["receiver_id"], {}).get(fn)
        if fd_s is None or fd_r is None: return 0.0
        return bbox_iou(fd_s["bbox"], fd_r["bbox"])

    after_iou = [e for e in after_cd if _iou(e) <= args.max_iou]
    print(f"  After IoU filter (<=0.35): {len(after_iou)} events")

    # Step 3: Optical flow gate
    cap = cv2.VideoCapture(VIDEO_IN)
    VH  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VW  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    print("  Building frame cache for optical flow...")
    frame_cache = build_frame_cache(VIDEO_IN, after_iou, FLOW_BASELINE_WINDOW, FLOW_IMPACT_WINDOW)

    print("  Computing optical flow ratios at receiver head region...")
    flow_results = compute_flow_ratio(
        after_iou, frame_cache, f2d, VH, VW,
        FLOW_BASELINE_WINDOW, FLOW_IMPACT_WINDOW,
    )

    # Attach flow results to events
    for e, fr in zip(after_iou, flow_results):
        e.update(fr)

    print()
    print("  Flow ratio distribution:")
    tp0, fp0, fn0, p0, r0, f1_0, mgt0, mdt0 = evaluate(after_iou, gt_frames)
    print(f"  Pre-flow baseline (cd+iou only): {len(after_iou)} events  "
          f"TP={tp0}  FP={fp0}  P={p0:.1%}  R={r0:.1%}  F1={f1_0:.1%}")

    tp_ratios = [after_iou[di]["flow_ratio"] for di in mdt0]
    fp_ratios = [after_iou[di]["flow_ratio"] for di in range(len(after_iou)) if di not in mdt0]
    if tp_ratios: print(f"  TP flow ratios: min={min(tp_ratios):.2f}  mean={np.mean(tp_ratios):.2f}  max={max(tp_ratios):.2f}")
    if fp_ratios: print(f"  FP flow ratios: min={min(fp_ratios):.2f}  mean={np.mean(fp_ratios):.2f}  max={max(fp_ratios):.2f}")

    print()
    print("  Flow ratio threshold sweep:")
    best_f1 = 0.0; best_thr = args.min_ratio; best_result = None
    for thr in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
        filtered = [e for e in after_iou if e["flow_ratio"] >= thr]
        if not filtered: continue
        tp, fp, fn, p, r, f1, mg, md = evaluate(filtered, gt_frames)
        mark = " <--" if f1 > best_f1 else ""
        print(f"  ratio>={thr:.1f}: {len(filtered):3d} events  TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}{mark}")
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr
            best_result = (filtered, tp, fp, fn, p, r, f1, mg, md)

    if best_result:
        filtered, tp, fp, fn, p, r, f1, mg, md = best_result
        print(f"\n  BEST: ratio>={best_thr:.1f} -> {len(filtered)} events  TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}")
        missed = [GT_TIMESTAMPS[gi] for gi in range(31) if gi not in mg]
        if missed: print(f"  Missed: {missed}")
    else:
        filtered = [e for e in after_iou if e["flow_ratio"] >= args.min_ratio]

    # Save JSON
    tag = f"cd{args.cooldown}_iou35_flow{int(best_thr*10):02d}"
    json_out = os.path.join(OUTPUT_SUB, f"results_H_{tag}.json")
    out_data = dict(raw_data)
    out_data["events"]    = filtered
    out_data["n_impacts"] = len(filtered)
    out_data["filters"]   = {
        "cooldown": args.cooldown,
        "max_iou": args.max_iou,
        "min_flow_ratio": best_thr,
    }
    os.makedirs(OUTPUT_SUB, exist_ok=True)
    with open(json_out, "w") as f2:
        json.dump(out_data, f2, indent=2)
    print(f"\n  JSON saved: {json_out}")

    # Render
    if not args.no_video:
        vid_out = os.path.join(OUTPUT_SUB, f"1_H_{tag}_real.mp4")
        render_video(filtered, f2d, VIDEO_IN, vid_out, src_fps)

    print()


if __name__ == "__main__":
    main()
