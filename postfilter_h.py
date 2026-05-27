#!/usr/bin/env python3
"""
Post-filter + renderer for Approach H results.
Optimal filter: global cooldown=50 frames (2s) + IoU<=0.35 (removes clinch events).
Grid-searched against 31 GT timestamps → F1=60.5%, P=47.3%, R=83.9%.

Usage:
    python postfilter_h.py                         # default: cd=50, iou<=0.35
    python postfilter_h.py --cooldown 30           # higher recall, more FPs
    python postfilter_h.py --max-iou 1.0           # disable IoU filter
    python postfilter_h.py --no-video
"""

import os, sys, json, argparse
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, PROCESS_EVERY_N_FRAMES, KP70_LEFT_WRIST, KP70_RIGHT_WRIST,\
                   KEYPOINTS_2D_PATH

JSON_IN  = os.path.join(OUTPUT_DIR, "fullscan", "results_H_fullscan.json")
VIDEO_IN = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)
OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "fullscan")

STRIDE    = PROCESS_EVERY_N_FRAMES   # 2
FLASH_FR  = 15
COL_F0    = (0, 255, 180)
COL_F1    = (255, 140, 0)
COL_WRIST = (0, 180, 255)
COL_IMP   = (0, 50, 255)
COL_HUD   = (15, 15, 20)

SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]

def parse_ts(ts, fps):
    p = ts.split(":")
    if len(p) == 2: return int(p[0]) + int(p[1]) / fps
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / fps


def apply_cooldown(events, cooldown_frames):
    filtered = []
    last = -9999
    for e in events:
        if e["impact_frame"] - last >= cooldown_frames:
            filtered.append(e)
            last = e["impact_frame"]
    return filtered


def evaluate_gt(events, src_fps, tol=30):
    gt_frames = [int(parse_ts(ts, src_fps) * src_fps) for ts in GT_TIMESTAMPS]
    det_frames = [e["impact_frame"] for e in events]
    matched_gt = set(); matched_det = set()
    for di, df in enumerate(det_frames):
        for gi, gf in enumerate(gt_frames):
            if gi in matched_gt: continue
            if abs(df - gf) <= tol:
                matched_gt.add(gi); matched_det.add(di); break
    tp = len(matched_gt); fp = len(det_frames) - len(matched_det); fn = 31 - tp
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    print(f"\n  GT evaluation (tol={tol} frames = ±{tol/src_fps:.1f}s)")
    print(f"  Detected={len(det_frames)}  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision={p:.1%}  Recall={r:.1%}  F1={f1:.1%}")
    print(f"\n  Matched ({tp}): " + ", ".join(GT_TIMESTAMPS[gi] for gi in sorted(matched_gt)))
    missed = [GT_TIMESTAMPS[gi] for gi in range(31) if gi not in matched_gt]
    if missed:
        print(f"  Missed  ({fn}): " + ", ".join(missed))
    return tp, fp, fn, p, r, f1


def load_2d_fast(path):
    with open(path) as f: raw = json.load(f)
    persons = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str); persons[pid] = {}
        for e in entries:
            fn = e["frame"]
            dims = e.get("frame_dims", {})
            sx = dims.get("original_width",  1920) / dims.get("resized_width",  640)
            sy = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            j = np.array(e["joints_2d"], dtype=np.float32)
            j[:, 0] *= sx; j[:, 1] *= sy
            b = e["bbox"]
            scaled_bbox = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
            persons[pid][fn] = {"joints": j, "bbox": scaled_bbox}
    return persons


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
                r = impact_map[fi]
                flash_q.clear(); flash_q.append(fi); n_impacts += 1
                event_log.appendleft(
                    f"F{r['striker_id']}->F{r['receiver_id']} {r['contact_region']} {r['impact_score']:.2f}"
                )
                cp = r.get("contact_point", [])
                if len(cp) >= 2:
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])), 18, COL_IMP, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])),  8, (0,255,255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"{r['impact_score']:.2f}",
                                (int(cp[0])+12, int(cp[1])-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                    # Gate breakdown bar
                    gates = r.get("gates", {})
                    gate_order = ["sam","decel","receiver_react","prox_3d","jerk","extension"]
                    gate_cols  = [(0,200,255),(0,255,100),(255,100,0),(200,0,255),(0,100,255),(255,200,0)]
                    bx, by = int(cp[0])+30, int(cp[1])-10
                    for gi, (gk, gc) in enumerate(zip(gate_order, gate_cols)):
                        gv = gates.get(gk, 0)
                        bw = int(gv * 40)
                        cv2.rectangle(canvas, (bx, by+gi*8), (bx+bw, by+gi*8+6), gc, -1)

            if flash_q and fi < flash_q[-1] + FLASH_FR:
                alpha = max(0.0, 1.0 - (fi - flash_q[-1]) / FLASH_FR)
                red = np.zeros_like(canvas); red[:, :] = (0, 0, 200)
                cv2.addWeighted(red, alpha * 0.35, canvas, 1.0, 0, canvas)

            hud_h = 36 + 22 * len(event_log)
            cv2.rectangle(canvas, (0, 0), (440, hud_h), COL_HUD, -1)
            cv2.putText(canvas,
                        f"Approach H (filtered) | Impacts: {n_impacts} | Frame: {fi}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,215,255), 1, cv2.LINE_AA)
            for k, ev_txt in enumerate(event_log):
                cv2.putText(canvas, ev_txt, (12, 42+k*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1, cv2.LINE_AA)

            writer.write(canvas)

    cap.release(); writer.release()
    print(f"  Video saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cooldown", type=int, default=50,
                    help="Global cooldown in source frames (default=50 ≈ 2s)")
    ap.add_argument("--max-iou",   type=float, default=0.35,
                    help="Max bbox IoU between fighters to accept (0.35 filters clinch frames)")
    ap.add_argument("--no-video",  action="store_true")
    ap.add_argument("--eval-gt",   action="store_true", default=True)
    args = ap.parse_args()

    print()
    print("=" * 68)
    print("  Approach H — Post-filter + Render")
    print("=" * 68)

    # Load 2D bboxes for IoU filter
    print("  Loading 2D keypoints for IoU filter and rendering...")
    f2d = load_2d_fast(KEYPOINTS_2D_PATH)

    def _bbox_iou(b0, b1):
        if b0 is None or b1 is None: return 0.0
        x1 = max(b0[0], b1[0]); y1 = max(b0[1], b1[1])
        x2 = min(b0[2], b1[2]); y2 = min(b0[3], b1[3])
        if x2 <= x1 or y2 <= y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a0 = (b0[2]-b0[0]) * (b0[3]-b0[1])
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        return inter / (a0 + a1 - inter + 1e-6)

    def event_iou(e, f2d_):
        fn = e["impact_frame"]
        sid = e["striker_id"]; rid = e["receiver_id"]
        fd_s = f2d_.get(sid, {}).get(fn); fd_r = f2d_.get(rid, {}).get(fn)
        if fd_s is None or fd_r is None: return 0.0
        return _bbox_iou(fd_s["bbox"], fd_r["bbox"])

    with open(JSON_IN) as f:
        data = json.load(f)
    src_fps = data["src_fps"]
    raw_events = data["events"]
    print(f"  Loaded {len(raw_events)} raw detections")
    print(f"  Filter: global cooldown={args.cooldown}fr ({args.cooldown/src_fps:.1f}s)  max_iou={args.max_iou}")

    # Step 1: global cooldown
    after_cd = apply_cooldown(raw_events, args.cooldown)
    print(f"  After cooldown: {len(after_cd)} detections")

    # Step 2: IoU filter — remove clinch events (both bodies heavily overlapping)
    filtered = [e for e in after_cd if event_iou(e, f2d) <= args.max_iou]
    n_iou_removed = len(after_cd) - len(filtered)
    print(f"  After IoU filter (removed {n_iou_removed} clinch events): {len(filtered)} detections")

    if args.eval_gt:
        evaluate_gt(filtered, src_fps)

    # Save filtered JSON
    tag = f"cd{args.cooldown}_iou{int(args.max_iou*100)}"
    json_out = os.path.join(OUTPUT_SUBDIR, f"results_H_{tag}.json")
    out_data = dict(data)
    out_data["events"]    = filtered
    out_data["n_impacts"] = len(filtered)
    out_data["filters"]   = {"global_cooldown": args.cooldown, "max_bbox_iou": args.max_iou}
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  Filtered JSON: {json_out}")

    if not args.no_video:
        vid_out = os.path.join(OUTPUT_SUBDIR, f"1_H_{tag}_real.mp4")
        render_video(filtered, f2d, VIDEO_IN, vid_out, src_fps)
        print(f"  Final video: {vid_out}")


if __name__ == "__main__":
    main()
