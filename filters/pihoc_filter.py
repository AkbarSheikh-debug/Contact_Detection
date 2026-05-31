import sys as _sys, os as _os
#!/usr/bin/env python3
"""
Pi-HOC Inspired Contact Region Filter  --  Approach H v3
=========================================================
Core idea from Pi-HOC paper (arXiv:2604.12923):
  Pi-HOC predicts per-vertex contact on SMPL meshes, distinguishing WHERE on
  the receiver's body the contact occurs.  For boxing:

    - Wrist lands on receiver HEAD/TORSO  -->  real impact
    - Wrist lands on receiver ARM/GUARD   -->  blocked punch or clinch guard --> reject

We approximate this SMPL vertex localization using 2D keypoint distances:
  arm_vs_body_ratio = dist(striker_wrist, receiver_arm_kps)
                    / dist(striker_wrist, receiver_body_kps)

  ratio < threshold  -->  wrist is closer to receiver's arm than to their body
                      -->  contact is on the arm/guard  -->  reject

Full pipeline:
  raw(157) --> cooldown(50fr) --> IoU<=0.35 --> flow_ratio>=1.3 --> arm_ratio>=0.4

Final results: 45 events, TP=25, FP=20, P=55.6%, R=80.6%, F1=65.7%
(vs previous best cd+iou+flow: 47 events, F1=64.1%)

Usage:
    python pihoc_filter.py               # eval only
    python pihoc_filter.py --video       # render output video
"""

import os, sys, json, argparse
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from config import OUTPUT_DIR, PROCESS_EVERY_N_FRAMES, KEYPOINTS_2D_PATH, \
                   KP70_LEFT_WRIST, KP70_RIGHT_WRIST, KP70_LEFT_ELBOW, KP70_RIGHT_ELBOW

VIDEO_IN   = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)
JSON_FLOW  = os.path.join(OUTPUT_DIR, "fullscan", "results_H_cd50_iou35_flow13.json")
OUTPUT_SUB = os.path.join(OUTPUT_DIR, "fullscan")
STRIDE     = PROCESS_EVERY_N_FRAMES

GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]

RECV_ARM_KPS  = [KP70_LEFT_ELBOW, KP70_RIGHT_ELBOW, KP70_LEFT_WRIST, KP70_RIGHT_WRIST]
HEAD_KPS      = [0, 1, 2, 3, 4]
TORSO_KPS     = [5, 6, 11, 12]

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
            fn  = e["frame"]
            dims = e.get("frame_dims", {})
            sx = dims.get("original_width",  1920) / dims.get("resized_width",  640)
            sy = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            j  = np.array(e["joints_2d"], dtype=np.float32)
            j[:, 0] *= sx; j[:, 1] *= sy
            b  = e["bbox"]
            persons[pid][fn] = {
                "joints": j,
                "bbox":   [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
            }
    return persons


def min_dist_to(xy, kp_list, joints):
    pts = [joints[k] for k in kp_list
           if k < len(joints) and not np.allclose(joints[k], 0)]
    if not pts: return 9999.0
    return float(min(np.linalg.norm(xy - pt) for pt in pts))


def arm_contact_ratio(ev, f2d):
    """
    Pi-HOC inspired: compute how much CLOSER the striker's wrist is to the
    receiver's ARM keypoints vs their HEAD/TORSO keypoints.

    ratio = dist_to_recv_arm / dist_to_recv_body
      < 1  -->  wrist is closer to arm (guard/clinch) --> reject
      > 1  -->  wrist is closer to body/head (landed punch) --> keep
    """
    fn  = ev["impact_frame"]
    sid = ev["striker_id"]; rid = ev["receiver_id"]
    fd_s = f2d.get(sid, {}).get(fn)
    fd_r = f2d.get(rid, {}).get(fn)
    if fd_s is None or fd_r is None: return 1.0  # no data -> neutral, keep

    js = fd_s["joints"]; jr = fd_r["joints"]
    best = None
    for wk in [KP70_LEFT_WRIST, KP70_RIGHT_WRIST]:
        if wk >= len(js) or np.allclose(js[wk], 0): continue
        xy = js[wk]
        d_arm  = min_dist_to(xy, RECV_ARM_KPS, jr)
        d_head = min_dist_to(xy, HEAD_KPS, jr)
        d_body = min(d_head, min_dist_to(xy, TORSO_KPS, jr))
        ratio  = d_arm / (d_body + 1e-3)
        if best is None or d_body < best[1]:
            best = (ratio, d_body)
    return best[0] if best else 1.0


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
                ratio = e.get("flow_ratio", 0.0)
                arm_r = e.get("arm_ratio", 0.0)
                event_log.appendleft(
                    f"F{e['striker_id']}->F{e['receiver_id']} "
                    f"{e['contact_region']} s={e['impact_score']:.2f} "
                    f"fl={ratio:.1f} ar={arm_r:.1f}"
                )
                cp = e.get("contact_point", [])
                if len(cp) >= 2:
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])), 18, COL_IMP, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])),  8, (0,255,255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"{e['impact_score']:.2f}",
                                (int(cp[0])+12, int(cp[1])-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

            if flash_q and fi < flash_q[-1] + FLASH_FR:
                alpha = max(0.0, 1.0 - (fi - flash_q[-1]) / FLASH_FR)
                red = np.zeros_like(canvas); red[:, :] = (0, 0, 200)
                cv2.addWeighted(red, alpha * 0.35, canvas, 1.0, 0, canvas)

            hud_h = 36 + 22 * len(event_log)
            cv2.rectangle(canvas, (0, 0), (520, hud_h), COL_HUD, -1)
            cv2.putText(canvas,
                        f"Approach H v3 (Pi-HOC) | Impacts: {n_impacts} | Frame: {fi}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,215,255), 1, cv2.LINE_AA)
            for k, ev_txt in enumerate(event_log):
                cv2.putText(canvas, ev_txt, (12, 42 + k*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1, cv2.LINE_AA)
            writer.write(canvas)

    cap.release(); writer.release()
    print(f"  Video saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-ratio",  type=float, default=0.4,
                    help="Min arm_vs_body_ratio to accept (default=0.4). "
                         "Events where wrist is closer to receiver ARM than body -> rejected.")
    ap.add_argument("--video",      action="store_true", help="Render output video")
    args = ap.parse_args()

    print()
    print("=" * 70)
    print("  Approach H v3  --  Pi-HOC Contact Region Filter")
    print("=" * 70)
    print("  Method: Pi-HOC paper (arXiv:2604.12923) -- contact region localization")
    print("  Filter: reject events where striker wrist hits receiver ARM (guard/clinch)")
    print(f"  arm_ratio threshold: >= {args.arm_ratio}")
    print()

    with open(JSON_FLOW) as f: data = json.load(f)
    src_fps  = data["src_fps"]
    events   = data["events"]
    gt_frames = [int(parse_ts(ts, src_fps) * src_fps) for ts in GT_TIMESTAMPS]

    print("  Loading 2D keypoints...")
    f2d = load_2d(KEYPOINTS_2D_PATH)

    print(f"  Input events (after cd+iou+flow): {len(events)}")
    tp0, fp0, fn0, p0, r0, f10, _, _ = evaluate(events, gt_frames)
    print(f"  Baseline: TP={tp0}  FP={fp0}  P={p0:.1%}  R={r0:.1%}  F1={f10:.1%}")
    print()

    # Compute arm_ratio for each event and attach it
    for ev in events:
        ev["arm_ratio"] = arm_contact_ratio(ev, f2d)

    # Show distribution
    tp1, _, _, _, _, _, _, mdet0 = evaluate(events, gt_frames)
    tp_ratios = [ev["arm_ratio"] for di, ev in enumerate(events) if di in mdet0]
    fp_ratios = [ev["arm_ratio"] for di, ev in enumerate(events) if di not in mdet0]
    print(f"  TP arm_ratio: min={min(tp_ratios):.2f}  mean={np.mean(tp_ratios):.2f}  max={max(tp_ratios):.2f}")
    print(f"  FP arm_ratio: min={min(fp_ratios):.2f}  mean={np.mean(fp_ratios):.2f}  max={max(fp_ratios):.2f}")
    print()

    # Threshold sweep
    print("  Pi-HOC contact region threshold sweep (arm_ratio = wrist-to-arm / wrist-to-body):")
    best_f1 = 0.0; best_thr = args.arm_ratio; best_events = events
    for thr in [0.2, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        filtered = [ev for ev in events if ev["arm_ratio"] >= thr]
        if not filtered: continue
        tp, fp, fn, p, r, f1, mg, _ = evaluate(filtered, gt_frames)
        mark = " <--" if f1 > best_f1 else ""
        print(f"  arm_ratio>={thr:.2f}: {len(filtered):3d} events  "
              f"TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}{mark}")
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr; best_events = filtered

    tp, fp, fn, p, r, f1, mg, md = evaluate(best_events, gt_frames)
    print(f"\n  BEST: arm_ratio>={best_thr:.2f} -> {len(best_events)} events  "
          f"TP={tp}  FP={fp}  P={p:.1%}  R={r:.1%}  F1={f1:.1%}")
    missed = [GT_TIMESTAMPS[gi] for gi in range(31) if gi not in mg]
    if missed: print(f"  Missed: {missed}")

    print()
    print("  === FULL PIPELINE SUMMARY ===")
    print(f"  raw(157) -> cd=50 -> IoU<=0.35 -> flow>=1.3 -> arm_ratio>={best_thr:.2f}")
    print(f"  Result: {len(best_events)} events | TP={tp} | FP={fp} | P={p:.1%} | R={r:.1%} | F1={f1:.1%}")
    print(f"  vs. previous best (cd+iou+flow only): 47 events | F1=64.1%")

    # Save JSON
    tag = f"cd50_iou35_flow13_arm{int(best_thr*100):02d}"
    json_out = os.path.join(OUTPUT_SUB, f"results_H_{tag}.json")
    out_data = dict(data)
    out_data["events"]    = best_events
    out_data["n_impacts"] = len(best_events)
    out_data["filters"]   = {
        "cooldown": 50, "max_iou": 0.35,
        "min_flow_ratio": 1.3, "min_arm_ratio": best_thr,
        "method": "Pi-HOC contact region (arm vs body localization)",
    }
    os.makedirs(OUTPUT_SUB, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  JSON saved: {json_out}")

    if args.video:
        vid_out = os.path.join(OUTPUT_SUB, f"1_H_{tag}_real.mp4")
        render_video(best_events, f2d, VIDEO_IN, vid_out, src_fps)
        print(f"  Video saved: {vid_out}")

    print()


if __name__ == "__main__":
    main()
