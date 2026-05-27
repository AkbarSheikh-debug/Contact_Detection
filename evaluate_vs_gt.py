"""
Evaluate any detection-event list against the 31 ground-truth impact timestamps.

Tries multiple FPS assumptions for the GT timestamps, plus multiple tolerance windows.
"""

import json
import os
import sys
from itertools import product

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACT_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
VIDEO_FPS  = 24.995    # measured fps of 3.mp4

# 31 ground-truth impact timestamps from the user (frame-of-second notation)
GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]


def parse_ts_to_sec(ts, label_fps):
    """Convert 'S:F' or 'M:S:F' to seconds, assuming F is at label_fps."""
    p = ts.split(":")
    if len(p) == 2:
        return int(p[0]) + int(p[1]) / label_fps
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / label_fps


def evaluate(det_frames, gt_frames, tol):
    matched_gt = set()
    matched_det = set()
    pairs = []
    for di, df in enumerate(det_frames):
        best, best_d = None, tol + 1
        for gi, gf in enumerate(gt_frames):
            if gi in matched_gt:
                continue
            d = abs(df - gf)
            if d < best_d:
                best_d, best = d, gi
        if best is not None:
            matched_gt.add(best)
            matched_det.add(di)
            pairs.append((di, best, best_d))

    tp = len(matched_gt)
    fp = len(det_frames) - len(matched_det)
    fn = len(gt_frames) - tp
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return tp, fp, fn, p, r, f1, matched_gt, pairs


def run_filter(events, region_filter, prob_min, dist_max, cooldown):
    """Filter contact_events and dedupe with cooldown."""
    kept = []
    last = -10**9
    for e in sorted(events, key=lambda x: x["frame"]):
        if e["contact_region"] not in region_filter:
            continue
        if e["contact_prob"] < prob_min:
            continue
        if e["contact_3d_distance_m"] > dist_max:
            continue
        if e["frame"] - last < cooldown:
            continue
        kept.append(e)
        last = e["frame"]
    return kept


def evaluate_config(events, region_filter, prob_min, dist_max, cooldown, gt_frames, tol):
    kept = run_filter(events, region_filter, prob_min, dist_max, cooldown)
    det_frames = [e["frame"] for e in kept]
    return evaluate(det_frames, gt_frames, tol), kept


def main():
    with open(SAM3D_JSON) as f:
        sam3d = json.load(f)
    contact_events = sam3d.get("contact_events", [])

    print(f"Loaded {len(contact_events)} raw contact_events")
    print(f"Video FPS: {VIDEO_FPS}")

    # Try different GT label fps assumptions
    for label_fps in [25.0, 30.0, 24.995]:
        gt_secs = [parse_ts_to_sec(t, label_fps) for t in GT_TIMESTAMPS]
        gt_frames = [int(round(s * VIDEO_FPS)) for s in gt_secs]
        print(f"\n{'='*70}")
        print(f"GT label fps={label_fps}  -> GT frames range {min(gt_frames)}-{max(gt_frames)}")
        print(f"{'='*70}")

        # ── Baseline: all head/torso contacts, no other filter ────────────────
        print("\nBaseline: head/torso only")
        for prob_min, dist_max, cd in product([0.20, 0.30, 0.40, 0.50],
                                              [0.10, 0.12, 0.15],
                                              [10, 15, 20]):
            (tp, fp, fn, p, r, f1, mg, _), kept = evaluate_config(
                contact_events, {"head", "torso"}, prob_min, dist_max, cd, gt_frames, tol=30)
            if f1 >= 0.30:
                print(f"  region=H+T prob>={prob_min:.2f} d<={dist_max:.2f} cd={cd:2d}  "
                      f"n={len(kept):3d}  TP={tp:2d} FP={fp:3d} FN={fn:2d}  "
                      f"P={p:.2f} R={r:.2f} F1={f1:.3f}")

        # ── ALL regions (include blocked, leg, etc.) ─────────────────────────
        print("\nALL regions:")
        for prob_min, dist_max, cd in product([0.30, 0.50, 0.70],
                                              [0.10, 0.15],
                                              [10, 15, 20]):
            (tp, fp, fn, p, r, f1, mg, _), kept = evaluate_config(
                contact_events,
                {"head","torso","left_arm","right_arm","left_leg","right_leg"},
                prob_min, dist_max, cd, gt_frames, tol=30)
            if f1 >= 0.30:
                print(f"  region=ALL prob>={prob_min:.2f} d<={dist_max:.2f} cd={cd:2d}  "
                      f"n={len(kept):3d}  TP={tp:2d} FP={fp:3d} FN={fn:2d}  "
                      f"P={p:.2f} R={r:.2f} F1={f1:.3f}")

    # ── Detailed report of best config ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED BEST CONFIG (sweep)")
    print("=" * 70)
    best = None
    for label_fps in [25.0, 30.0]:
        gt_secs = [parse_ts_to_sec(t, label_fps) for t in GT_TIMESTAMPS]
        gt_frames = [int(round(s * VIDEO_FPS)) for s in gt_secs]
        for regions in [{"head", "torso"},
                         {"head", "torso", "left_arm", "right_arm"},
                         {"head","torso","left_arm","right_arm","left_leg","right_leg"}]:
            for prob_min in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                for dist_max in [0.08, 0.10, 0.12, 0.15, 0.20]:
                    for cd in [8, 10, 12, 15, 20, 25]:
                        for tol in [20, 30, 45]:
                            res, kept = evaluate_config(contact_events, regions,
                                                        prob_min, dist_max, cd, gt_frames, tol)
                            tp, fp, fn, p, r, f1, mg, pairs = res
                            if best is None or f1 > best[0]:
                                best = (f1, p, r, tp, fp, fn,
                                        label_fps, regions, prob_min, dist_max, cd, tol,
                                        kept, mg, gt_frames, pairs)
    (f1, p, r, tp, fp, fn,
     label_fps, regions, prob_min, dist_max, cd, tol,
     kept, mg, gt_frames, pairs) = best
    print(f"\nBest: F1={f1:.3f} P={p:.2f} R={r:.2f}  TP={tp} FP={fp} FN={fn}")
    print(f"  label_fps={label_fps}  regions={regions}")
    print(f"  prob>={prob_min}  d<={dist_max}m  cd={cd}fr  tol={tol}fr")
    print(f"  detections: {len(kept)}")

    print("\nMatched GT impacts:")
    by_gt = {gi: (di, d) for (di, gi, d) in pairs}
    for gi, ts in enumerate(GT_TIMESTAMPS):
        gf = gt_frames[gi]
        if gi in by_gt:
            di, dd = by_gt[gi]
            e = kept[di]
            print(f"  [HIT]  {ts:8s} (frame {gf:4d})  <- det frame {e['frame']:4d}  "
                  f"diff={dd:2d}fr  prob={e['contact_prob']:.2f} "
                  f"{e['striker_body_part']}->{e['contact_region']}")
        else:
            print(f"  [MISS] {ts:8s} (frame {gf:4d})")


if __name__ == "__main__":
    main()
