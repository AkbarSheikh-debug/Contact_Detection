#!/usr/bin/env python3
"""
Hybrid Impact Detection Pipeline
==================================
Combines pre-extracted SAM3D keypoints (fighter_0) with live YOLO-pose
detection (opponent) to perform proper proximity-based impact detection.

Why hybrid?
  - External keypoints are accurate but only track ONE fighter.
  - Impact detection requires BOTH fighters: wrist-to-opponent proximity,
    bounding box overlap, directed velocity toward the opponent.
  - YOLO detects the opponent each frame; the existing PairInteractionAnalyzer
    handles all contact scoring using both fighters' keypoints.

Architecture:
  1. Load external 2D keypoints for fighter_0 (from SAM3D)
  2. For each video frame:
     a. Run YOLO-pose to detect all persons
     b. Match one YOLO detection to fighter_0 (by bbox IoU with external data)
     c. REPLACE that detection's keypoints with the accurate external ones
     d. Select the opponent as the other large non-referee detection
     e. Feed BOTH into PairInteractionAnalyzer + ImpactClassifier
  3. Render annotated output video showing both fighters + impact events
"""
import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import PersonDetector
from models.tracker import PersonTracker
from impact_detection.pair_analyzer import PairInteractionAnalyzer
from impact_detection.impact_classifier import ImpactClassifier, ImpactEvent
from keypoint_loader import KeypointLoader, FrameData2D
from config import (
    YOLO_MODEL, KEYPOINTS_2D_PATH, KEYPOINTS_3D_PATH, ACTIONS_PATH,
    OUTPUT_DIR, PROCESS_EVERY_N_FRAMES, MIN_PERSON_AREA_RATIO,
    MAX_PERSON_POOL, KP70_LEFT_WRIST, KP70_RIGHT_WRIST,
    IMPACT_FLASH_DURATION,
)

DEFAULT_VIDEO = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)

# ── Skeleton for the 17 COCO body keypoints (indices 0-16 of the 70-joint model)
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # face
    (5, 6),                                  # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

# Colours
COL_FIGHTER_A = (0, 255, 200)   # cyan-green for tracked fighter
COL_FIGHTER_B = (255, 140, 0)   # orange for opponent
COL_WRIST     = (0, 180, 255)   # bright orange wrist highlight
COL_IMPACT    = (0, 50, 255)    # red flash
COL_HUD_TITLE = (0, 215, 255)   # yellow-cyan
COL_HUD_TEXT  = (200, 200, 200)

FLASH_DURATION = 10


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Impact Detection: External keypoints + YOLO opponent"
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--kp2d", default=KEYPOINTS_2D_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=PROCESS_EVERY_N_FRAMES,
                        help="Process every N-th frame (default from config)")
    args = parser.parse_args()

    print("")
    print("=" * 70)
    print("  Hybrid Impact Detection Pipeline")
    print("  External keypoints (fighter) + YOLO-pose (opponent)")
    print("=" * 70)

    t0 = time.time()

    # ── 1. Load external 2D keypoints ────────────────────────────────────
    loader = KeypointLoader()
    ext_frames = loader.load_2d(args.kp2d)

    # ── 2. Initialise YOLO + tracker + contact analysis ──────────────────
    print("\n[Pipeline] Initialising modules ...")
    detector = PersonDetector(YOLO_MODEL)
    tracker = PersonTracker()
    analyzer = PairInteractionAnalyzer()
    classifier = ImpactClassifier()

    # ── 3. Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = W * H

    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{base}_impact_detected.mp4")

    out_fps = fps / args.stride
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (W, H))

    print(f"\n  Input      : {args.video}")
    print(f"  Output     : {out_path}")
    print(f"  Resolution : {W}x{H} @ {fps:.1f} fps")
    print(f"  Total      : {total_frames} frames")
    print(f"  Stride     : every {args.stride} frames")
    print(f"  External   : {len(ext_frames)} keypoint frames loaded")
    print()

    # ── 4. Processing loop ───────────────────────────────────────────────
    flash_queue = deque()
    event_log = deque(maxlen=6)
    fighter_a_id = None   # track_id assigned to the external-keypoint fighter
    frame_idx = 0
    processed = 0

    with tqdm(total=total_frames, unit="fr", desc="Processing") as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.stride != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # ── YOLO detection ───────────────────────────────────────
            persons = detector.detect(frame)

            # Filter small detections
            persons = [
                p for p in persons
                if ((p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1])
                    / frame_area) >= MIN_PERSON_AREA_RATIO
            ]

            # Track
            persons = tracker.update(persons)

            # Sort by area, keep top pool
            persons = sorted(
                persons,
                key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]),
                reverse=True,
            )[:MAX_PERSON_POOL]

            # ── Match external keypoints to YOLO detection ───────────
            ext_data = ext_frames.get(frame_idx)
            if ext_data is not None and persons:
                best_match_idx = _match_external_to_yolo(ext_data, persons)
                if best_match_idx >= 0:
                    # Replace YOLO keypoints with accurate external ones
                    persons[best_match_idx]["keypoints"] = _convert_70_to_17(
                        ext_data.joints_2d
                    )
                    fighter_a_id = persons[best_match_idx]["track_id"]

            # ── Select 2 fighters (exclude referee) ──────────────────
            fighters = _select_two_fighters(persons, W, frame, fighter_a_id)

            # ── Contact analysis ─────────────────────────────────────
            pairs = analyzer.form_pairs(fighters)
            contacts = analyzer.analyze(fighters, pairs, frame_idx)
            new_impacts = classifier.process(contacts, frame_idx, fps)

            # ── Render annotated frame ───────────────────────────────
            canvas = frame.copy()
            _draw_fighters(canvas, fighters, fighter_a_id)
            _draw_pair_lines(canvas, fighters, pairs)
            _register_impacts(new_impacts, flash_queue, event_log, frame_idx)
            _draw_flashes(canvas, flash_queue, frame_idx)
            _draw_hud(canvas, frame_idx, fps, len(classifier.events),
                      event_log, W, H)
            _draw_footer(canvas, W, H)

            writer.write(canvas)

            pbar.update(args.stride)
            frame_idx += args.stride
            processed += 1

    cap.release()
    writer.release()

    # ── 5. Summary ───────────────────────────────────────────────────────
    summary = classifier.summary()
    elapsed = time.time() - t0

    print(f"\n{'-' * 60}")
    print(f"  Processing complete!")
    print(f"  Frames processed   : {processed}")
    print(f"  Total impacts      : {summary['total_impacts']}")
    print(f"    Head impacts     : {summary['head_impacts']}")
    print(f"    Torso impacts    : {summary['torso_impacts']}")
    print(f"  Output video       : {out_path}")
    print(f"  Time               : {elapsed:.1f}s")
    print(f"{'-' * 60}")

    # Print event table
    if summary["events"]:
        print(f"\n  {'#':>3}  {'Time':>6}  {'Label':<30}  {'Prob':>5}  {'Vel':>6}")
        print("  " + "-" * 60)
        for i, ev in enumerate(summary["events"], 1):
            m, s = divmod(int(ev.time_sec), 60)
            print(f"  {i:>3}  {m:02d}:{s:02d}  {ev.label:<30}  "
                  f"{ev.probability:>4.0%}  {ev.velocity:>5.1f}")
    else:
        print("\n  No impact events detected.")

    # Save JSON
    _save_events_json(summary["events"], fps, out_path)

    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 70 + "\n")
    return 0


# ═════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════

def _match_external_to_yolo(ext_data: FrameData2D, persons: list[dict]) -> int:
    """
    Find which YOLO detection corresponds to the externally-tracked fighter
    by comparing bounding box IoU.
    """
    ext_bbox = ext_data.bbox
    best_iou = 0.0
    best_idx = -1

    for i, p in enumerate(persons):
        iou = _box_iou(ext_bbox, p["bbox"].astype(float))
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    # Require minimum IoU to accept the match
    return best_idx if best_iou >= 0.15 else -1


def _convert_70_to_17(joints_70: np.ndarray) -> dict:
    """
    Convert 70-joint keypoints to 17-joint COCO format expected by
    PairInteractionAnalyzer.

    Returns keypoints dict: {points: (17,2), confidence: (17,)}
    """
    pts_17 = joints_70[:17].copy().astype(np.float32)
    # Set confidence to 1.0 for all joints (external data is high quality)
    conf_17 = np.ones(17, dtype=np.float32)
    # Zero out confidence for any zero-position joints
    for i in range(17):
        if np.allclose(pts_17[i], 0):
            conf_17[i] = 0.0
    return {"points": pts_17, "confidence": conf_17}


def _select_two_fighters(
    persons: list[dict],
    frame_w: int,
    frame: np.ndarray,
    fighter_a_id: int | None,
) -> list[dict]:
    """
    Select exactly 2 fighters from detected persons.
    If fighter_a_id is known, always include that one.
    Exclude referees (white-shirt heuristic).
    """
    if len(persons) <= 2:
        return persons

    # Filter out referees
    non_refs = []
    for p in persons:
        if not _is_referee(p, frame):
            non_refs.append(p)

    if len(non_refs) < 2:
        non_refs = persons[:2]

    # If we know fighter_a, always include them
    if fighter_a_id is not None:
        fighter_a = None
        others = []
        for p in non_refs:
            if p["track_id"] == fighter_a_id:
                fighter_a = p
            else:
                others.append(p)

        if fighter_a and others:
            # Pick the closest other person as opponent
            a_cx = (fighter_a["bbox"][0] + fighter_a["bbox"][2]) / 2
            a_cy = (fighter_a["bbox"][1] + fighter_a["bbox"][3]) / 2
            best_dist = float("inf")
            best_other = others[0]
            for o in others:
                o_cx = (o["bbox"][0] + o["bbox"][2]) / 2
                o_cy = (o["bbox"][1] + o["bbox"][3]) / 2
                dist = ((a_cx - o_cx) ** 2 + (a_cy - o_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_other = o
            return [fighter_a, best_other]
        elif fighter_a:
            return [fighter_a]

    # Fallback: top 2 by area
    return sorted(
        non_refs,
        key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]),
        reverse=True,
    )[:2]


def _is_referee(person: dict, frame: np.ndarray) -> bool:
    """Referee heuristic: bright, low-saturation upper body."""
    x1, y1, x2, y2 = person["bbox"]
    h = y2 - y1
    crop_y2 = min(int(y1 + h * 0.45), frame.shape[0])
    crop = frame[y1:crop_y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean()) > 165 and float(hsv[:, :, 1].mean()) < 55


def _box_iou(a, b) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


# ═════════════════════════════════════════════════════════════════════════
# Drawing
# ═════════════════════════════════════════════════════════════════════════

def _draw_fighters(canvas, fighters, fighter_a_id):
    """Draw bbox + skeleton for both fighters with different colours."""
    for p in fighters:
        is_a = (p["track_id"] == fighter_a_id)
        color = COL_FIGHTER_A if is_a else COL_FIGHTER_B
        label = "Fighter A (tracked)" if is_a else "Opponent"

        # Bbox
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{label} ID:{p['track_id']}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Skeleton
        if p["keypoints"] is not None:
            pts = p["keypoints"]["points"]
            conf = p["keypoints"]["confidence"]

            for a, b in SKELETON_PAIRS:
                if a >= len(pts) or b >= len(pts):
                    continue
                if conf[a] < 0.25 or conf[b] < 0.25:
                    continue
                pa = tuple(pts[a].astype(int))
                pb = tuple(pts[b].astype(int))
                if pa == (0, 0) or pb == (0, 0):
                    continue
                cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)

            for i in range(min(17, len(pts))):
                if conf[i] < 0.25 or np.allclose(pts[i], 0):
                    continue
                if i in (9, 10):  # wrists
                    cv2.circle(canvas, tuple(pts[i].astype(int)),
                               7, COL_WRIST, -1, cv2.LINE_AA)
                    cv2.circle(canvas, tuple(pts[i].astype(int)),
                               10, COL_WRIST, 2, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, tuple(pts[i].astype(int)),
                               4, color, -1, cv2.LINE_AA)


def _draw_pair_lines(canvas, fighters, pairs):
    """Draw interaction lines between paired fighters."""
    for idx_a, idx_b in pairs:
        if idx_a < len(fighters) and idx_b < len(fighters):
            ca = _box_center(fighters[idx_a]["bbox"])
            cb = _box_center(fighters[idx_b]["bbox"])
            cv2.line(canvas, tuple(ca.astype(int)), tuple(cb.astype(int)),
                     (180, 180, 60), 1, cv2.LINE_AA)


def _register_impacts(new_impacts, flash_queue, event_log, frame_idx):
    """Queue flash effects and log events."""
    for ev in new_impacts:
        if ev.contact_point:
            flash_queue.append((
                frame_idx + FLASH_DURATION,
                ev.contact_point,
                ev.label,
                ev.probability,
            ))
        m, s = divmod(int(ev.time_sec), 60)
        event_log.append(
            f"[{m:02d}:{s:02d}] {ev.label}  p={ev.probability:.2f}"
        )


def _draw_flashes(canvas, flash_queue, frame_idx):
    """Render impact flash effects."""
    still_active = deque()
    for expire, pt, label, prob in flash_queue:
        if frame_idx <= expire:
            still_active.append((expire, pt, label, prob))
            age = FLASH_DURATION - (expire - frame_idx)
            alpha = max(0.0, 1.0 - age / FLASH_DURATION)

            if pt:
                radius = 35 + age * 10
                thickness = max(1, 4 - age // 3)
                color = (0, int(80 * alpha), int(255 * alpha))
                cv2.circle(canvas, tuple(pt), radius, color, thickness, cv2.LINE_AA)

                if age < 4:
                    ov = canvas.copy()
                    cv2.circle(ov, tuple(pt), 22, (0, 60, 255), -1)
                    cv2.addWeighted(ov, 0.3 * alpha, canvas, 1 - 0.3 * alpha, 0, canvas)

                cv2.putText(canvas, f"IMPACT! {prob:.0%}",
                            (pt[0] - 70, pt[1] - radius - 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75,
                            (0, int(100 * alpha), int(255 * alpha)),
                            2, cv2.LINE_AA)

            # Red border flash
            if age < 5:
                h, w = canvas.shape[:2]
                b = max(4, 14 - age * 2)
                ov = canvas.copy()
                for rect in [(0, 0, w, b), (0, h-b, w, h),
                             (0, 0, b, h), (w-b, 0, w, h)]:
                    cv2.rectangle(ov, (rect[0], rect[1]),
                                  (rect[2], rect[3]), COL_IMPACT, -1)
                cv2.addWeighted(ov, 0.5 * alpha, canvas, 1 - 0.5 * alpha, 0, canvas)

    flash_queue.clear()
    flash_queue.extend(still_active)


def _draw_hud(canvas, frame_idx, fps, total_impacts, event_log, W, H):
    """Draw HUD panel."""
    panel_w = 420
    panel_h = 200

    ov = canvas.copy()
    cv2.rectangle(ov, (8, 8), (8 + panel_w, 8 + panel_h), (15, 15, 20), -1)
    cv2.addWeighted(ov, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (8, 8), (8 + panel_w, 8 + panel_h), (40, 40, 50), 1)

    ts = frame_idx / max(fps, 1e-6)
    m, s = divmod(int(ts), 60)

    lines = [
        ("SAM3D Hybrid Impact Detection", COL_HUD_TITLE, 0.58, 2),
        (f"Frame: {frame_idx:5d}   Time: {m:02d}:{s:02d}", COL_HUD_TEXT, 0.48, 1),
        (f"Total Impacts: {total_impacts}", COL_HUD_TEXT, 0.48, 1),
        ("-" * 42, (80, 80, 80), 0.38, 1),
    ]
    for entry in list(event_log)[-4:]:
        lines.append((entry, (0, 230, 118), 0.40, 1))
    if not event_log:
        lines.append(("  (no impacts yet)", (100, 100, 100), 0.40, 1))

    y = 32
    for text, color, scale, thick in lines:
        cv2.putText(canvas, text, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += 22


def _draw_footer(canvas, W, H):
    label = "Pi-HOC Contact Estimation | SAM3D External Keypoints + YOLO Opponent"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, label, (W - tw - 12, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)


def _box_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def _save_events_json(events: list[ImpactEvent], fps: float, video_out: str):
    """Save detected impact events to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "hybrid_impact_results.json")

    data = {
        "total_impacts": len(events),
        "head_impacts": sum(1 for e in events if e.contact_region == "head"),
        "torso_impacts": sum(1 for e in events if e.contact_region == "torso"),
        "output_video": video_out,
        "events": [
            {
                "frame": e.frame,
                "time_sec": round(e.time_sec, 2),
                "aggressor_id": e.aggressor_id,
                "receiver_id": e.receiver_id,
                "contact_region": e.contact_region,
                "striking_limb": e.striking_limb,
                "probability": round(e.probability, 4),
                "velocity": round(e.velocity, 2),
                "impact_type": e.impact_type,
                "label": e.label,
            }
            for e in events
        ],
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results JSON : {out_path}")


if __name__ == "__main__":
    sys.exit(main())
