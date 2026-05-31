import sys as _sys, os as _os
#!/usr/bin/env python3
"""
Impact Detection Video Renderer
================================
Reads the source video and the impact detection results, then renders
an annotated output video with:
  - 2D skeleton overlay from pre-extracted keypoints
  - Impact flash effects when punches land
  - HUD showing running stats, event log, and per-punch labels
  - Colour-coded bounding boxes

Usage:
    python render_impact_video.py
    python render_impact_video.py --video path/to/video.mp4
"""
import os
import sys
import json
import argparse
import time

import cv2
import numpy as np
from collections import deque

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from keypoint_loader import KeypointLoader
from impact_detector import ImpactDetector, ImpactResult
from config import (
    KEYPOINTS_2D_PATH, KEYPOINTS_3D_PATH, ACTIONS_PATH,
    IMPACT_SCORE_THRESHOLD, OUTPUT_DIR,
    KP70_NOSE, KP70_LEFT_EYE, KP70_RIGHT_EYE,
    KP70_LEFT_EAR, KP70_RIGHT_EAR,
    KP70_LEFT_SHOULDER, KP70_RIGHT_SHOULDER,
    KP70_LEFT_ELBOW, KP70_RIGHT_ELBOW,
    KP70_LEFT_WRIST, KP70_RIGHT_WRIST,
    KP70_LEFT_HIP, KP70_RIGHT_HIP,
    KP70_LEFT_KNEE, KP70_RIGHT_KNEE,
    KP70_LEFT_ANKLE, KP70_RIGHT_ANKLE,
)

# Video source
DEFAULT_VIDEO = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)

# ── Skeleton connections (COCO body subset of the 70-joint model) ────────────
SKELETON_PAIRS = [
    (KP70_NOSE, KP70_LEFT_EYE), (KP70_NOSE, KP70_RIGHT_EYE),
    (KP70_LEFT_EYE, KP70_LEFT_EAR), (KP70_RIGHT_EYE, KP70_RIGHT_EAR),
    (KP70_LEFT_SHOULDER, KP70_RIGHT_SHOULDER),
    (KP70_LEFT_SHOULDER, KP70_LEFT_ELBOW), (KP70_LEFT_ELBOW, KP70_LEFT_WRIST),
    (KP70_RIGHT_SHOULDER, KP70_RIGHT_ELBOW), (KP70_RIGHT_ELBOW, KP70_RIGHT_WRIST),
    (KP70_LEFT_SHOULDER, KP70_LEFT_HIP), (KP70_RIGHT_SHOULDER, KP70_RIGHT_HIP),
    (KP70_LEFT_HIP, KP70_RIGHT_HIP),
    (KP70_LEFT_HIP, KP70_LEFT_KNEE), (KP70_LEFT_KNEE, KP70_LEFT_ANKLE),
    (KP70_RIGHT_HIP, KP70_RIGHT_KNEE), (KP70_RIGHT_KNEE, KP70_RIGHT_ANKLE),
]

WRIST_INDICES = {KP70_LEFT_WRIST, KP70_RIGHT_WRIST}

# Colours
COL_SKELETON  = (0, 255, 200)    # cyan-green
COL_WRIST     = (0, 180, 255)    # orange
COL_BBOX      = (255, 200, 0)    # yellow
COL_IMPACT    = (0, 50, 255)     # red
COL_LANDED    = (0, 230, 118)    # green
COL_MISSED    = (80, 80, 200)    # dull red
COL_HUD_BG    = (15, 15, 20)
COL_HUD_TITLE = (0, 215, 255)
COL_HUD_TEXT  = (200, 200, 200)

FLASH_DURATION = 12   # frames to show impact flash


def main():
    parser = argparse.ArgumentParser(description="Render impact-annotated video")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path")
    parser.add_argument("--kp2d", default=KEYPOINTS_2D_PATH)
    parser.add_argument("--kp3d", default=KEYPOINTS_3D_PATH)
    parser.add_argument("--actions", default=ACTIONS_PATH)
    parser.add_argument("--threshold", type=float, default=IMPACT_SCORE_THRESHOLD)
    parser.add_argument("--output", default=None, help="Output video path")
    args = parser.parse_args()

    print("")
    print("=" * 70)
    print("  SAM3D Impact Detection -- Video Renderer")
    print("=" * 70)

    t0 = time.time()

    # ── 1. Load keypoints and run impact detection ───────────────────────
    loader = KeypointLoader()
    frames_2d = loader.load_2d(args.kp2d)
    frames_3d = loader.load_3d(args.kp3d)
    actions = loader.load_actions(args.actions)

    detector = ImpactDetector(frames_2d, frames_3d, threshold=args.threshold)
    results = detector.analyze_all(actions)

    # Build frame-indexed lookup: frame_number -> list[ImpactResult]
    impact_map = _build_impact_map(results)

    # ── 2. Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{base}_impact_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    print(f"\n  Input       : {args.video}")
    print(f"  Output      : {out_path}")
    print(f"  Resolution  : {W}x{H} @ {fps:.1f} fps")
    print(f"  Frames      : {total_frames}")
    print(f"  Impacts     : {sum(1 for r in results if r.is_impact)} landed / "
          f"{sum(1 for r in results if not r.is_impact)} missed")
    print(f"\n  Rendering ...\n")

    # ── 3. Render loop ───────────────────────────────────────────────────
    flash_queue = deque()      # (expire_frame, contact_x, contact_y, label, prob)
    event_log = deque(maxlen=6)
    total_landed = 0
    total_missed = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = frame.copy()

        # ── Draw skeleton if we have keypoints for this frame ────────
        if frame_idx in frames_2d:
            fd = frames_2d[frame_idx]
            _draw_skeleton(canvas, fd.joints_2d, fd.bbox)

        # ── Check for impacts at this frame ──────────────────────────
        if frame_idx in impact_map:
            for r in impact_map[frame_idx]:
                if r.is_impact:
                    total_landed += 1
                    # Get wrist position for contact point
                    wrist_idx = KP70_LEFT_WRIST if r.striking_hand == "left" else KP70_RIGHT_WRIST
                    cx, cy = _get_wrist_pos(frames_2d, r.impact_frame, wrist_idx)
                    label = f"{r.action.replace('_', ' ').title()} -> {r.target}"
                    flash_queue.append((
                        frame_idx + FLASH_DURATION, cx, cy, label, r.impact_score
                    ))
                    event_log.append(
                        f"[{r.timestamp_seconds:5.1f}s] * {label}  score={r.impact_score:.2f}"
                    )
                else:
                    total_missed += 1
                    event_log.append(
                        f"[{r.timestamp_seconds:5.1f}s]   {r.action} missed  "
                        f"score={r.impact_score:.2f}"
                    )

        # ── Draw active impact flashes ───────────────────────────────
        _draw_flashes(canvas, flash_queue, frame_idx)

        # ── HUD overlay ──────────────────────────────────────────────
        _draw_hud(canvas, frame_idx, fps, total_landed, total_missed, event_log, W, H)

        # ── Bottom bar ───────────────────────────────────────────────
        _draw_bottom_bar(canvas, W, H)

        writer.write(canvas)
        frame_idx += 1

        # Progress
        if frame_idx % 500 == 0:
            pct = frame_idx / total_frames * 100
            print(f"    Frame {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"\n  Rendering complete!")
    print(f"  Output : {out_path}")
    print(f"  Time   : {elapsed:.1f}s")
    print("=" * 70 + "\n")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Drawing functions
# ═══════════════════════════════════════════════════════════════════════════

def _build_impact_map(results: list[ImpactResult]) -> dict[int, list[ImpactResult]]:
    """Map impact_frame -> list of ImpactResult for frame-level lookup."""
    m: dict[int, list[ImpactResult]] = {}
    for r in results:
        f = r.impact_frame if r.is_impact else r.action_frame
        m.setdefault(f, []).append(r)
    return m


def _get_wrist_pos(frames_2d, frame_num, wrist_idx):
    """Get wrist position for a given frame, falling back to nearby frames."""
    for offset in [0, -1, 1, -2, 2]:
        f = frame_num + offset
        if f in frames_2d:
            pos = frames_2d[f].joints_2d[wrist_idx]
            return int(pos[0]), int(pos[1])
    return -1, -1


def _draw_skeleton(canvas, joints_2d, bbox):
    """Draw COCO body skeleton + bounding box on frame."""
    # Bounding box
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_BBOX, 2)
    cv2.putText(canvas, "Fighter", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_BBOX, 2, cv2.LINE_AA)

    # Skeleton bones
    for a, b in SKELETON_PAIRS:
        if a >= len(joints_2d) or b >= len(joints_2d):
            continue
        pa = tuple(joints_2d[a].astype(int))
        pb = tuple(joints_2d[b].astype(int))
        if pa == (0, 0) or pb == (0, 0):
            continue
        cv2.line(canvas, pa, pb, COL_SKELETON, 2, cv2.LINE_AA)

    # Joint dots
    for i in range(min(17, len(joints_2d))):
        pt = tuple(joints_2d[i].astype(int))
        if pt == (0, 0):
            continue
        if i in WRIST_INDICES:
            cv2.circle(canvas, pt, 7, COL_WRIST, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 10, COL_WRIST, 2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 4, COL_SKELETON, -1, cv2.LINE_AA)


def _draw_flashes(canvas, flash_queue, frame_idx):
    """Render impact flash effects for active events."""
    still_active = deque()
    for expire, cx, cy, label, prob in flash_queue:
        if frame_idx <= expire:
            still_active.append((expire, cx, cy, label, prob))
            age = FLASH_DURATION - (expire - frame_idx)
            alpha_factor = max(0.0, 1.0 - age / FLASH_DURATION)

            if cx > 0 and cy > 0:
                # Expanding circle
                radius = 35 + age * 10
                thickness = max(1, 4 - age // 3)
                color = (0, int(80 * alpha_factor), int(255 * alpha_factor))
                cv2.circle(canvas, (cx, cy), radius, color, thickness, cv2.LINE_AA)

                # Inner glow
                if age < 4:
                    overlay = canvas.copy()
                    cv2.circle(overlay, (cx, cy), 20, (0, 60, 255), -1)
                    cv2.addWeighted(overlay, 0.3 * alpha_factor, canvas,
                                    1 - 0.3 * alpha_factor, 0, canvas)

                # Label
                cv2.putText(canvas, f"IMPACT! {prob:.0%}",
                            (cx - 80, cy - radius - 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75,
                            (0, int(100 * alpha_factor), int(255 * alpha_factor)),
                            2, cv2.LINE_AA)
                cv2.putText(canvas, label,
                            (cx - 80, cy - radius - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (int(200 * alpha_factor),) * 3,
                            1, cv2.LINE_AA)

            # Red border flash
            if age < 5:
                h, w = canvas.shape[:2]
                b = max(4, 14 - age * 2)
                ov = canvas.copy()
                for rect in [(0, 0, w, b), (0, h - b, w, h),
                             (0, 0, b, h), (w - b, 0, w, h)]:
                    cv2.rectangle(ov, (rect[0], rect[1]),
                                  (rect[2], rect[3]), COL_IMPACT, -1)
                cv2.addWeighted(ov, 0.5 * alpha_factor, canvas,
                                1 - 0.5 * alpha_factor, 0, canvas)

    flash_queue.clear()
    flash_queue.extend(still_active)


def _draw_hud(canvas, frame_idx, fps, landed, missed, event_log, W, H):
    """Draw heads-up display with stats and event log."""
    panel_w = 420
    panel_h = 210

    # Semi-transparent background
    ov = canvas.copy()
    cv2.rectangle(ov, (8, 8), (8 + panel_w, 8 + panel_h), COL_HUD_BG, -1)
    cv2.addWeighted(ov, 0.7, canvas, 0.3, 0, canvas)

    # Border
    cv2.rectangle(canvas, (8, 8), (8 + panel_w, 8 + panel_h), (40, 40, 50), 1)

    ts = frame_idx / max(fps, 1e-6)
    m, s = divmod(int(ts), 60)

    lines = [
        ("SAM3D Impact Detection", COL_HUD_TITLE, 0.6, 2),
        (f"Frame: {frame_idx:5d}   Time: {m:02d}:{s:02d}", COL_HUD_TEXT, 0.48, 1),
        (f"Landed: {landed}   Missed: {missed}   "
         f"Total: {landed + missed}", COL_HUD_TEXT, 0.48, 1),
        ("-" * 44, (80, 80, 80), 0.38, 1),
    ]

    for entry in list(event_log)[-4:]:
        col = COL_LANDED if "* " in entry else (150, 150, 150)
        lines.append((entry, col, 0.40, 1))

    if not event_log:
        lines.append(("  (no impacts yet)", (100, 100, 100), 0.40, 1))

    y = 32
    for text, color, scale, thick in lines:
        cv2.putText(canvas, text, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += 22


def _draw_bottom_bar(canvas, W, H):
    """Draw footer label."""
    label = "SAM3D Keypoint-Based Impact Detection | Pre-extracted 2D+3D Analysis"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, label, (W - tw - 12, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)


if __name__ == "__main__":
    sys.exit(main())
