"""
Main SAM3D Boxing Impact Detection Pipeline.

Architecture (Pi-HOC inspired, Sec 3 of paper):
  1. YOLOv8-pose  →  person detection + 17-keypoint pose estimation
  2. PersonTracker →  consistent IDs across frames
  3. SAM ViT-B     →  body segmentation (Pi-HOC contact decoder role)
  4. OpticalFlow   →  motion magnitude heatmap + wrist velocity estimation
  5. PairInteractionAnalyzer → Pi-HOC pair formation + contact feature scoring
  6. ImpactClassifier → deduplication + event logging
  7. Visualizer    →  annotated output video (real-player + 3D SMPL)

Fighter selection:
  - Detects top MAX_PERSON_POOL persons by area
  - Keeps exactly 2: the closest large pair within the central ring area
  - Registered pair is re-used across frames for tracking consistency
  - Referee and corner-men are automatically excluded
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os
import cv2
import numpy as np
from tqdm import tqdm

from models.detector import PersonDetector
from models.segmenter import SAMSegmenter
from models.tracker import PersonTracker
from impact_detection.pair_analyzer import PairInteractionAnalyzer
from impact_detection.impact_classifier import ImpactClassifier
from utils.visualization import Visualizer
from utils.flow import OpticalFlowEstimator
from utils.smpl_video_viz import SMPLVideoRenderer
from config import (
    YOLO_MODEL, SAM_CHECKPOINT, SAM_MODEL_TYPE,
    PROCESS_EVERY_N_FRAMES, MAX_PERSON_POOL,
    MIN_PERSON_AREA_RATIO, OUTPUT_DIR,
)


class ImpactDetectionPipeline:
    """End-to-end pipeline: video in → two annotated videos + impact event log out."""

    def __init__(self, use_sam: bool = True):
        print("\n[Pipeline] Initialising modules …")

        print("  Step 1 — YOLOv8-pose detector")
        self.detector = PersonDetector(YOLO_MODEL)

        print("  Step 2 — Person tracker")
        self.tracker = PersonTracker()

        print("  Step 3 — SAM ViT-B segmenter")
        self.segmenter = SAMSegmenter(SAM_CHECKPOINT, SAM_MODEL_TYPE) if use_sam else SAMSegmenter.__new__(SAMSegmenter)
        if not use_sam:
            self.segmenter.available = False

        print("  Step 4 — Optical flow estimator")
        self.flow_estimator = OpticalFlowEstimator()

        print("  Step 5 — Pi-HOC pair interaction analyzer")
        self.pair_analyzer = PairInteractionAnalyzer()

        print("  Step 6 — Impact classifier")
        self.classifier = ImpactClassifier()

        print("  Step 7 — Visualiser")
        self.viz = Visualizer()

        # Stable fighter-pair tracking across frames
        self._fighter_ids: set[int] = set()

        print("[Pipeline] Ready.\n")

    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        video_path: str,
        output_path: str | None = None,
        max_frames: int | None = None,
    ) -> dict:
        """
        Process video and write two output videos:
          *_real_impact.mp4  — annotated real-player video
          *_3d_impact.mp4    — 3D SMPL body visualization video
        Returns summary dict with impact counts and event list.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_real = output_path or os.path.join(OUTPUT_DIR, f"{base}_real_impact.mp4")
        output_3d   = os.path.join(OUTPUT_DIR, f"{base}_3d_impact.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS)
        W           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area  = W * H

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        out_fps = fps / PROCESS_EVERY_N_FRAMES
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer_real = cv2.VideoWriter(output_real, fourcc, out_fps, (W, H))
        writer_3d   = cv2.VideoWriter(output_3d,   fourcc, out_fps, (W, H))
        smpl_render = SMPLVideoRenderer(W, H)

        print(f"[Pipeline] Input     : {video_path}")
        print(f"[Pipeline] Real out  : {output_real}")
        print(f"[Pipeline] 3D out    : {output_3d}")
        print(f"[Pipeline] Video     : {W}×{H} @ {fps:.1f} fps  |  {total_frames} frames")
        print(f"[Pipeline] Stride    : every {PROCESS_EVERY_N_FRAMES} frames\n")

        frame_idx = 0
        processed = 0

        with tqdm(total=total_frames, unit="fr", desc="Processing") as pbar:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # ── Optical flow ──────────────────────────────────────────────
                flow_map = self.flow_estimator.update(frame)

                # ── Detection ─────────────────────────────────────────────────
                persons = self.detector.detect(frame)
                persons = self._filter_by_size(persons, frame_area)
                persons = self.tracker.update(persons)

                # Sort by area, keep a small pool then select fighters
                persons = sorted(
                    persons,
                    key=lambda p: (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]),
                    reverse=True,
                )[:MAX_PERSON_POOL]

                fighters = self._select_fighters(persons, W, frame)

                # ── SAM segmentation (fighters only) ──────────────────────────
                seg_masks: dict[int, np.ndarray] = {}
                if self.segmenter.available and fighters:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.segmenter.set_image(frame_rgb)
                    for p in fighters:
                        mask, score = self.segmenter.segment(p["bbox"])
                        if mask is not None and score > 0.45:
                            seg_masks[p["track_id"]] = mask
                    self.segmenter.reset()

                # ── Pi-HOC pair formation + contact analysis ───────────────────
                pairs    = self.pair_analyzer.form_pairs(fighters)
                contacts = self.pair_analyzer.analyze(fighters, pairs, frame_idx)

                # ── Impact classification ──────────────────────────────────────
                new_impacts = self.classifier.process(contacts, frame_idx, fps)

                # ── Real-player output ────────────────────────────────────────
                vis_real = self.viz.draw_frame(
                    frame, fighters, pairs, new_impacts,
                    seg_masks, flow_map, frame_idx, fps,
                    len(self.classifier.events),
                )
                writer_real.write(vis_real)

                # ── 3D SMPL output ────────────────────────────────────────────
                impact_for_3d = new_impacts[0] if new_impacts else None
                vis_3d = smpl_render.update(impact_for_3d, frame_idx, fps)
                writer_3d.write(vis_3d)

                pbar.update(PROCESS_EVERY_N_FRAMES)
                frame_idx += PROCESS_EVERY_N_FRAMES
                processed += 1

        cap.release()
        writer_real.release()
        writer_3d.release()

        summary = self.classifier.summary()
        summary["output_path"]      = output_real
        summary["output_3d_path"]   = output_3d
        summary["frames_processed"] = processed

        print(f"\n{'─'*60}")
        print(f"  Processing complete!")
        print(f"  Frames processed   : {processed}")
        print(f"  Total impacts      : {summary['total_impacts']}")
        print(f"    Head impacts     : {summary['head_impacts']}")
        print(f"    Torso impacts    : {summary['torso_impacts']}")
        print(f"  Real video         : {output_real}")
        print(f"  3D video           : {output_3d}")
        print(f"{'─'*60}\n")

        self._print_event_table(summary["events"])
        return summary

    # ─────────────────────────────────────────────────────────────────────────

    def _filter_by_size(self, persons: list[dict], frame_area: float) -> list[dict]:
        return [
            p for p in persons
            if ((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) / frame_area)
            >= MIN_PERSON_AREA_RATIO
        ]

    def _is_referee(self, person: dict, frame: np.ndarray) -> bool:
        """
        Referee heuristic: white/light shirt in upper-body crop.
        Referees universally wear white shirts in professional boxing.
        """
        x1, y1, x2, y2 = person["bbox"]
        h = y2 - y1
        crop_y2 = min(int(y1 + h * 0.45), frame.shape[0])
        crop = frame[y1:crop_y2, x1:x2]
        if crop.size == 0:
            return False
        # Check both mean brightness AND saturation — referee white shirt is bright+unsaturated
        import cv2 as _cv2
        hsv = _cv2.cvtColor(crop, _cv2.COLOR_BGR2HSV)
        bright = float(hsv[:, :, 2].mean()) > 165      # V channel high
        low_sat = float(hsv[:, :, 1].mean()) < 55      # S channel low (white/grey)
        return bright and low_sat

    def _select_fighters(self, persons: list[dict], W: int, frame: np.ndarray) -> list[dict]:
        """
        Return exactly the 2 fighters from the detected pool.

        Strategy:
          1. Filter out edge persons (ringside spectators/corner men).
          2. Tag referees by white-shirt brightness; exclude them from candidates.
          3. Re-use registered fighter IDs if still visible (sticky tracking).
          4. Otherwise pick the closest large pair from non-referee candidates.
        """
        if not persons:
            return []

        # Edge filter: ringside people cluster near frame edges
        # Use a narrow margin so fighters near the ropes aren't excluded
        margin_x = W * 0.05
        central = [
            p for p in persons
            if margin_x <= (p["bbox"][0]+p["bbox"][2])/2 <= W - margin_x
        ]
        pool = central if len(central) >= 2 else persons

        # Referee detection — always excluded; never used as fighters
        for p in pool:
            p["_ref"] = self._is_referee(p, frame)

        non_refs = [p for p in pool if not p["_ref"]]
        candidates = non_refs  # referees are never fallback

        # Re-use registered fighters if still visible among non-referees
        if self._fighter_ids:
            registered = [p for p in non_refs if p["track_id"] in self._fighter_ids]
            if len(registered) == 2:
                return registered
            if len(registered) == 1:
                others = [p for p in non_refs if p["track_id"] not in self._fighter_ids]
                if others:
                    partner = max(
                        others,
                        key=lambda p: (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]),
                    )
                    fighters = [registered[0], partner]
                    self._fighter_ids = {f["track_id"] for f in fighters}
                    return fighters

        # Fresh selection: among top-4 candidates, choose the closest large pair
        if len(candidates) < 2:
            return candidates  # not enough non-referees — skip frame's pair analysis

        top = sorted(
            candidates,
            key=lambda p: (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]),
            reverse=True,
        )[:4]

        if len(top) <= 2:
            fighters = top
        else:
            centers = [
                np.array([(p["bbox"][0]+p["bbox"][2])/2,
                           (p["bbox"][1]+p["bbox"][3])/2])
                for p in top
            ]
            areas = [
                (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1])
                for p in top
            ]
            best_score = float("inf")
            best_pair  = (0, 1)
            for i in range(len(top)):
                for j in range(i+1, len(top)):
                    dist  = float(np.linalg.norm(centers[i] - centers[j]))
                    score = dist / (areas[i] + areas[j] + 1e-6) * 1e6
                    if score < best_score:
                        best_score = score
                        best_pair  = (i, j)
            fighters = [top[best_pair[0]], top[best_pair[1]]]

        # Only register IDs confirmed as non-referees
        confirmed = [f for f in fighters if not f.get("_ref")]
        if len(confirmed) == 2:
            self._fighter_ids = {f["track_id"] for f in confirmed}
        return fighters

    def _print_event_table(self, events):
        if not events:
            print("  No impact events logged.")
            return
        print(f"  {'#':>3}  {'Time':>6}  {'Label':<30}  {'Prob':>5}  {'Vel':>6}")
        print("  " + "─" * 62)
        for i, ev in enumerate(events, 1):
            m, s = divmod(int(ev.time_sec), 60)
            print(f"  {i:>3}  {m:02d}:{s:02d}  {ev.label:<30}  {ev.probability:>4.0%}  {ev.velocity:>5.1f}")
