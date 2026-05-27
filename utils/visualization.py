"""
Visualization module — draws detection results, SAM masks, skeleton, and impact overlays.
"""
import cv2
import numpy as np
from collections import deque

from config import (
    VIZ_COLORS, KP, IMPACT_FLASH_DURATION, STRIKING_KPS_INFO,
)
from impact_detection.impact_classifier import ImpactEvent
from utils.body_contact_viz import BodyContactDiagram, BW, BH

# COCO skeleton connections
_SKELETON = [
    (KP["nose"], KP["left_eye"]), (KP["nose"], KP["right_eye"]),
    (KP["left_eye"], KP["left_ear"]), (KP["right_eye"], KP["right_ear"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]), (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]), (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]), (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]), (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]), (KP["right_knee"], KP["right_ankle"]),
]

_PERSON_COLORS = [
    VIZ_COLORS["fighter_1"],
    VIZ_COLORS["fighter_2"],
    VIZ_COLORS["referee"],
]

# Body-diagram panel sizing
_DIAG_SCALE  = 2          # upscale factor for legibility
_DIAG_W      = BW * _DIAG_SCALE
_DIAG_H      = BH * _DIAG_SCALE
_DIAG_GAP    = 4          # px gap between the two diagrams
_DIAG_MARGIN = 10         # px from frame edge


class Visualizer:
    def __init__(self):
        self._flash_queue: deque[tuple[int, list, str, float]] = deque()
        self._impact_history: list[str] = []
        self._body_diag = BodyContactDiagram()

        # Per-fighter cumulative contact history
        # track_id → {hit_regions: list, strike_regions: list, hit_counts: dict}
        self._fighter_hits:    dict[int, list[str]] = {}
        self._fighter_strikes: dict[int, list[str]] = {}
        self._fighter_hcounts: dict[int, dict[str, int]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def draw_frame(
        self,
        frame:            np.ndarray,
        persons:          list[dict],
        pairs:            list[tuple],
        new_impacts:      list[ImpactEvent],
        seg_masks:        dict[int, np.ndarray],
        flow_map:         np.ndarray | None,
        frame_idx:        int,
        fps:              float,
        all_impacts_count: int,
    ) -> np.ndarray:
        canvas = frame.copy()

        # ── Flow heatmap overlay ──────────────────────────────────────────────
        if flow_map is not None:
            self._draw_flow_heatmap(canvas, flow_map, alpha=0.18)

        # ── SAM segmentation masks ────────────────────────────────────────────
        for i, (_, mask) in enumerate(seg_masks.items()):
            color = _PERSON_COLORS[min(i, len(_PERSON_COLORS) - 1)]
            self._draw_mask(canvas, mask, color, alpha=0.30)

        # ── Per-person: bbox + skeleton ───────────────────────────────────────
        for i, person in enumerate(persons):
            color = _PERSON_COLORS[min(i, len(_PERSON_COLORS) - 1)]
            self._draw_bbox(canvas, person["bbox"], color,
                            person.get("track_id", -1), person["conf"])
            if person["keypoints"] is not None:
                self._draw_skeleton(canvas, person["keypoints"], color)

        # ── Interaction pair lines ────────────────────────────────────────────
        for idx_a, idx_b in pairs:
            if idx_a < len(persons) and idx_b < len(persons):
                ca = _box_center(persons[idx_a]["bbox"])
                cb = _box_center(persons[idx_b]["bbox"])
                cv2.line(canvas, tuple(ca.astype(int)), tuple(cb.astype(int)),
                         (180, 180, 60), 1, cv2.LINE_AA)

        # ── Register new impacts ──────────────────────────────────────────────
        for ev in new_impacts:
            if ev.contact_point:
                self._flash_queue.append((
                    frame_idx + IMPACT_FLASH_DURATION,
                    ev.contact_point,
                    ev.label,
                    ev.probability,
                ))
            self._impact_history.append(
                f"[{ev.time_sec:5.1f}s] {ev.label}  p={ev.probability:.2f}"
            )
            if len(self._impact_history) > 8:
                self._impact_history.pop(0)

            # Accumulate body-diagram state
            hit_r, strike_r = self._body_diag.regions_from_event(
                ev.contact_region, ev.striking_limb)
            rid = ev.receiver_id
            aid = ev.aggressor_id
            if rid not in self._fighter_hits:
                self._fighter_hits[rid]    = []
                self._fighter_hcounts[rid] = {}
            self._fighter_hits[rid].extend(hit_r)
            for r in hit_r:
                self._fighter_hcounts[rid][r] = self._fighter_hcounts[rid].get(r, 0) + 1
            if aid not in self._fighter_strikes:
                self._fighter_strikes[aid] = []
            self._fighter_strikes[aid].extend(strike_r)

        # ── Draw active impact flashes ────────────────────────────────────────
        self._draw_active_flashes(canvas, frame_idx)

        # ── HUD text overlay ──────────────────────────────────────────────────
        self._draw_hud(canvas, frame_idx, fps, all_impacts_count)

        # ── Body contact diagram panel (bottom-right) ─────────────────────────
        self._draw_contact_panel(canvas)

        return canvas

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_contact_panel(self, canvas: np.ndarray):
        """
        Render per-fighter cumulative body contact diagrams in the bottom-right corner.
        Shows up to two fighters side by side.
        """
        h, w = canvas.shape[:2]

        # Collect up to 2 fighter IDs (union of hitters and receivers)
        fids = list(dict.fromkeys(
            list(self._fighter_hits.keys()) + list(self._fighter_strikes.keys())
        ))[:2]

        if not fids:
            return

        panels = []
        for fid in fids:
            img = self._body_diag.draw(
                hit_regions    = list(set(self._fighter_hits.get(fid, []))),
                strike_regions = list(set(self._fighter_strikes.get(fid, []))),
                hit_counts     = self._fighter_hcounts.get(fid, {}),
                label          = f"ID{fid}",
            )
            img = cv2.resize(img, (_DIAG_W, _DIAG_H), interpolation=cv2.INTER_NEAREST)
            panels.append(img)

        gap  = np.full((_DIAG_H, _DIAG_GAP, 3), 14, dtype=np.uint8)
        combined = panels[0] if len(panels) == 1 else np.hstack([panels[0], gap, panels[1]])

        cw, ch = combined.shape[1], combined.shape[0]
        x0 = w - cw - _DIAG_MARGIN
        y0 = h - ch - _DIAG_MARGIN

        # Dark backing
        backing_pad = 6
        ov = canvas.copy()
        cv2.rectangle(ov,
                       (x0 - backing_pad, y0 - 18),
                       (x0 + cw + backing_pad, y0 + ch + backing_pad),
                       (14, 14, 14), -1)
        cv2.addWeighted(ov, 0.75, canvas, 0.25, 0, canvas)

        # Label
        cv2.putText(canvas, "Body Contact Map",
                    (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

        # Paste diagrams
        canvas[y0:y0 + ch, x0:x0 + cw] = combined

    def _draw_bbox(self, canvas, bbox, color, track_id, conf):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"ID:{track_id}  {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_skeleton(self, canvas, keypoints_data, color):
        pts   = keypoints_data["points"]
        confs = keypoints_data["confidence"]

        for a, b in _SKELETON:
            if confs[a] < 0.25 or confs[b] < 0.25:
                continue
            pa, pb = tuple(pts[a].astype(int)), tuple(pts[b].astype(int))
            if pa == (0, 0) or pb == (0, 0):
                continue
            cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)

        for i, (pt, c) in enumerate(zip(pts, confs)):
            if c < 0.25 or np.allclose(pt, 0):
                continue
            r = 5 if i in [kp for kp, _ in STRIKING_KPS_INFO] else 3
            cv2.circle(canvas, tuple(pt.astype(int)), r, color, -1)

        for kp_idx, _ in STRIKING_KPS_INFO[:2]:
            if confs[kp_idx] >= 0.3 and not np.allclose(pts[kp_idx], 0):
                cv2.circle(canvas, tuple(pts[kp_idx].astype(int)), 10,
                           VIZ_COLORS["contact_pt"], 2, cv2.LINE_AA)

    def _draw_mask(self, canvas, mask, color, alpha=0.35):
        overlay = canvas.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        mask_u8 = mask.astype(np.uint8) * 255
        ctrs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, ctrs, -1, color, 2)

    def _draw_flow_heatmap(self, canvas, flow_map, alpha=0.18):
        norm = cv2.normalize(flow_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.addWeighted(heat, alpha, canvas, 1 - alpha, 0, canvas)

    def _draw_active_flashes(self, canvas, frame_idx):
        still_active = deque()
        for expire, pt, label, prob in self._flash_queue:
            if frame_idx <= expire:
                still_active.append((expire, pt, label, prob))
                age          = IMPACT_FLASH_DURATION - (expire - frame_idx)
                radius       = 30 + age * 8
                thickness    = max(1, 3 - age)
                alpha_factor = 1.0 - age / IMPACT_FLASH_DURATION
                flash_color  = (0, int(60 * alpha_factor), int(255 * alpha_factor))
                if pt:
                    cv2.circle(canvas, tuple(pt), radius, flash_color, thickness, cv2.LINE_AA)
                    cv2.putText(canvas, f"IMPACT! {prob:.0%}",
                                (pt[0] - 60, pt[1] - radius - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 50, 255), 2, cv2.LINE_AA)
                h, w  = canvas.shape[:2]
                b     = 12
                ov    = canvas.copy()
                for rect in [(0,0,w,b),(0,h-b,w,h),(0,0,b,h),(w-b,0,w,h)]:
                    cv2.rectangle(ov, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,200), -1)
                cv2.addWeighted(ov, 0.5*alpha_factor, canvas, 1-0.5*alpha_factor, 0, canvas)
        self._flash_queue = still_active

    def _draw_hud(self, canvas, frame_idx, fps, total_impacts):
        h, w    = canvas.shape[:2]
        panel_w = 380
        panel_h = 200
        ov = canvas.copy()
        cv2.rectangle(ov, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.65, canvas, 0.35, 0, canvas)

        ts       = frame_idx / max(fps, 1e-6)
        m, s     = divmod(int(ts), 60)
        time_str = f"{m:02d}:{s:02d}"

        lines = [
            "SAM3D Boxing Impact Detection",
            f"Frame: {frame_idx:5d}   Time: {time_str}",
            f"Impacts Detected: {total_impacts}",
            "-" * 38,
        ] + (self._impact_history[-4:] if self._impact_history else ["  (no impacts yet)"])

        for i, line in enumerate(lines):
            y     = 30 + i * 22
            color = (0, 200, 255) if i == 0 else (200, 200, 200)
            size  = 0.58 if i == 0 else 0.48
            cv2.putText(canvas, line, (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, size, color, 1 + (i == 0), cv2.LINE_AA)

        label = "Pi-HOC Contact Estimation | SAM Segmentation"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(canvas, label, (w - tw - 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)


def _box_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
