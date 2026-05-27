#!/usr/bin/env python3
"""
JSON Keypoint-Driven Boxing Impact Detection Pipeline  (GPU + SAM Hybrid)
=========================================================================
Combines Pi-HOC's SAM-based contact surface detection with physics-based
temporal gates.  All heavy computation runs on GPU (CUDA).

Keypoints sourced exclusively from pre-extracted JSON files:
  - 2d_points.json    (2 persons, 4575 frames, 70 joints, original resolution)
  - 3d_points.json    (world-space + body-centred 3D coords)
  - full_results.json (138 ASFormer action windows)

Six-gate scoring (sum = 1.0):
  Gate 1 (0.40): SAM mask overlap  -- wrist pixel vs opponent SAM body mask [Pi-HOC]
  Gate 2 (0.20): Wrist deceleration -- sharp velocity drop at contact  [physics/GPU]
  Gate 3 (0.15): 3D depth proximity -- world-space backup gate         [Pi-HOC]
  Gate 4 (0.12): 3D jerk            -- sudden force spike              [physics/GPU]
  Gate 5 (0.08): Action confidence  -- ASFormer score x speed x power  [learned]
  Gate 6 (0.05): Arm extension      -- near-full extension confirms punch [physics/GPU]

SAM runs on GPU with the opponent's bounding box taken from JSON keypoints.
Wrist positions used as point-probes into the SAM mask also come from JSON.

Usage:
    python pipeline_json.py
    python pipeline_json.py --threshold 0.42
    python pipeline_json.py --no-video   # analysis only
"""
import os
import sys
import json
import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.smpl_video_viz import SMPLVideoRenderer
from config import (
    KEYPOINTS_2D_PATH, KEYPOINTS_3D_PATH, ACTIONS_PATH,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    OUTPUT_DIR, PROCESS_EVERY_N_FRAMES,
    IMPACT_SCORE_THRESHOLD, WINDOW_PAD_FRAMES,
    ACTION_HAND_MAP, ACTION_ARM_CHAIN,
    KP70_LEFT_WRIST, KP70_RIGHT_WRIST,
    PROXIMITY_THRESHOLD, IMPACT_COOLDOWN_FRAMES, GLOBAL_COOLDOWN_FRAMES,
)

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# ─── Source video ─────────────────────────────────────────────────────────────
DEFAULT_VIDEO = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)

# ─── Gate weights — standard (A/B/C, must sum to 1.0) ────────────────────────
W_SAM      = 0.40
W_DECEL    = 0.20
W_PROX_3D  = 0.15
W_JERK     = 0.12
W_CONF     = 0.08
W_EXT      = 0.05

# ─── Gate weights — enhanced (D, 7 gates, must sum to 1.0) ───────────────────
W_SAM_E      = 0.33   # SAM mask overlap  (elbow probe added)
W_DECEL_E    = 0.18   # wrist deceleration
W_RECEIVER_E = 0.14   # receiver head reaction  ← new gate
W_PROX_3D_E  = 0.12   # 3D proximity
W_JERK_E     = 0.10   # 3D jerk
W_CONF_E     = 0.08   # action confidence
W_EXT_E      = 0.05   # arm extension

# ─── Gate weights — dual-body correlation (G, 7 gates, must sum to 1.0) ──────
W_SAM_G      = 0.28   # SAM mask overlap
W_DECEL_G    = 0.16   # wrist deceleration
W_DUAL_G     = 0.18   # dual-body correlation gate  ← new gate
W_PROX_3D_G  = 0.12   # 3D proximity
W_JERK_G     = 0.10   # 3D jerk
W_CONF_G     = 0.08   # action confidence
W_EXT_G      = 0.08   # arm extension

# ─── Proximity thresholds ─────────────────────────────────────────────────────
PROX_2D_MAX = PROXIMITY_THRESHOLD   # pixels — distance at which SAM edge score = 0
PROX_3D_MAX = 0.60                  # metres in shared_space_coords

# ─── Keypoint groups ──────────────────────────────────────────────────────────
HEAD_KPS  = [0, 1, 2, 3, 4]        # nose, eyes, ears
TORSO_KPS = [5, 6, 11, 12]         # shoulders, hips

# ─── Skeleton connections (COCO-17 subset of 70-joint model) ─────────────────
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
WRIST_INDICES = {9, 10}

# ─── Colours ─────────────────────────────────────────────────────────────────
COL_F0     = (0, 255, 180)
COL_F1     = (255, 140, 0)
COL_WRIST  = (0, 180, 255)
COL_IMPACT = (0, 50, 255)
COL_HUD_BG = (15, 15, 20)
COL_TITLE  = (0, 215, 255)
COL_TEXT   = (200, 200, 200)

FLASH_FRAMES = 15


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Frame2D:
    frame: int
    bbox: np.ndarray        # (4,) x1 y1 x2 y2, original resolution
    joints: np.ndarray      # (70, 2) original resolution


@dataclass
class Frame3D:
    frame: int
    shared: np.ndarray      # (70, 3) world-space coordinates
    normalized: np.ndarray  # (70, 3) body-centred


@dataclass
class ActionEvent:
    fighter_type: str
    action: str
    confidence: float
    frame: int
    window_start: int
    window_end: int
    timestamp_seconds: float
    target: str
    speed_kmh: float
    power_watts: float


@dataclass
class ImpactResult:
    """Result of six-gate hybrid impact analysis for one action event."""
    action: str
    fighter_type: str
    action_frame: int
    timestamp_seconds: float
    striking_hand: str
    target: str

    is_impact: bool = False
    impact_score: float = 0.0
    impact_frame: int = -1
    contact_region: str = "torso"
    contact_point: list = field(default_factory=list)
    striker_id: int = 0
    receiver_id: int = 1

    # Gate sub-scores
    sam_score: float = 0.0
    decel_score: float = 0.0
    jerk_score: float = 0.0
    ext_score: float = 0.0
    prox_3d_score: float = 0.0
    conf_score: float = 0.0
    receiver_reaction_score: float = 0.0   # D: head-snap of receiver at impact
    dual_corr_score: float = 0.0           # G: striker_decel ↔ receiver_head_accel

    velocity_profile: list = field(default_factory=list)

    @property
    def label(self) -> str:
        zone = self.contact_region.replace("_", " ").title()
        return (f"F{self.striker_id}→F{self.receiver_id}  "
                f"{self.action.replace('_', ' ').title()}  [{zone}]")

    @property
    def striking_limb(self) -> str:
        return "left_jab" if self.striking_hand == "left" else "right_cross"

    @property
    def aggressor_id(self) -> int:
        return self.striker_id


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_two_person_2d(path: str) -> dict[int, dict[int, Frame2D]]:
    print(f"[Load] 2D keypoints: {path}")
    with open(path) as f:
        raw = json.load(f)

    persons: dict[int, dict[int, Frame2D]] = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str)
        persons[pid] = {}
        for e in entries:
            frame_num = e["frame"]
            dims = e.get("frame_dims", {})
            sx = dims.get("original_width",  1920) / dims.get("resized_width",  640)
            sy = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            joints = np.array(e["joints_2d"], dtype=np.float64)
            joints[:, 0] *= sx
            joints[:, 1] *= sy
            bbox = np.array(e["bbox"], dtype=np.float64)
            persons[pid][frame_num] = Frame2D(frame=frame_num, bbox=bbox, joints=joints)

    for pid, frames in persons.items():
        if frames:
            mn, mx = min(frames), max(frames)
            print(f"  Person {pid}: {len(frames)} frames  [{mn}-{mx}]")
    return persons


def load_two_person_3d(path: str) -> dict[int, dict[int, Frame3D]]:
    print(f"[Load] 3D keypoints: {path}")
    with open(path) as f:
        raw = json.load(f)

    persons: dict[int, dict[int, Frame3D]] = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str)
        persons[pid] = {}
        for e in entries:
            frame_num = e["frame"]
            shared     = np.array(e["shared_space_coords"], dtype=np.float64)
            normalized = np.array(e["normalized_coords"],   dtype=np.float64)
            persons[pid][frame_num] = Frame3D(
                frame=frame_num, shared=shared, normalized=normalized
            )

    for pid, frames in persons.items():
        if frames:
            print(f"  Person {pid}: {len(frames)} 3D frames")
    return persons


def load_actions(path: str) -> list[ActionEvent]:
    print(f"[Load] Actions: {path}")
    with open(path) as f:
        raw = json.load(f)

    events = []
    for e in raw.get("actions", []):
        speed = e.get("speed_estimation",  {}).get("estimated_speed_kmh",   0.0)
        power = e.get("power_estimation",  {}).get("estimated_power_watts",  0.0)
        events.append(ActionEvent(
            fighter_type      = e.get("fighter_type", "fighter_0"),
            action            = e["action"],
            confidence        = e["confidence"],
            frame             = e["frame"],
            window_start      = e["window_start"],
            window_end        = e["window_end"],
            timestamp_seconds = e.get("timestamp_seconds", 0.0),
            target            = e.get("target", "Head"),
            speed_kmh         = speed,
            power_watts       = power,
        ))

    events.sort(key=lambda a: a.frame)
    print(f"  {len(events)} action events  "
          f"(types: {sorted(set(a.action for a in events))})")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# SAM proximity gate  (Pi-HOC concept on GPU)
# ─────────────────────────────────────────────────────────────────────────────

class SamProximityGate:
    """
    Pi-HOC-inspired gate: use SAM to segment the opponent's body, then probe
    whether the striker's wrist pixel (from JSON) falls inside that mask.

    The bounding-box SAM prompt comes from JSON 2D keypoints, NOT from any
    live detector.  SAM runs on GPU.
    """

    def __init__(self, checkpoint: str, model_type: str, device: str):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment_anything not installed. Run:  pip install segment-anything"
            )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

        print(f"[SAM] Loading {model_type} on {device} ...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(torch.device(device))
        self.predictor = SamPredictor(sam)
        self.device = device
        print(f"[SAM] Ready on {device}.")

    def score(
        self,
        frame_bgr: np.ndarray,
        receiver_bbox: np.ndarray,              # (4,) from JSON — SAM prompt
        striker_wrist: np.ndarray,              # (2,) from JSON — primary probe
        receiver_joints: np.ndarray,            # (70,2) for region classification
        striker_elbow: Optional[np.ndarray] = None,  # (2,) D: secondary probe
    ) -> tuple[float, str, list]:
        """Returns (sam_overlap_score, contact_region, contact_point).
        When striker_elbow is provided (Approach D), probes both wrist and
        elbow into the mask and takes the maximum — catches hooks/uppercuts."""
        if frame_bgr is None:
            return 0.0, "torso", []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)

        x1, y1, x2, y2 = [float(v) for v in receiver_bbox]
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
        mask = masks[0]

        H, W = mask.shape
        wx = int(np.clip(striker_wrist[0], 0, W - 1))
        wy = int(np.clip(striker_wrist[1], 0, H - 1))

        wrist_score = self._overlap_score(mask, wx, wy)

        # Elbow probe (Approach D) — take max of wrist and elbow scores
        if striker_elbow is not None and not np.allclose(striker_elbow, 0):
            ex = int(np.clip(striker_elbow[0], 0, W - 1))
            ey = int(np.clip(striker_elbow[1], 0, H - 1))
            elbow_score = self._overlap_score(mask, ex, ey)
            sam_score   = max(wrist_score, elbow_score)
            best_pt     = [ex, ey] if elbow_score > wrist_score else [wx, wy]
        else:
            sam_score = wrist_score
            best_pt   = [wx, wy]

        contact_region = self._classify_region(striker_wrist, receiver_joints)
        return sam_score, contact_region, best_pt

    def segment_both_fighters(
        self,
        frame_bgr: np.ndarray,
        bbox0: np.ndarray,
        bbox1: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segment both fighters in one set_image call (shares the encoder pass).
        Returns two boolean masks (H,W) or None if bbox is missing.
        Used for heatmap rendering (Approach C).
        """
        if frame_bgr is None:
            return None, None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)

        def _predict(bbox):
            if bbox is None:
                return None
            x1, y1, x2, y2 = [float(v) for v in bbox]
            box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
            return masks[0]

        return _predict(bbox0), _predict(bbox1)

    def _overlap_score(self, mask: np.ndarray, wx: int, wy: int) -> float:
        """1.0 if wrist is inside mask; falls off with distance to mask edge."""
        if mask[wy, wx]:
            return 1.0
        # Distance transform on inverted mask: exact Euclidean dist to nearest mask pixel
        inv = (~mask).astype(np.uint8)
        dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        return max(0.0, 1.0 - float(dist_map[wy, wx]) / PROX_2D_MAX)

    def _classify_region(
        self, wrist_xy: np.ndarray, receiver_joints: np.ndarray
    ) -> str:
        """6-zone classification: head_left/right, upper_torso_left/right, lower_torso_left/right."""
        SHOULDER_KPS = [5, 6]
        HIP_KPS      = [11, 12]

        def _valid(k):
            return k < len(receiver_joints) and not np.allclose(receiver_joints[k], 0)

        wx, wy = float(wrist_xy[0]), float(wrist_xy[1])

        # Body geometry
        shoulders = [receiver_joints[k] for k in SHOULDER_KPS if _valid(k)]
        hips      = [receiver_joints[k] for k in HIP_KPS      if _valid(k)]

        shoulder_y = float(np.mean([s[1] for s in shoulders])) if shoulders else None
        hip_y      = float(np.mean([h[1] for h in hips]))      if hips      else None
        center_x   = float(np.mean(
            [s[0] for s in shoulders] + [h[0] for h in hips]
        )) if (shoulders or hips) else None

        side = ("left" if (center_x is not None and wx < center_x) else "right")

        # Head check — closest head keypoint within 100px
        head_dists = [
            float(np.linalg.norm(wrist_xy - receiver_joints[k]))
            for k in HEAD_KPS if _valid(k)
        ]
        if head_dists and min(head_dists) < 100:
            return f"head_{side}"

        # Torso zone using shoulder / hip Y bounds
        if shoulder_y is not None and hip_y is not None:
            mid_y = (shoulder_y + hip_y) / 2
            if wy < mid_y:
                return f"upper_torso_{side}"
            else:
                return f"lower_torso_{side}"
        elif shoulder_y is not None:
            return f"upper_torso_{side}"

        return "torso"


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid impact detector
# ─────────────────────────────────────────────────────────────────────────────

class JsonImpactDetector:
    """
    Six-gate impact detector driven entirely by pre-extracted JSON keypoints.
    Gate 1 (SAM) uses GPU inference.  Gates 2-4 use CUDA tensors.
    """

    def __init__(
        self,
        frames_2d: dict[int, dict[int, Frame2D]],
        frames_3d: dict[int, dict[int, Frame3D]],
        threshold: float = IMPACT_SCORE_THRESHOLD,
        device: str = DEVICE,
        sam_checkpoint: str = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
        sam_window: int = 1,        # A: 5 — scan ±(sam_window//2) frames for max SAM score
        gate_soft: bool = False,    # B: True — physics gates can rescue when SAM fails
        sam_gate: Optional["SamProximityGate"] = None,  # reuse existing model
        use_elbow_probe: bool = False,    # D: also probe elbow into SAM mask
        use_receiver_gate: bool = False,  # D: score receiver head reaction at impact
        temporal_ema_alpha: float = 0.0,  # E: Gaussian EMA smoothing on SAM window scores
        use_dual_correlation: bool = False, # G: striker_decel <-> receiver_head_accel
        learned_model = None,             # F: fitted sklearn classifier (predict_proba)
    ):
        self.f2d                   = frames_2d
        self.f3d                   = frames_3d
        self.threshold             = threshold
        self.device                = device
        self.sam_window            = sam_window
        self.gate_soft             = gate_soft
        self.use_elbow_probe       = use_elbow_probe
        self.use_receiver_gate     = use_receiver_gate
        self.temporal_ema_alpha    = temporal_ema_alpha
        self.use_dual_correlation  = use_dual_correlation
        self.learned_model         = learned_model

        if sam_gate is not None:
            self.sam_gate = sam_gate
        else:
            self.sam_gate: Optional[SamProximityGate] = None
            try:
                self.sam_gate = SamProximityGate(sam_checkpoint, sam_model_type, device)
            except (ImportError, FileNotFoundError) as e:
                print(f"[SAM] Unavailable ({e}) — falling back to keypoint proximity.")

    # ── Public API ─────────────────────────────────────────────────────────

    def find_candidate_frames(self, actions: list[ActionEvent]) -> set[int]:
        """Phase 1: keypoint-only pass to determine best impact frame per action.
        Expands by ±(sam_window//2) so all SAM-probe frames are pre-extracted."""
        half   = self.sam_window // 2
        frames = set()
        for action in actions:
            sid  = 0 if action.fighter_type == "fighter_0" else 1
            rid  = 1 - sid
            widx = ACTION_HAND_MAP.get(action.action, KP70_RIGHT_WRIST)
            ws   = max(0, action.window_start - WINDOW_PAD_FRAMES)
            we   = action.window_end + WINDOW_PAD_FRAMES
            bf   = self._find_best_impact_frame(sid, rid, widx, ws, we, action.frame)
            for offset in range(-half, half + 1):
                frames.add(bf + offset)
        return frames

    def analyze_all(
        self,
        actions: list[ActionEvent],
        video_frames: Optional[dict[int, np.ndarray]] = None,
    ) -> list[ImpactResult]:
        return [
            self.analyze_action(a, video_frames=video_frames)
            for a in tqdm(actions, desc="Detecting impacts", unit="action")
        ]

    def analyze_action(
        self,
        action: ActionEvent,
        video_frames: Optional[dict[int, np.ndarray]] = None,
    ) -> ImpactResult:
        striker_id  = 0 if action.fighter_type == "fighter_0" else 1
        receiver_id = 1 - striker_id

        wrist_idx    = ACTION_HAND_MAP.get(action.action, KP70_RIGHT_WRIST)
        hand_name    = "left" if wrist_idx == KP70_LEFT_WRIST else "right"
        elbow_idx, shoulder_idx = ACTION_ARM_CHAIN[wrist_idx]

        ws = max(0, action.window_start - WINDOW_PAD_FRAMES)
        we = action.window_end + WINDOW_PAD_FRAMES

        result = ImpactResult(
            action            = action.action,
            fighter_type      = action.fighter_type,
            action_frame      = action.frame,
            timestamp_seconds = action.timestamp_seconds,
            striking_hand     = hand_name,
            target            = action.target,
            striker_id        = striker_id,
            receiver_id       = receiver_id,
        )

        # ── Best impact frame (max 2D proximity within window) ────────────
        best_impact_frame = self._find_best_impact_frame(
            striker_id, receiver_id, wrist_idx, ws, we, action.frame
        )

        # ── Gate 1: SAM mask overlap (Pi-HOC concept, GPU) ────────────────
        sam_score, contact_region, contact_pt = self._gate_sam(
            striker_id, receiver_id, wrist_idx, elbow_idx,
            best_impact_frame, video_frames,
        )

        # ── Gate 2: Wrist deceleration (GPU tensor) ───────────────────────
        decel_score, vel_profile = self._gate_decel(striker_id, wrist_idx, ws, we)

        # ── Gate 3: 3D jerk magnitude (GPU tensor) ────────────────────────
        jerk_score, _ = self._gate_jerk(striker_id, wrist_idx, ws, we)

        # ── Gate 4: Arm extension (GPU tensor) ───────────────────────────
        ext_score = self._gate_extension(
            striker_id, wrist_idx, elbow_idx, shoulder_idx,
            action.window_start, action.window_end, ws, we,
        )

        # ── Gate 5: 3D depth proximity (GPU tensor, Pi-HOC depth gate) ───
        prox_3d = self._gate_3d_proximity(
            striker_id, receiver_id, wrist_idx, best_impact_frame, ws, we
        )

        # ── Gate 6: Action confidence ─────────────────────────────────────
        conf_score = self._gate_confidence(action)

        # ── Gate 7: Receiver head reaction (Approach D / F only) ─────────
        receiver_score = 0.0
        if self.use_receiver_gate:
            receiver_score = self._gate_receiver_reaction(
                receiver_id, best_impact_frame, ws, we
            )

        # ── Gate 8: Dual-body correlation (Approach G only) ───────────────
        dual_corr_score = 0.0
        if self.use_dual_correlation:
            dual_corr_score = self._gate_dual_body_correlation(
                striker_id, receiver_id, wrist_idx, best_impact_frame, ws, we
            )

        # ── Scoring ───────────────────────────────────────────────────────
        zero_condition = sam_score < 0.05 and prox_3d < 0.05
        if self.gate_soft:
            zero_condition = zero_condition and decel_score < 0.3

        if zero_condition:
            score = 0.0
        elif self.learned_model is not None:
            # Approach F: learned LogisticRegression on 7 gate features
            feats = [[sam_score, decel_score, jerk_score, ext_score,
                      prox_3d, conf_score, receiver_score]]
            score = float(self.learned_model.predict_proba(feats)[0, 1])
        elif self.use_dual_correlation:
            # Approach G: dual-body correlation replaces receiver gate
            score = (
                W_SAM_G      * sam_score
                + W_DECEL_G    * decel_score
                + W_DUAL_G     * dual_corr_score
                + W_PROX_3D_G  * prox_3d
                + W_JERK_G     * jerk_score
                + W_CONF_G     * conf_score
                + W_EXT_G      * ext_score
            )
        elif self.use_receiver_gate:
            # Approach D / E-enhanced: 7-gate with receiver reaction
            score = (
                W_SAM_E      * sam_score
                + W_DECEL_E    * decel_score
                + W_RECEIVER_E * receiver_score
                + W_PROX_3D_E  * prox_3d
                + W_JERK_E     * jerk_score
                + W_CONF_E     * conf_score
                + W_EXT_E      * ext_score
            )
        else:
            # Approaches A / B / C / E: standard 6-gate
            score = (
                W_SAM     * sam_score
                + W_DECEL   * decel_score
                + W_PROX_3D * prox_3d
                + W_JERK    * jerk_score
                + W_EXT     * ext_score
                + W_CONF    * conf_score
            )

        result.is_impact               = score >= self.threshold
        result.impact_score            = round(score, 4)
        result.impact_frame            = best_impact_frame
        result.contact_region          = contact_region
        result.contact_point           = contact_pt
        result.sam_score               = round(sam_score,       4)
        result.decel_score             = round(decel_score,     4)
        result.jerk_score              = round(jerk_score,      4)
        result.ext_score               = round(ext_score,       4)
        result.prox_3d_score           = round(prox_3d,         4)
        result.conf_score              = round(conf_score,      4)
        result.receiver_reaction_score = round(receiver_score,  4)
        result.dual_corr_score         = round(dual_corr_score, 4)
        result.velocity_profile        = vel_profile

        return result

    # ── Gate implementations ──────────────────────────────────────────────

    def _find_best_impact_frame(
        self, striker_id: int, receiver_id: int, wrist_idx: int,
        ws: int, we: int, fallback: int,
    ) -> int:
        """Frame within [ws,we] where wrist-to-opponent keypoint distance is minimum."""
        f2d_s = self.f2d.get(striker_id, {})
        f2d_r = self.f2d.get(receiver_id, {})
        best_frame = fallback
        best_prox  = -1.0
        target_kps = HEAD_KPS + TORSO_KPS

        for f in range(ws, we + 1):
            fs = f2d_s.get(f)
            fr = f2d_r.get(f)
            if fs is None or fr is None:
                continue
            wrist_2d = fs.joints[wrist_idx]
            if np.allclose(wrist_2d, 0):
                continue
            dists = [
                float(np.linalg.norm(wrist_2d - fr.joints[k]))
                for k in target_kps
                if k < len(fr.joints) and not np.allclose(fr.joints[k], 0)
            ]
            if not dists:
                continue
            prox = max(0.0, 1.0 - min(dists) / PROX_2D_MAX)
            if prox > best_prox:
                best_prox  = prox
                best_frame = f
        return best_frame

    def _gate_sam(
        self,
        striker_id: int,
        receiver_id: int,
        wrist_idx: int,
        elbow_idx: int,
        impact_frame: int,
        video_frames: Optional[dict[int, np.ndarray]],
    ) -> tuple[float, str, list]:
        """Pi-HOC gate: SAM body mask overlap.
        Approach A: scans ±(sam_window//2) frames, returns the max score frame.
        Approach D: also probes striker elbow into the mask."""
        half = self.sam_window // 2
        offsets = range(-half, half + 1)

        frame_scores: list[tuple[float, str, list]] = []

        for offset in offsets:
            f = impact_frame + offset
            fd_s = self.f2d.get(striker_id,  {}).get(f)
            fd_r = self.f2d.get(receiver_id, {}).get(f)
            if fd_s is None or fd_r is None:
                frame_scores.append((0.0, "torso", []))
                continue

            frame_bgr = video_frames.get(f) if video_frames else None

            if self.sam_gate is not None and frame_bgr is not None:
                elbow_xy = fd_s.joints[elbow_idx] if self.use_elbow_probe else None
                sc, region, pt = self.sam_gate.score(
                    frame_bgr, fd_r.bbox, fd_s.joints[wrist_idx],
                    fd_r.joints, striker_elbow=elbow_xy,
                )
            else:
                sc, region, pt = self._kp_proximity_fallback(fd_s, fd_r, wrist_idx)

            frame_scores.append((sc, region, pt))

        if not frame_scores:
            return 0.0, "torso", []

        # Approach E: EMA-weighted temporal smoothing (Gaussian from center frame)
        if self.temporal_ema_alpha > 0.0:
            n = len(frame_scores)
            center = (n - 1) / 2.0
            ema_w = [self.temporal_ema_alpha ** abs(i - center) for i in range(n)]
            total_w = sum(ema_w)
            ema_score = sum(w * s[0] for w, s in zip(ema_w, frame_scores)) / total_w
            # Region/pt from the highest-scoring individual frame
            best_idx = max(range(n), key=lambda i: frame_scores[i][0])
            return ema_score, frame_scores[best_idx][1], frame_scores[best_idx][2]

        # Default: take the best single-frame score
        best = max(frame_scores, key=lambda x: x[0])
        return best[0], best[1], best[2]

    def _kp_proximity_fallback(
        self, fd_s: Frame2D, fd_r: Frame2D, wrist_idx: int
    ) -> tuple[float, str, list]:
        wrist_2d = fd_s.joints[wrist_idx]
        head_dists  = [float(np.linalg.norm(wrist_2d - fd_r.joints[k]))
                       for k in HEAD_KPS  if not np.allclose(fd_r.joints[k], 0)]
        torso_dists = [float(np.linalg.norm(wrist_2d - fd_r.joints[k]))
                       for k in TORSO_KPS if not np.allclose(fd_r.joints[k], 0)]
        if not head_dists and not torso_dists:
            return 0.0, "torso", []
        min_head  = min(head_dists)  if head_dists  else float("inf")
        min_torso = min(torso_dists) if torso_dists else float("inf")
        min_dist  = min(min_head, min_torso)
        # Use SAM gate's 6-zone classifier if possible, else coarse fallback
        if self.sam_gate is not None:
            region = self.sam_gate._classify_region(wrist_2d, fd_r.joints)
        else:
            region = "head" if min_head < min_torso else "torso"
        score = max(0.0, 1.0 - min_dist / PROX_2D_MAX)
        return score, region, wrist_2d.astype(int).tolist()

    def _gate_decel(
        self, pid: int, wrist_idx: int, ws: int, we: int
    ) -> tuple[float, list]:
        """Wrist deceleration gate — runs on GPU tensor."""
        frames_3d = self.f3d.get(pid, {})
        frange = sorted(f for f in frames_3d if ws <= f <= we)
        if len(frange) < 3:
            return 0.0, []

        pos = torch.tensor(
            np.array([frames_3d[f].normalized[wrist_idx] for f in frange]),
            dtype=torch.float32, device=self.device,
        )
        dt = torch.tensor(
            np.diff(frange).astype(np.float32), device=self.device
        ).clamp(min=1.0)

        velocities = torch.norm(torch.diff(pos, dim=0) / dt.unsqueeze(1), dim=1)

        vel_profile = [
            {"frame": int(frange[i + 1]), "velocity": float(velocities[i])}
            for i in range(len(velocities))
        ]

        if len(velocities) < 2:
            return 0.0, vel_profile

        peak_vel = float(velocities.max())
        if peak_vel < 1e-6:
            return 0.0, vel_profile

        dt2   = dt[1:] if len(dt) > 1 else dt
        accel = torch.diff(velocities)
        if len(accel) == 0:
            return 0.0, vel_profile
        dt2 = dt2[: len(accel)]
        accel = accel / dt2

        max_decel = float(torch.clamp(-accel.min(), min=0.0))
        score = min(1.0, (max_decel / peak_vel) / 0.6)
        return score, vel_profile

    def _gate_jerk(
        self, pid: int, wrist_idx: int, ws: int, we: int
    ) -> tuple[float, int]:
        """3D jerk magnitude gate — runs on GPU tensor."""
        frames_3d = self.f3d.get(pid, {})
        frange = sorted(f for f in frames_3d if ws <= f <= we)
        if len(frange) < 5:
            return 0.0, -1

        pos = torch.tensor(
            np.array([frames_3d[f].normalized[wrist_idx] for f in frange]),
            dtype=torch.float32, device=self.device,
        )
        dt = torch.tensor(
            np.diff(frange).astype(np.float32), device=self.device
        ).clamp(min=1.0)

        vel_vecs  = torch.diff(pos, dim=0) / dt.unsqueeze(1)
        if len(vel_vecs) < 2:
            return 0.0, -1
        accel_vecs = torch.diff(vel_vecs, dim=0) / dt[1:].unsqueeze(1)
        if len(accel_vecs) < 2:
            return 0.0, -1
        jerk_vecs  = torch.diff(accel_vecs, dim=0) / dt[2:].unsqueeze(1)

        jerk_mags = torch.norm(jerk_vecs, dim=1)
        if len(jerk_mags) == 0:
            return 0.0, -1

        peak_idx     = int(jerk_mags.argmax())
        frame_offset = peak_idx + 3
        impact_frame = int(frange[frame_offset]) if frame_offset < len(frange) else int(frange[-1])

        score = min(1.0, float(jerk_mags[peak_idx]) / 0.03)
        return score, impact_frame

    def _gate_extension(
        self,
        pid: int, wrist_idx: int, elbow_idx: int, shoulder_idx: int,
        action_start: int, action_end: int, ws: int, we: int,
    ) -> float:
        """Arm extension gate — GPU tensor over the action window frames."""
        frames_2d = self.f2d.get(pid, {})
        window_frames = sorted(
            f for f in frames_2d
            if ws <= f <= we and action_start <= f <= action_end
        )
        if not window_frames:
            return 0.0

        # Stack joints for vectorised computation
        joints_list = [frames_2d[f].joints for f in window_frames]
        joints_t = torch.tensor(
            np.stack(joints_list), dtype=torch.float32, device=self.device
        )  # (N, 70, 2)

        w  = joints_t[:, wrist_idx]
        e  = joints_t[:, elbow_idx]
        s  = joints_t[:, shoulder_idx]

        # Mask out zero-keypoints
        valid = ~(
            (w.abs().sum(1) < 1e-3) |
            (e.abs().sum(1) < 1e-3) |
            (s.abs().sum(1) < 1e-3)
        )
        if not valid.any():
            return 0.0

        w, e, s = w[valid], e[valid], s[valid]
        full_arm = torch.norm(w - e, dim=1) + torch.norm(e - s, dim=1)
        valid2   = full_arm > 1e-3
        if not valid2.any():
            return 0.0

        ext  = torch.norm(w[valid2] - s[valid2], dim=1) / full_arm[valid2]
        peak = float(ext.max())
        return float(torch.clamp(
            torch.tensor((peak - 0.60) / 0.35), min=0.0, max=1.0
        ))

    def _gate_3d_proximity(
        self,
        striker_id: int, receiver_id: int, wrist_idx: int,
        impact_frame: int, ws: int, we: int,
    ) -> float:
        """3D world-space proximity gate — GPU tensor."""
        f3d_s = self.f3d.get(striker_id,  {})
        f3d_r = self.f3d.get(receiver_id, {})

        check_start = max(ws,  impact_frame - 8)
        check_end   = min(we,  impact_frame + 12)

        best = 0.0
        target_kps = HEAD_KPS + TORSO_KPS

        frames = [f for f in range(check_start, check_end + 1)
                  if f in f3d_s and f in f3d_r]
        if not frames:
            return 0.0

        wrists   = torch.tensor(
            np.array([f3d_s[f].shared[wrist_idx] for f in frames]),
            dtype=torch.float32, device=self.device
        )  # (N, 3)
        opp_kps  = torch.tensor(
            np.array([[f3d_r[f].shared[k] for k in target_kps] for f in frames]),
            dtype=torch.float32, device=self.device
        )  # (N, K, 3)

        # Filter zero keypoints per frame
        for i in range(len(frames)):
            valid_kps = opp_kps[i][opp_kps[i].abs().sum(1) > 1e-3]
            if len(valid_kps) == 0:
                continue
            dists  = torch.norm(wrists[i].unsqueeze(0) - valid_kps, dim=1)
            min_d  = float(dists.min())
            score  = max(0.0, 1.0 - min_d / PROX_3D_MAX)
            if score > best:
                best = score

        return best

    def _gate_confidence(self, action: ActionEvent) -> float:
        conf_part  = min(1.0, action.confidence)
        power_part = min(1.0, action.power_watts / 3000.0)
        speed_part = min(1.0, action.speed_kmh   / 25.0)
        return 0.50 * conf_part + 0.25 * power_part + 0.25 * speed_part

    def _gate_receiver_reaction(
        self, receiver_id: int, impact_frame: int, ws: int, we: int
    ) -> float:
        """Approach D: measure sudden head-snap of the receiver at impact.
        When a punch lands the receiver's nose/eye keypoints jerk.
        Score = max acceleration of the mean head position normalised to 15 px/frame²."""
        HEAD_PROBE = [0, 1, 2]   # nose, left_eye, right_eye
        f2d_r = self.f2d.get(receiver_id, {})

        check_start = max(ws, impact_frame - 6)
        check_end   = min(we, impact_frame + 6)

        head_pos, head_frames = [], []
        for f in range(check_start, check_end + 1):
            fd = f2d_r.get(f)
            if fd is None:
                continue
            valid = [fd.joints[k] for k in HEAD_PROBE
                     if k < len(fd.joints) and not np.allclose(fd.joints[k], 0)]
            if valid:
                head_pos.append(np.mean(valid, axis=0))
                head_frames.append(f)

        if len(head_pos) < 3:
            return 0.0

        pos_t = torch.tensor(np.array(head_pos), dtype=torch.float32, device=self.device)
        dt    = torch.tensor(np.diff(head_frames, prepend=head_frames[0]-1).astype(np.float32),
                             device=self.device).clamp(min=1.0)[1:]
        vel   = torch.norm(torch.diff(pos_t, dim=0) / dt.unsqueeze(1), dim=1)

        if len(vel) < 2:
            return 0.0

        accel = torch.diff(vel)
        max_snap = float(accel.abs().max())
        return min(1.0, max_snap / 15.0)

    def _gate_dual_body_correlation(
        self, striker_id: int, receiver_id: int, wrist_idx: int,
        impact_frame: int, ws: int, we: int,
    ) -> float:
        """Approach G: Pearson correlation between striker wrist deceleration and
        receiver head acceleration over ±5 frames. Physics ground truth for contact:
        the striker slows down at the same moment the receiver's head speeds up."""
        WINDOW = 5
        f3d_s = self.f3d.get(striker_id,  {})
        f3d_r = self.f3d.get(receiver_id, {})

        check_start = max(ws, impact_frame - WINDOW)
        check_end   = min(we, impact_frame + WINDOW)

        frames = sorted(
            f for f in range(check_start, check_end + 1)
            if f in f3d_s and f in f3d_r
        )
        if len(frames) < 4:
            return 0.0

        # Striker wrist 3D positions (normalised body-space)
        s_wrist = torch.tensor(
            np.array([f3d_s[f].normalized[wrist_idx] for f in frames]),
            dtype=torch.float32, device=self.device,
        )
        # Receiver head centroid (mean of HEAD_KPS in 3D)
        n_kps = f3d_r[frames[0]].normalized.shape[0]
        valid_head = [k for k in HEAD_KPS if k < n_kps]
        r_head = torch.tensor(
            np.array([f3d_r[f].normalized[valid_head].mean(axis=0) for f in frames]),
            dtype=torch.float32, device=self.device,
        )

        dt = torch.tensor(
            np.diff(frames).astype(np.float32), device=self.device
        ).clamp(min=1.0)

        s_speed = torch.norm(torch.diff(s_wrist, dim=0) / dt.unsqueeze(1), dim=1)
        r_speed = torch.norm(torch.diff(r_head,  dim=0) / dt.unsqueeze(1), dim=1)

        if len(s_speed) < 3:
            return 0.0

        # Striker deceleration (>0 when punch lands and wrist slows)
        s_decel = -torch.diff(s_speed)
        # Receiver head acceleration (>0 when head snaps on impact)
        r_accel =  torch.diff(r_speed)

        min_len = min(len(s_decel), len(r_accel))
        if min_len < 2:
            return 0.0

        s_d = s_decel[:min_len]
        r_a = r_accel[:min_len]

        s_mu = s_d.mean()
        r_mu = r_a.mean()
        num  = ((s_d - s_mu) * (r_a - r_mu)).sum()
        den  = torch.sqrt(((s_d - s_mu) ** 2).sum() * ((r_a - r_mu) ** 2).sum())

        if float(den) < 1e-8:
            return 0.0

        corr = float(num / den)
        # Map Pearson r from [-1,1] to [0,1]; positive = contact signal
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown filter
# ─────────────────────────────────────────────────────────────────────────────

class CooldownFilter:
    """Prevents same pair from firing within cooldown window of each other.
    Approach D uses reduced cooldowns to allow combos through."""

    def __init__(
        self,
        pair_cooldown: int = IMPACT_COOLDOWN_FRAMES,
        global_cooldown: int = GLOBAL_COOLDOWN_FRAMES,
    ):
        self._pair_last: dict[tuple, int] = {}
        self._global_last: int = -999999
        self._pair_cooldown   = pair_cooldown
        self._global_cooldown = global_cooldown

    def accept(self, result: ImpactResult) -> bool:
        if not result.is_impact:
            return False
        frame = result.impact_frame
        if frame - self._global_last < self._global_cooldown:
            return False
        key = (result.striker_id, result.receiver_id)
        rev = (result.receiver_id, result.striker_id)
        last = max(self._pair_last.get(key, -999999),
                   self._pair_last.get(rev, -999999))
        if frame - last < self._pair_cooldown:
            return False
        self._pair_last[key] = frame
        self._global_last    = frame
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Video frame extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_candidate_frames(
    video_path: str, frame_indices: set[int]
) -> dict[int, np.ndarray]:
    """
    Extract specific frames from the video by direct seek.
    Returns {frame_idx: BGR ndarray}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for frame extraction: {video_path}")
        return {}

    frames: dict[int, np.ndarray] = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame

    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Video rendering
# ─────────────────────────────────────────────────────────────────────────────

def build_impact_map(results: list[ImpactResult]) -> dict[int, list[ImpactResult]]:
    m: dict[int, list[ImpactResult]] = {}
    for r in results:
        if r.is_impact:
            m.setdefault(r.impact_frame, []).append(r)
    return m


def run_video(
    video_path: str,
    frames_2d: dict[int, dict[int, Frame2D]],
    impact_map: dict[int, list[ImpactResult]],
    output_real: str,
    output_3d: str,
    fps_video: float,
    stride: int = PROCESS_EVERY_N_FRAMES,
    max_frames: Optional[int] = None,
    render_sam_masks: bool = False,
    sam_gate: Optional[SamProximityGate] = None,
    approach_label: str = "",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or fps_video
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    out_fps = fps / stride
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer_real = cv2.VideoWriter(output_real, fourcc, out_fps, (W, H))
    writer_3d   = cv2.VideoWriter(output_3d,   fourcc, out_fps, (W, H))
    smpl        = SMPLVideoRenderer(W, H)

    print(f"\n  Input       : {video_path}")
    print(f"  Real output : {output_real}")
    print(f"  3D output   : {output_3d}")
    print(f"  Resolution  : {W}x{H} @ {fps:.1f} fps")
    print(f"  Total       : {total} frames  (stride={stride})")
    if render_sam_masks:
        print("  SAM heatmap : ON (live mask per frame)")
    print()

    flash_queue   = deque()
    event_log     = deque(maxlen=6)
    total_impacts = 0

    frame_idx = 0
    with tqdm(total=total, unit="fr", desc=f"Rendering {approach_label}") as pbar:
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                canvas = frame.copy()

                # ── SAM heatmap overlay (Approach C) ─────────────────────
                if render_sam_masks and sam_gate is not None:
                    fd0 = frames_2d.get(0, {}).get(frame_idx)
                    fd1 = frames_2d.get(1, {}).get(frame_idx)
                    bbox0 = fd0.bbox if fd0 is not None else None
                    bbox1 = fd1.bbox if fd1 is not None else None
                    mask0, mask1 = sam_gate.segment_both_fighters(frame, bbox0, bbox1)
                    if mask0 is not None:
                        _draw_sam_heatmap(canvas, mask0)
                    if mask1 is not None:
                        _draw_sam_heatmap(canvas, mask1)

                # ── Skeleton overlays ─────────────────────────────────────
                for pid, color in [(0, COL_F0), (1, COL_F1)]:
                    fd = frames_2d.get(pid, {}).get(frame_idx)
                    if fd is not None:
                        _draw_person(canvas, fd, color, f"Fighter {pid}",
                                     label_prefix=f"F{pid} ")

                # ── Impact flashes ────────────────────────────────────────
                FIGHTER_COLORS = {0: COL_F0, 1: COL_F1}
                for check_f in range(frame_idx, min(frame_idx + stride, total)):
                    if check_f in impact_map:
                        for r in impact_map[check_f]:
                            total_impacts += 1
                            striker_color = FIGHTER_COLORS.get(r.striker_id, COL_IMPACT)
                            flash_queue.append((
                                frame_idx + FLASH_FRAMES,
                                r.contact_point[0] if r.contact_point else -1,
                                r.contact_point[1] if r.contact_point else -1,
                                r.label,
                                r.impact_score,
                                striker_color,
                            ))
                            m_, s_ = divmod(int(r.timestamp_seconds), 60)
                            event_log.append(
                                f"[{m_:02d}:{s_:02d}] * {r.label}  {r.impact_score:.0%}"
                            )

                _draw_flashes(canvas, flash_queue, frame_idx)
                _draw_hud(canvas, frame_idx, fps, total_impacts, event_log,
                          approach_label=approach_label)
                _draw_footer(canvas, W, H, approach_label=approach_label)
                writer_real.write(canvas)

                impact_evt = None
                for check_f in range(frame_idx, min(frame_idx + stride, total)):
                    if check_f in impact_map and impact_map[check_f]:
                        impact_evt = impact_map[check_f][0]
                        break
                vis_3d = smpl.update(impact_evt, frame_idx, fps)
                writer_3d.write(vis_3d)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer_real.release()
    writer_3d.release()
    print(f"\n  Done.  Impacts shown: {total_impacts}")


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_sam_heatmap(canvas, mask: np.ndarray, alpha: float = 0.45):
    """SAM mask → distance-transform heatmap aura (Approach C)."""
    mask_u8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = (dist / dist.max() * 255).astype(np.uint8)
    else:
        dist_norm = mask_u8 * 255
    heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    kernel  = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel)
    overlay = canvas.copy()
    overlay[dilated > 0] = heatmap[dilated > 0]
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def _draw_person(canvas, fd: Frame2D, color, name: str, label_prefix: str = ""):
    x1, y1, x2, y2 = [int(v) for v in fd.bbox]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, label_prefix + name,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    joints = fd.joints
    for a, b in SKELETON_PAIRS:
        if a >= len(joints) or b >= len(joints):
            continue
        pa = tuple(joints[a].astype(int))
        pb = tuple(joints[b].astype(int))
        if pa == (0, 0) or pb == (0, 0):
            continue
        cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)

    for i in range(min(17, len(joints))):
        pt = tuple(joints[i].astype(int))
        if pt == (0, 0):
            continue
        if i in WRIST_INDICES:
            cv2.circle(canvas, pt, 7,  COL_WRIST, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 10, COL_WRIST,  2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)


def _draw_flashes(canvas, flash_queue, frame_idx):
    still_active = deque()
    for item in flash_queue:
        expire, cx, cy, label, prob = item[:5]
        color = item[5] if len(item) > 5 else COL_IMPACT   # striker color or default red

        if frame_idx <= expire:
            still_active.append(item)
            age   = FLASH_FRAMES - (expire - frame_idx)
            alpha = max(0.0, 1.0 - age / FLASH_FRAMES)
            b0, g0, r0 = color  # BGR

            if cx > 0 and cy > 0:
                radius = 30 + age * 8
                flash_col = (int(b0 * alpha), int(g0 * alpha), int(r0 * alpha))
                cv2.circle(canvas, (cx, cy), radius, flash_col,
                           max(1, 4 - age // 3), cv2.LINE_AA)
                if age < 4:
                    ov = canvas.copy()
                    cv2.circle(ov, (cx, cy), 20, color, -1)
                    cv2.addWeighted(ov, 0.3 * alpha, canvas, 1 - 0.3 * alpha, 0, canvas)
                cv2.putText(canvas, f"IMPACT! {prob:.0%}",
                            (cx - 80, cy - radius - 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, flash_col, 2, cv2.LINE_AA)
                cv2.putText(canvas, label,
                            (cx - 80, cy - radius - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (int(200 * alpha),) * 3, 1, cv2.LINE_AA)

            if age < 5:
                h, w = canvas.shape[:2]
                b  = max(4, 14 - age * 2)
                ov = canvas.copy()
                for rx0, ry0, rx1, ry1 in [
                    (0, 0, w, b), (0, h - b, w, h),
                    (0, 0, b, h), (w - b, 0, w, h)
                ]:
                    cv2.rectangle(ov, (rx0, ry0), (rx1, ry1), color, -1)
                cv2.addWeighted(ov, 0.5 * alpha, canvas, 1 - 0.5 * alpha, 0, canvas)

    flash_queue.clear()
    flash_queue.extend(still_active)


def _draw_hud(canvas, frame_idx, fps, total_impacts, event_log,
              _W=None, _H=None, approach_label: str = ""):
    panel_w, panel_h = 460, 250
    ov = canvas.copy()
    cv2.rectangle(ov, (8, 8), (8 + panel_w, 8 + panel_h), COL_HUD_BG, -1)
    cv2.addWeighted(ov, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (8, 8), (8 + panel_w, 8 + panel_h), (40, 40, 55), 1)

    ts = frame_idx / max(fps, 1e-6)
    m_, s_ = divmod(int(ts), 60)

    lines = [
        ("SAM3D  |  Hybrid SAM+Physics Impact Detection", COL_TITLE, 0.52, 2),
        (f"Frame: {frame_idx:5d}   Time: {m_:02d}:{s_:02d}", COL_TEXT, 0.46, 1),
        (f"Impacts Detected: {total_impacts}", COL_TEXT, 0.46, 1),
    ]
    if approach_label:
        lines.append((f"Approach: {approach_label}", (0, 200, 255), 0.44, 1))
    lines.append(("-" * 50, (70, 70, 70), 0.35, 1))
    for entry in list(event_log)[-5:]:
        col = (0, 230, 100) if "* " in entry else (140, 140, 140)
        lines.append((entry, col, 0.37, 1))
    if not event_log:
        lines.append(("  (monitoring ...)", (90, 90, 90), 0.37, 1))

    y = 30
    for text, color, scale, thick in lines:
        cv2.putText(canvas, text, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += 22


def _draw_footer(canvas, W, H, approach_label: str = ""):
    suffix = f"  [{approach_label}]" if approach_label else ""
    label = f"SAM mask overlap (Pi-HOC) + Physics gates (GPU)  |  JSON keypoints{suffix}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(canvas, label, (W - tw - 12, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list[ImpactResult]):
    landed = [r for r in results if r.is_impact]
    print(f"\n  {'#':>3}  {'Time':>6}  {'Type':<20}  {'Score':>6}  "
          f"{'SAM':>6}  {'3D':>6}  {'Decel':>6}  {'Jerk':>6}  {'Result':<10}")
    print("  " + "-" * 96)
    for i, r in enumerate(results, 1):
        m_, s_ = divmod(int(r.timestamp_seconds), 60)
        tag = "* LANDED" if r.is_impact else "  missed"
        print(
            f"  {i:>3}  {m_:02d}:{s_:02d}  {r.action:<20}  {r.impact_score:>6.3f}  "
            f"{r.sam_score:>6.3f}  {r.prox_3d_score:>6.3f}  "
            f"{r.decel_score:>6.3f}  {r.jerk_score:>6.3f}  {tag}"
        )
    print(f"\n  Total: {len(results)} actions -> {len(landed)} landed  "
          f"({len(landed)/max(len(results),1):.0%} landing rate)")


def save_impact_heatmap(results: list[ImpactResult], out_path: str):
    """Render a body silhouette with impact zone dots — between-round summary."""
    W, H = 520, 820
    canvas = np.full((H, W, 3), 18, dtype=np.uint8)

    cx = W // 2

    # ── Body silhouette ───────────────────────────────────────────────────────
    sil = (70, 70, 70)
    cv2.ellipse(canvas, (cx, 110), (52, 62), 0, 0, 360, sil, 2)           # head
    cv2.line(canvas, (cx, 172), (cx, 210), sil, 3)                         # neck
    cv2.line(canvas, (cx - 110, 215), (cx + 110, 215), sil, 3)            # shoulders
    cv2.rectangle(canvas, (cx - 95, 215), (cx + 95, 390), sil, 2)         # upper torso
    cv2.line(canvas, (cx, 390), (cx, 395), sil, 2)                         # waist join
    cv2.rectangle(canvas, (cx - 80, 395), (cx + 80, 530), sil, 2)         # lower torso
    cv2.line(canvas, (cx - 110, 215), (cx - 135, 390), sil, 2)            # left arm
    cv2.line(canvas, (cx + 110, 215), (cx + 135, 390), sil, 2)            # right arm
    cv2.line(canvas, (cx - 40, 530), (cx - 50, 710), sil, 2)              # left leg
    cv2.line(canvas, (cx + 40, 530), (cx + 50, 710), sil, 2)              # right leg

    # Zone → pixel position on silhouette
    ZONE_POS = {
        "head_left":         (cx - 28, 100),
        "head_right":        (cx + 28, 100),
        "head":              (cx,      100),
        "upper_torso_left":  (cx - 52, 285),
        "upper_torso_right": (cx + 52, 285),
        "upper_torso":       (cx,      285),
        "lower_torso_left":  (cx - 42, 455),
        "lower_torso_right": (cx + 42, 455),
        "lower_torso":       (cx,      455),
        "torso":             (cx,      370),
    }

    # Accumulate per-zone counts split by receiver
    from collections import defaultdict
    zone_data: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    landed = [r for r in results if r.is_impact]
    for r in landed:
        zone_data[r.contact_region][r.receiver_id] += 1

    if zone_data:
        max_count = max(sum(v.values()) for v in zone_data.values())
    else:
        max_count = 1

    FIGHTER_COLORS = {0: COL_F0, 1: COL_F1}

    for zone, fighter_counts in zone_data.items():
        pos = ZONE_POS.get(zone, (cx, 370))
        total = sum(fighter_counts.values())
        intensity = total / max_count
        radius = max(14, int(14 + intensity * 36))

        # Dominant receiver determines fill colour
        dominant = max(fighter_counts, key=fighter_counts.get)
        fill = FIGHTER_COLORS.get(dominant, (180, 180, 180))

        cv2.circle(canvas, pos, radius, fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, pos, radius, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(total),
                    (pos[0] - (7 if total < 10 else 10), pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 2, cv2.LINE_AA)

    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(canvas, "Impact Zone Map", (cx - 105, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Total Impacts: {len(landed)}", (cx - 72, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Fighter colour legend ─────────────────────────────────────────────────
    cv2.circle(canvas, (30, H - 110), 8, COL_F0, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Fighter 0 (receiver)", (44, H - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(canvas, (30, H - 85), 8, COL_F1, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Fighter 1 (receiver)", (44, H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Zone breakdown table ──────────────────────────────────────────────────
    y = H - 55
    cv2.putText(canvas, "Zone breakdown:", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
    y += 18
    for zone, fc in sorted(zone_data.items(), key=lambda x: -sum(x[1].values())):
        parts = "  ".join(f"F{fid}:{cnt}" for fid, cnt in sorted(fc.items()))
        txt = f"  {zone.replace('_',' ').title()}: {parts}"
        cv2.putText(canvas, txt, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (160, 160, 160), 1, cv2.LINE_AA)
        y += 16
        if y > H - 8:
            break

    cv2.imwrite(out_path, canvas)
    print(f"  Impact map : {out_path}")


def save_results_json(results: list[ImpactResult], out_path: str):
    data = {
        "summary": {
            "total_actions": len(results),
            "total_landed":  sum(1 for r in results if r.is_impact),
            "total_missed":  sum(1 for r in results if not r.is_impact),
            "head_impacts":  sum(1 for r in results if r.is_impact and r.contact_region == "head"),
            "torso_impacts": sum(1 for r in results if r.is_impact and r.contact_region == "torso"),
        },
        "events": [
            {
                "action":            r.action,
                "fighter_type":      r.fighter_type,
                "action_frame":      r.action_frame,
                "impact_frame":      r.impact_frame,
                "timestamp_seconds": r.timestamp_seconds,
                "striking_hand":     r.striking_hand,
                "is_impact":         r.is_impact,
                "impact_score":      r.impact_score,
                "contact_region":    r.contact_region,
                "contact_point":     r.contact_point,
                "gate_scores": {
                    "sam_mask_overlap":      r.sam_score,
                    "deceleration":          r.decel_score,
                    "jerk":                  r.jerk_score,
                    "extension":             r.ext_score,
                    "proximity_3d":          r.prox_3d_score,
                    "confidence":            r.conf_score,
                    "receiver_reaction":     r.receiver_reaction_score,
                    "dual_body_correlation": r.dual_corr_score,
                },
            }
            for r in results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results JSON: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Approach F helper — train logistic regression on gate scores
# ─────────────────────────────────────────────────────────────────────────────

def _train_learned_scorer(results_json_path: str):
    """Load gate scores + labels from a prior approach's JSON, fit a
    sklearn LogisticRegression, and return the trained model.
    Returns None if the file is missing or has too few positives."""
    from sklearn.linear_model import LogisticRegression

    if not os.path.isfile(results_json_path):
        print(f"  [F] Training JSON not found: {results_json_path}")
        return None

    with open(results_json_path) as f:
        data = json.load(f)

    X, y = [], []
    for r in data.get("events", []):
        gs = r.get("gate_scores", {})
        features = [
            gs.get("sam_mask_overlap",      0.0),
            gs.get("deceleration",          0.0),
            gs.get("jerk",                  0.0),
            gs.get("extension",             0.0),
            gs.get("proximity_3d",          0.0),
            gs.get("confidence",            0.0),
            gs.get("receiver_reaction",     0.0),
        ]
        X.append(features)
        y.append(1 if r.get("is_impact", False) else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 3 or n_neg < 3:
        print(f"  [F] Not enough labelled samples (pos={n_pos} neg={n_neg}) — skipping.")
        return None

    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, random_state=42
    )
    clf.fit(X, y)

    coef = clf.coef_[0]
    feat_names = ["SAM", "Decel", "Jerk", "Ext", "Prox3D", "Conf", "RecvReact"]
    coef_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(feat_names, coef))
    print(f"  [F] Trained on {len(X)} samples  (pos={n_pos} neg={n_neg})")
    print(f"  [F] Coefs: {coef_str}")
    print(f"  [F] Intercept: {clf.intercept_[0]:.3f}")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _run_approach(
    label: str,
    tag: str,
    frames_2d, frames_3d, actions,
    video_path: str,
    output_dir: str,
    shared_sam_gate: SamProximityGate,
    base: str,
    stride: int,
    max_frames: Optional[int],
    threshold: float,
    sam_window: int,
    gate_soft: bool,
    render_sam_masks: bool,
    no_video: bool,
    use_elbow_probe: bool = False,
    use_receiver_gate: bool = False,
    pair_cooldown: int = IMPACT_COOLDOWN_FRAMES,
    global_cooldown: int = GLOBAL_COOLDOWN_FRAMES,
    save_subdir: Optional[str] = None,
    temporal_ema_alpha: float = 0.0,       # E: Gaussian EMA smoothing on SAM window
    use_dual_correlation: bool = False,    # G: physics dual-body gate
    use_learned_scorer: bool = False,      # F: train LogReg on Approach D labels
    train_json_path: Optional[str] = None, # F: path to Approach D results JSON
):
    """Run one complete approach: detect → save JSON → render videos."""
    out_dir = os.path.join(output_dir, save_subdir) if save_subdir else output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Approach {tag}: {label}")
    print(f"  threshold={threshold}  sam_window={sam_window}  gate_soft={gate_soft}")
    print(f"  elbow_probe={use_elbow_probe}  receiver_gate={use_receiver_gate}")
    print(f"  dual_corr={use_dual_correlation}  ema_alpha={temporal_ema_alpha}")
    print(f"  pair_cooldown={pair_cooldown}  global_cooldown={global_cooldown}")
    print(f"  output -> {out_dir}")
    print(f"{'=' * 72}")

    # Approach F: train learned scorer from Approach D results
    learned_model = None
    if use_learned_scorer:
        learned_model = _train_learned_scorer(train_json_path or "")

    detector = JsonImpactDetector(
        frames_2d, frames_3d,
        threshold            = threshold,
        device               = DEVICE,
        sam_window           = sam_window,
        gate_soft            = gate_soft,
        sam_gate             = shared_sam_gate,
        use_elbow_probe      = use_elbow_probe,
        use_receiver_gate    = use_receiver_gate,
        temporal_ema_alpha   = temporal_ema_alpha,
        use_dual_correlation = use_dual_correlation,
        learned_model        = learned_model,
    )

    candidate_frames = detector.find_candidate_frames(actions)
    print(f"  {len(candidate_frames)} candidate frames")

    video_frames = extract_candidate_frames(video_path, candidate_frames)
    print(f"  Extracted {len(video_frames)} frames from video")

    all_results = detector.analyze_all(actions, video_frames=video_frames)

    cooldown = CooldownFilter(pair_cooldown=pair_cooldown, global_cooldown=global_cooldown)
    for r in sorted(all_results, key=lambda x: x.impact_frame):
        if r.is_impact and not cooldown.accept(r):
            r.is_impact = False

    print_results(all_results)

    json_out    = os.path.join(out_dir, f"results_{tag}.json")
    heatmap_out = os.path.join(out_dir, f"impact_map_{tag}.png")
    save_results_json(all_results, json_out)
    save_impact_heatmap(all_results, heatmap_out)

    if not no_video:
        out_real   = os.path.join(out_dir, f"{base}_{tag}_real.mp4")
        out_3d     = os.path.join(out_dir, f"{base}_{tag}_3d.mp4")
        impact_map = build_impact_map(all_results)
        run_video(
            video_path        = video_path,
            frames_2d         = frames_2d,
            impact_map        = impact_map,
            output_real       = out_real,
            output_3d         = out_3d,
            fps_video         = 30.0,
            stride            = stride,
            max_frames        = max_frames,
            render_sam_masks  = render_sam_masks,
            sam_gate          = shared_sam_gate if render_sam_masks else None,
            approach_label    = label,
        )
        landed = sum(1 for r in all_results if r.is_impact)
        print(f"\n  [{tag}]  {landed} impacts  |  real: {out_real}")
        print(f"  [{tag}]  3D  : {out_3d}")
        print(f"  [{tag}]  map : {heatmap_out}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="JSON Keypoint-Driven Hybrid (SAM+Physics) Impact Detection — 3 Approaches"
    )
    parser.add_argument("--video",       default=DEFAULT_VIDEO)
    parser.add_argument("--kp2d",        default=KEYPOINTS_2D_PATH)
    parser.add_argument("--kp3d",        default=KEYPOINTS_3D_PATH)
    parser.add_argument("--actions",     default=ACTIONS_PATH)
    parser.add_argument("--threshold",   type=float, default=IMPACT_SCORE_THRESHOLD)
    parser.add_argument("--stride",      type=int,   default=PROCESS_EVERY_N_FRAMES)
    parser.add_argument("--max-frames",  type=int,   default=None)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--no-video",    action="store_true",
                        help="Skip video rendering (analysis only)")
    parser.add_argument("--approach",    default="all",
                        choices=["all", "A", "B", "C", "D", "E", "F", "G"],
                        help="Run a single approach or all (default: all). "
                             "D=Enhanced  E=SAM2-style EMA  F=Learned LogReg  G=Dual-body corr")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  SAM3D  |  Seven-Approach Hybrid Impact Detection Pipeline")
    print(f"  Device : {DEVICE.upper()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    print("=" * 72)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data once ─────────────────────────────────────────────────
    print("\n-- Loading data " + "-" * 50)
    frames_2d = load_two_person_2d(args.kp2d)
    frames_3d = load_two_person_3d(args.kp3d)
    actions   = load_actions(args.actions)

    # ── 2. Load SAM once — shared across all three approaches ─────────────
    print("\n-- Loading SAM checkpoint (shared) " + "-" * 31)
    shared_sam: Optional[SamProximityGate] = None
    try:
        shared_sam = SamProximityGate(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)
    except (ImportError, FileNotFoundError) as e:
        print(f"[SAM] Unavailable ({e}) — all approaches will use keypoint fallback.")

    base = os.path.splitext(os.path.basename(args.video))[0]

    # ── Approach definitions ──────────────────────────────────────────────
    approaches = {
        "A": dict(
            label           = "A — Multi-frame SAM (±2 frames, window=5)",
            tag             = "A_multiframe",
            threshold       = args.threshold,
            sam_window      = 5,
            gate_soft       = False,
            render_sam_masks= False,
        ),
        "B": dict(
            label           = "B — Soft gate (physics rescue when SAM weak)",
            tag             = "B_softgate",
            threshold       = args.threshold,
            sam_window      = 1,
            gate_soft       = True,
            render_sam_masks= False,
        ),
        "C": dict(
            label           = "C — Lower threshold (0.38) + SAM heatmap",
            tag             = "C_heatmap",
            threshold       = 0.38,
            sam_window      = 1,
            gate_soft       = False,
            render_sam_masks= True,
        ),
        "D": dict(
            label            = "D — Enhanced (elbow probe + receiver reaction + reduced cooldown)",
            tag              = "D_enhanced",
            threshold        = 0.45,
            sam_window       = 3,
            gate_soft        = False,
            render_sam_masks = False,
            use_elbow_probe  = True,
            use_receiver_gate= True,
            pair_cooldown    = 15,
            global_cooldown  = 8,
            save_subdir      = "enhanced",
        ),
        "E": dict(
            label              = "E — SAM2-style temporal EMA (window=7, alpha=0.65)",
            tag                = "E_sam2style",
            threshold          = 0.45,
            sam_window         = 7,
            gate_soft          = False,
            render_sam_masks   = False,
            temporal_ema_alpha = 0.65,
            save_subdir        = "sam2style",
        ),
        "F": dict(
            label              = "F — Learned LogReg (trained on Approach D gate scores)",
            tag                = "F_learned",
            threshold          = 0.50,
            sam_window         = 3,
            gate_soft          = False,
            render_sam_masks   = False,
            use_elbow_probe    = True,
            use_receiver_gate  = True,
            pair_cooldown      = 15,
            global_cooldown    = 8,
            use_learned_scorer = True,
            save_subdir        = "learned",
        ),
        "G": dict(
            label                = "G — Dual-body correlation (striker decel x receiver accel)",
            tag                  = "G_dual_corr",
            threshold            = 0.40,
            sam_window           = 3,
            gate_soft            = False,
            render_sam_masks     = False,
            use_elbow_probe      = True,
            use_dual_correlation = True,
            pair_cooldown        = 15,
            global_cooldown      = 8,
            save_subdir          = "dual_corr",
        ),
    }

    # Approach F needs Approach D results — ensure D runs before F
    run_keys_input = ["A", "B", "C", "D", "E", "F", "G"] if args.approach == "all" else [args.approach]
    # If F is requested without D having run, warn
    if "F" in run_keys_input and "D" not in run_keys_input:
        d_json = os.path.join(args.output_dir, "enhanced", "results_D_enhanced.json")
        if not os.path.isfile(d_json):
            print("[WARN] Approach F needs Approach D results. Adding D to this run.")
            run_keys_input = ["D"] + run_keys_input
    run_keys = run_keys_input
    all_approach_results: dict[str, list[ImpactResult]] = {}

    for key in run_keys:
        cfg = approaches[key]
        results = _run_approach(
            label             = cfg["label"],
            tag               = cfg["tag"],
            frames_2d         = frames_2d,
            frames_3d         = frames_3d,
            actions           = actions,
            video_path        = args.video,
            output_dir        = args.output_dir,
            shared_sam_gate   = shared_sam,
            base              = base,
            stride            = args.stride,
            max_frames        = args.max_frames,
            threshold         = cfg["threshold"],
            sam_window        = cfg["sam_window"],
            gate_soft         = cfg["gate_soft"],
            render_sam_masks  = cfg["render_sam_masks"],
            no_video              = args.no_video,
            use_elbow_probe       = cfg.get("use_elbow_probe",      False),
            use_receiver_gate     = cfg.get("use_receiver_gate",    False),
            pair_cooldown         = cfg.get("pair_cooldown",        IMPACT_COOLDOWN_FRAMES),
            global_cooldown       = cfg.get("global_cooldown",      GLOBAL_COOLDOWN_FRAMES),
            save_subdir           = cfg.get("save_subdir",          None),
            temporal_ema_alpha    = cfg.get("temporal_ema_alpha",   0.0),
            use_dual_correlation  = cfg.get("use_dual_correlation", False),
            use_learned_scorer    = cfg.get("use_learned_scorer",   False),
            train_json_path       = os.path.join(
                args.output_dir, "enhanced", "results_D_enhanced.json"
            ),
        )
        all_approach_results[key] = results

    # ── Summary comparison ────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  All approaches complete in {elapsed:.1f}s")
    print(f"  {'Approach':<55}  {'Landed':>7}  {'Missed':>7}  {'Rate':>6}")
    print(f"  {'-' * 78}")
    for key in run_keys_input:          # only user-requested keys in summary
        if key not in all_approach_results:
            continue
        cfg = approaches[key]
        res = all_approach_results[key]
        landed = sum(1 for r in res if r.is_impact)
        missed = sum(1 for r in res if not r.is_impact)
        rate   = landed / max(len(res), 1)
        print(f"  {cfg['label']:<55}  {landed:>7}  {missed:>7}  {rate:>5.0%}")
    print(f"{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
