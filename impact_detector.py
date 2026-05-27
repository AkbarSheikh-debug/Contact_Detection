"""
Impact Detector — 5-Gate Physics-Based Impact Classification
=============================================================
Determines whether each ASFormer-detected punch actually LANDS by analysing
the striker's wrist kinematics in both 2D and 3D space.

Five scoring gates
------------------
1. Wrist deceleration ratio   — sharp velocity drop = fist hit something
2. 3D jerk magnitude          — sudden force change pinpoints impact frame
3. Arm extension pattern      — punch must reach near-full extension
4. 3D depth convergence       — fist reverses direction at impact
5. Action confidence boost    — ASFormer confidence + power as support

Each gate produces a [0, 1] sub-score; final score is a weighted sum.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from keypoint_loader import (
    KeypointLoader, FrameData2D, FrameData3D, ActionEvent,
)
from config import (
    IMPACT_SCORE_THRESHOLD, WINDOW_PAD_FRAMES,
    DECEL_RATIO_WEIGHT, JERK_WEIGHT, EXTENSION_WEIGHT,
    DEPTH_WEIGHT, CONFIDENCE_WEIGHT,
    ACTION_HAND_MAP, ACTION_ARM_CHAIN,
    KP70_LEFT_HIP, KP70_RIGHT_HIP,
)


@dataclass
class ImpactResult:
    """Result of impact analysis for one action event."""
    # Source action
    action: str
    action_frame: int
    window_start: int
    window_end: int
    timestamp_seconds: float
    target: str
    action_confidence: float
    speed_kmh: float
    power_watts: float

    # Impact analysis
    is_impact: bool = False
    impact_score: float = 0.0
    impact_frame: int = -1          # precise frame from jerk analysis
    striking_hand: str = "unknown"

    # Gate sub-scores
    decel_score: float = 0.0
    jerk_score: float = 0.0
    extension_score: float = 0.0
    depth_score: float = 0.0
    confidence_score: float = 0.0

    # Kinematics at impact
    peak_velocity_3d: float = 0.0
    deceleration_magnitude: float = 0.0
    arm_extension_at_impact: float = 0.0

    # Debug
    velocity_profile: list = field(default_factory=list)
    gate_details: dict = field(default_factory=dict)


class ImpactDetector:
    """
    Analyses each ASFormer action window using pre-extracted keypoints
    to determine whether the punch actually landed.
    """

    def __init__(
        self,
        frames_2d: dict[int, FrameData2D],
        frames_3d: dict[int, FrameData3D],
        threshold: float = IMPACT_SCORE_THRESHOLD,
    ):
        self.frames_2d = frames_2d
        self.frames_3d = frames_3d
        self.threshold = threshold
        self.loader = KeypointLoader()

    def analyze_all(self, actions: list[ActionEvent]) -> list[ImpactResult]:
        """Run impact detection on every action event."""
        results: list[ImpactResult] = []
        for i, act in enumerate(actions):
            result = self.analyze_action(act)
            results.append(result)
        return results

    def analyze_action(self, action: ActionEvent) -> ImpactResult:
        """
        Full 5-gate analysis for a single action event.
        """
        # Determine which wrist to track
        wrist_idx = ACTION_HAND_MAP.get(action.action)
        if wrist_idx is None:
            # Unknown action type — default to right wrist
            wrist_idx = 10

        hand_name = "left" if wrist_idx == 9 else "right"
        elbow_idx, shoulder_idx = ACTION_ARM_CHAIN[wrist_idx]

        # Extended analysis window
        ws = max(0, action.window_start - WINDOW_PAD_FRAMES)
        we = action.window_end + WINDOW_PAD_FRAMES

        # Create base result
        result = ImpactResult(
            action=action.action,
            action_frame=action.frame,
            window_start=action.window_start,
            window_end=action.window_end,
            timestamp_seconds=action.timestamp_seconds,
            target=action.target,
            action_confidence=action.confidence,
            speed_kmh=action.speed_kmh,
            power_watts=action.power_watts,
            striking_hand=hand_name,
        )

        # Get available frames in window
        frame_range = self.loader.get_frame_range(self.frames_3d, ws, we)
        if len(frame_range) < 3:
            # Not enough frames for kinematic analysis
            return result

        # ── Gate 1: Wrist Deceleration (2D + 3D combined) ────────────────
        decel_score, peak_vel, decel_mag, vel_profile = self._gate_deceleration(
            frame_range, wrist_idx
        )

        # ── Gate 2: 3D Jerk Analysis ─────────────────────────────────────
        jerk_score, impact_frame = self._gate_jerk(frame_range, wrist_idx)

        # ── Gate 3: Arm Extension Pattern ────────────────────────────────
        ext_score, ext_at_impact = self._gate_extension(
            frame_range, wrist_idx, elbow_idx, shoulder_idx,
            action.window_start, action.window_end,
        )

        # ── Gate 4: 3D Depth Convergence ─────────────────────────────────
        depth_score = self._gate_depth_convergence(frame_range, wrist_idx)

        # ── Gate 5: Action Confidence Boost ──────────────────────────────
        conf_score = self._gate_confidence(action)

        # ── Weighted scoring ─────────────────────────────────────────────
        impact_score = (
            DECEL_RATIO_WEIGHT  * decel_score
            + JERK_WEIGHT       * jerk_score
            + EXTENSION_WEIGHT  * ext_score
            + DEPTH_WEIGHT      * depth_score
            + CONFIDENCE_WEIGHT * conf_score
        )

        # Use best impact frame from jerk; fall back to action frame
        if impact_frame < 0:
            impact_frame = action.frame

        result.is_impact = impact_score >= self.threshold
        result.impact_score = round(impact_score, 4)
        result.impact_frame = impact_frame
        result.decel_score = round(decel_score, 4)
        result.jerk_score = round(jerk_score, 4)
        result.extension_score = round(ext_score, 4)
        result.depth_score = round(depth_score, 4)
        result.confidence_score = round(conf_score, 4)
        result.peak_velocity_3d = round(peak_vel, 4)
        result.deceleration_magnitude = round(decel_mag, 4)
        result.arm_extension_at_impact = round(ext_at_impact, 4)
        result.velocity_profile = vel_profile

        result.gate_details = {
            "decel": f"{decel_score:.3f} (peak_vel={peak_vel:.2f}, decel={decel_mag:.2f})",
            "jerk":  f"{jerk_score:.3f} (impact_frame={impact_frame})",
            "ext":   f"{ext_score:.3f} (ext@impact={ext_at_impact:.3f})",
            "depth": f"{depth_score:.3f}",
            "conf":  f"{conf_score:.3f} (asformer={action.confidence:.3f}, "
                     f"power={action.power_watts:.0f}W)",
        }

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # Gate Implementations
    # ═════════════════════════════════════════════════════════════════════════

    def _gate_deceleration(
        self,
        frame_range: list[int],
        wrist_idx: int,
    ) -> tuple[float, float, float, list]:
        """
        Gate 1: Velocity deceleration ratio.

        A landed punch shows:  acceleration → peak → SHARP deceleration
        A missed punch shows:  acceleration → peak → gradual follow-through

        Returns: (score, peak_velocity, max_deceleration, velocity_profile)
        """
        # Use 3D normalized coordinates for velocity
        fnums, positions = self.loader.extract_joint_trajectory(
            self.frames_3d, frame_range, wrist_idx, use_normalized=True,
        )

        if len(fnums) < 3:
            return 0.0, 0.0, 0.0, []

        # Compute frame-to-frame velocity magnitudes
        dt = np.diff(fnums).astype(float)
        dt[dt == 0] = 1.0
        displacements = np.diff(positions, axis=0)
        velocities = np.linalg.norm(displacements, axis=1) / dt

        vel_profile = [
            {"frame": int(fnums[i + 1]), "velocity": float(velocities[i])}
            for i in range(len(velocities))
        ]

        if len(velocities) < 2:
            return 0.0, 0.0, 0.0, vel_profile

        peak_vel = float(np.max(velocities))
        if peak_vel < 1e-6:
            return 0.0, 0.0, 0.0, vel_profile

        # Compute acceleration (change in velocity magnitude)
        accel = np.diff(velocities) / dt[1:]

        # Maximum deceleration (most negative acceleration)
        max_decel = float(-np.min(accel)) if len(accel) > 0 else 0.0
        max_decel = max(0.0, max_decel)

        # Deceleration ratio: how sharply velocity drops relative to peak
        decel_ratio = max_decel / peak_vel

        # Score: sigmoid-like mapping. Ratio > 0.5 is strong evidence.
        score = min(1.0, decel_ratio / 0.6)

        return score, peak_vel, max_decel, vel_profile

    def _gate_jerk(
        self,
        frame_range: list[int],
        wrist_idx: int,
    ) -> tuple[float, int]:
        """
        Gate 2: 3D jerk analysis (derivative of acceleration).

        High jerk at a specific frame = sudden force change = impact.
        Returns: (score, impact_frame)
        """
        fnums, positions = self.loader.extract_joint_trajectory(
            self.frames_3d, frame_range, wrist_idx, use_normalized=True,
        )

        if len(fnums) < 4:
            return 0.0, -1

        dt = np.diff(fnums).astype(float)
        dt[dt == 0] = 1.0

        # Velocity vectors
        vel_vecs = np.diff(positions, axis=0) / dt[:, None]

        # Acceleration vectors
        if len(vel_vecs) < 2:
            return 0.0, -1
        dt2 = dt[1:]
        accel_vecs = np.diff(vel_vecs, axis=0) / dt2[:, None]

        # Jerk vectors
        if len(accel_vecs) < 2:
            return 0.0, -1
        dt3 = dt2[1:]
        jerk_vecs = np.diff(accel_vecs, axis=0) / dt3[:, None]

        jerk_mags = np.linalg.norm(jerk_vecs, axis=1)

        if len(jerk_mags) == 0:
            return 0.0, -1

        max_jerk_idx = int(np.argmax(jerk_mags))
        max_jerk = float(jerk_mags[max_jerk_idx])

        # The impact frame is 3 indices ahead in fnums due to 3 derivatives
        impact_frame = int(fnums[max_jerk_idx + 3]) if max_jerk_idx + 3 < len(fnums) else int(fnums[-1])

        # Score: normalize jerk. Empirical threshold ~0.02 for normalized coords.
        score = min(1.0, max_jerk / 0.03)

        return score, impact_frame

    def _gate_extension(
        self,
        frame_range: list[int],
        wrist_idx: int,
        elbow_idx: int,
        shoulder_idx: int,
        action_start: int,
        action_end: int,
    ) -> tuple[float, float]:
        """
        Gate 3: Arm extension pattern.

        Track extension ratio = wrist-to-shoulder / (wrist-to-elbow + elbow-to-shoulder).
        A valid punch must reach peak extension WITHIN the action window.

        Returns: (score, extension_at_impact)
        """
        extensions = []
        ext_in_window = []

        for f in frame_range:
            if f not in self.frames_2d:
                continue
            joints = self.frames_2d[f].joints_2d

            wrist = joints[wrist_idx]
            elbow = joints[elbow_idx]
            shoulder = joints[shoulder_idx]

            w2s = float(np.linalg.norm(wrist - shoulder))
            w2e = float(np.linalg.norm(wrist - elbow))
            e2s = float(np.linalg.norm(elbow - shoulder))
            full_arm = w2e + e2s

            if full_arm < 1e-3:
                continue

            ext = w2s / full_arm
            extensions.append((f, ext))

            if action_start <= f <= action_end:
                ext_in_window.append((f, ext))

        if not extensions:
            return 0.0, 0.0

        # Peak extension across entire range
        peak_ext_overall = max(e for _, e in extensions)

        # Peak extension within action window
        if ext_in_window:
            peak_ext_window = max(e for _, e in ext_in_window)
            peak_frame = max(ext_in_window, key=lambda x: x[1])[0]
        else:
            peak_ext_window = peak_ext_overall
            peak_frame = max(extensions, key=lambda x: x[1])[0]

        # Score: how close to full extension (1.0). Good punches reach 0.80+
        score = min(1.0, max(0.0, (peak_ext_window - 0.60) / 0.35))

        return score, peak_ext_window

    def _gate_depth_convergence(
        self,
        frame_range: list[int],
        wrist_idx: int,
    ) -> float:
        """
        Gate 4: 3D depth convergence.

        In 3D space a punch moves the fist forward. A LANDED punch reverses
        direction (recoil/retraction) more sharply than a miss.

        Measure: ratio of forward displacement to total displacement.
        Then check for direction reversal in the dominant movement axis.
        """
        fnums, positions = self.loader.extract_joint_trajectory(
            self.frames_3d, frame_range, wrist_idx, use_normalized=False,
        )

        if len(fnums) < 4:
            return 0.0

        # Determine primary direction of motion (first half → second half)
        mid = len(positions) // 2
        first_half_disp = positions[mid] - positions[0]
        second_half_disp = positions[-1] - positions[mid]

        # Direction reversal: dot product < 0 means reversal
        dot = float(np.dot(first_half_disp, second_half_disp))
        mag_first = float(np.linalg.norm(first_half_disp))
        mag_second = float(np.linalg.norm(second_half_disp))

        if mag_first < 1e-6 or mag_second < 1e-6:
            return 0.0

        # Cosine similarity: -1 = perfect reversal, +1 = same direction
        cos_sim = dot / (mag_first * mag_second)

        # Score: reversal (cos < 0) is evidence of impact
        # Perfect reversal (cos=-1) → score=1.0
        # No reversal (cos=1) → score=0.0
        score = max(0.0, min(1.0, (1.0 - cos_sim) / 1.5))

        return score

    def _gate_confidence(self, action: ActionEvent) -> float:
        """
        Gate 5: Action confidence and power boost.

        Combines ASFormer confidence with estimated power.
        High confidence + high power = likely real impact.
        """
        # Confidence component [0, 1] — ASFormer confidence as-is
        conf_part = min(1.0, action.confidence)

        # Power component [0, 1] — 2000W is a strong punch
        power_part = min(1.0, action.power_watts / 3000.0)

        # Speed component [0, 1] — 20 km/h is a solid punch speed
        speed_part = min(1.0, action.speed_kmh / 25.0)

        # Combined: confidence dominates, power/speed as support
        score = 0.50 * conf_part + 0.25 * power_part + 0.25 * speed_part

        return score

    # ═════════════════════════════════════════════════════════════════════════
    # Summary helpers
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def summary(results: list[ImpactResult]) -> dict:
        """Generate aggregate statistics from impact results."""
        total = len(results)
        landed = [r for r in results if r.is_impact]
        missed = [r for r in results if not r.is_impact]

        by_type: dict[str, dict] = {}
        for r in results:
            if r.action not in by_type:
                by_type[r.action] = {"total": 0, "landed": 0}
            by_type[r.action]["total"] += 1
            if r.is_impact:
                by_type[r.action]["landed"] += 1

        return {
            "total_actions": total,
            "total_landed": len(landed),
            "total_missed": len(missed),
            "landing_rate": len(landed) / max(total, 1),
            "avg_impact_score": (
                float(np.mean([r.impact_score for r in landed])) if landed else 0.0
            ),
            "avg_peak_velocity": (
                float(np.mean([r.peak_velocity_3d for r in landed])) if landed else 0.0
            ),
            "by_type": by_type,
        }
