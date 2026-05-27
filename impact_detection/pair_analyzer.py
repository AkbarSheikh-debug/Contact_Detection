"""
Pi-HOC Inspired Pairwise Interaction Analyzer — adapted for boxing punch detection.

Three root-cause fixes for false positives (see diagnostic data):
  Fix 1 — Elbows removed from STRIKING_KPS_INFO (guard position always near torso)
  Fix 2 — Arm extension gate: wrist must be arm-extended (ratio >= ARM_EXTENSION_MIN)
           to be a punch, not a guard or retraction
  Fix 3 — Directional velocity: wrist velocity vector projected onto direction-to-target
           must exceed MIN_DIRECTED_VELOCITY (eliminates lateral swings, retreats, jitter)

Scoring formula (four components, weights sum to 1.0):
  contact_prob = W_PROXIMITY*proximity + W_DIRECTED_VEL*directed_vel
               + W_EXTENSION*extension + W_CONFIDENCE*conf
"""
import numpy as np
from dataclasses import dataclass, field

from config import (
    IOI_THRESHOLD, CONTACT_THRESHOLD,
    PROXIMITY_THRESHOLD, VELOCITY_THRESHOLD,
    W_PROXIMITY, W_DIRECTED_VEL, W_EXTENSION, W_CONFIDENCE,
    ARM_EXTENSION_MIN, MIN_DIRECTED_VELOCITY, MAX_WRIST_VELOCITY,
    STRIKING_KPS_INFO, HEAD_KPS, TORSO_KPS, KP,
)


@dataclass
class ContactFeatures:
    """Pair token equivalent from Pi-HOC Sec 3.2."""
    contact_probability: float = 0.0
    contact_point: list = field(default_factory=list)
    impact_type: str = "none"
    striking_limb: str = "none"
    velocity: float = 0.0           # net directed velocity (px/frame) toward opponent
    extension_ratio: float = 0.0    # arm extension ratio at time of detection
    contact_region: str = "none"
    aggressor_id: int = -1
    receiver_id: int = -1


class PairInteractionAnalyzer:
    """
    Pi-HOC style pairwise contact analyzer adapted for boxing.

    Step 1 — Pair formation:  all fighter pairs with box_iou >= γ (paper Sec 3.2)
    Step 2 — Token construction: wrist position + directed velocity + extension → feature
    Step 3 — Gate: extension < ARM_EXTENSION_MIN → skip (bent arm, not a punch)
              Gate: directed_vel < MIN_DIRECTED_VELOCITY → skip (not moving toward opponent)
    Step 4 — Contact presence: score >= δ → impact event
    Step 5 — Contact localisation: head vs torso
    """

    def __init__(self):
        self._prev: dict[int, dict] = {}   # track_id → {keypoints: (17,2), frame: int}

    # ── Public API ───────────────────────────────────────────────────────────

    def form_pairs(self, persons: list[dict]) -> list[tuple[int, int]]:
        """Return all (i, j) index pairs whose bounding boxes overlap by >= γ."""
        pairs = []
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                if _box_iou(persons[i]["bbox"], persons[j]["bbox"]) >= IOI_THRESHOLD:
                    pairs.append((i, j))
        return pairs

    def analyze(
        self,
        persons: list[dict],
        pairs: list[tuple[int, int]],
        frame_idx: int,
    ) -> list[ContactFeatures]:
        """Compute contact features for all pairs; return those above threshold δ."""
        impacts: list[ContactFeatures] = []

        for idx_a, idx_b in pairs:
            pa, pb = persons[idx_a], persons[idx_b]
            for striker, receiver, s_id, r_id in [
                (pa, pb, pa["track_id"], pb["track_id"]),
                (pb, pa, pb["track_id"], pa["track_id"]),
            ]:
                if striker["keypoints"] is None or receiver["keypoints"] is None:
                    continue
                feat = self._compute_contact_features(striker, receiver, s_id, r_id, frame_idx)
                if feat.contact_probability >= CONTACT_THRESHOLD:
                    impacts.append(feat)

        # Store keypoints for next-frame velocity/extension computation
        for p in persons:
            if p["keypoints"] is not None:
                self._prev[p["track_id"]] = {
                    "keypoints": p["keypoints"]["points"].copy(),
                    "frame": frame_idx,
                }

        return impacts

    # ── Core contact feature computation ────────────────────────────────────

    def _compute_contact_features(
        self,
        striker: dict,
        receiver: dict,
        striker_id: int,
        receiver_id: int,
        frame_idx: int,
    ) -> ContactFeatures:
        kp_s   = striker["keypoints"]["points"]    # (17, 2)
        conf_s = striker["keypoints"]["confidence"]
        kp_r   = receiver["keypoints"]["points"]
        conf_r = receiver["keypoints"]["confidence"]

        # Fast pre-check: receiver bbox must be within reach of any wrist
        rx1, ry1, rx2, ry2 = receiver["bbox"]
        reach = PROXIMITY_THRESHOLD * 2
        expanded = (rx1 - reach, ry1 - reach, rx2 + reach, ry2 + reach)
        wrists_in_range = False
        for sk_idx, _ in STRIKING_KPS_INFO:
            if conf_s[sk_idx] >= 0.25:
                wx, wy = kp_s[sk_idx]
                if expanded[0] <= wx <= expanded[2] and expanded[1] <= wy <= expanded[3]:
                    wrists_in_range = True
                    break
        if not wrists_in_range:
            return ContactFeatures(aggressor_id=striker_id, receiver_id=receiver_id)

        # Pre-compute whole-body velocity once (subtracted from wrist vel)
        body_vel = self._get_body_velocity(striker_id, kp_s, frame_idx)

        target_regions = {"head": HEAD_KPS, "torso": TORSO_KPS}
        best = ContactFeatures(aggressor_id=striker_id, receiver_id=receiver_id)

        for sk_idx, limb_name in STRIKING_KPS_INFO:
            if conf_s[sk_idx] < 0.25:
                continue

            wrist_pos = kp_s[sk_idx]
            if np.allclose(wrist_pos, 0):
                continue

            # ── Fix 2: Arm extension gate ─────────────────────────────────────
            elbow_idx    = KP["left_elbow"]    if sk_idx == KP["left_wrist"]  else KP["right_elbow"]
            shoulder_idx = KP["left_shoulder"] if sk_idx == KP["left_wrist"]  else KP["right_shoulder"]

            elbow_pos    = kp_s[elbow_idx]
            shoulder_pos = kp_s[shoulder_idx]

            joints_visible = (
                not np.allclose(elbow_pos, 0)
                and not np.allclose(shoulder_pos, 0)
                and conf_s[elbow_idx]    >= 0.25
                and conf_s[shoulder_idx] >= 0.25
            )

            if joints_visible:
                wrist_to_shoulder = float(np.linalg.norm(wrist_pos    - shoulder_pos))
                wrist_to_elbow    = float(np.linalg.norm(wrist_pos    - elbow_pos))
                elbow_to_shoulder = float(np.linalg.norm(elbow_pos    - shoulder_pos))
                full_arm_len      = wrist_to_elbow + elbow_to_shoulder

                if full_arm_len < 1e-3:
                    continue  # degenerate joint positions

                extension_ratio = wrist_to_shoulder / full_arm_len
                if extension_ratio < ARM_EXTENSION_MIN:
                    continue  # arm bent — guard or retraction — not a punch
            else:
                # Can't measure extension; use neutral value (won't add bonus)
                extension_ratio = ARM_EXTENSION_MIN

            # ── Fix 3: Directional velocity ───────────────────────────────────
            vel_vec     = self._get_velocity_vector(striker_id, sk_idx, wrist_pos, frame_idx)
            net_vel_vec = vel_vec - body_vel   # subtract whole-body drift

            for region, kp_indices in target_regions.items():
                for t_idx in kp_indices:
                    if conf_r[t_idx] < 0.25:
                        continue
                    target_pos = kp_r[t_idx]
                    if np.allclose(target_pos, 0):
                        continue

                    diff = target_pos.astype(float) - wrist_pos.astype(float)
                    dist = float(np.linalg.norm(diff))
                    if dist < 1e-3:
                        continue

                    dir_unit     = diff / dist
                    directed_vel = float(np.dot(net_vel_vec, dir_unit))

                    # Gate: wrist must be moving toward opponent
                    if directed_vel < MIN_DIRECTED_VELOCITY:
                        continue

                    # ── Four-component scoring ────────────────────────────────
                    proximity_score    = max(0.0, 1.0 - dist / PROXIMITY_THRESHOLD)
                    directed_vel_score = min(1.0, max(0.0, directed_vel) / max(VELOCITY_THRESHOLD, 1e-6))
                    extension_score    = min(1.0, (extension_ratio - ARM_EXTENSION_MIN)
                                             / max(1.0 - ARM_EXTENSION_MIN, 1e-6))
                    conf_score         = float(conf_s[sk_idx]) * float(conf_r[t_idx])

                    contact_prob = (
                        W_PROXIMITY    * proximity_score
                        + W_DIRECTED_VEL * directed_vel_score
                        + W_EXTENSION    * extension_score
                        + W_CONFIDENCE   * conf_score
                    )

                    if contact_prob > best.contact_probability:
                        mid = ((wrist_pos.astype(float) + target_pos.astype(float)) / 2
                               ).astype(int).tolist()
                        best = ContactFeatures(
                            contact_probability=contact_prob,
                            contact_point=mid,
                            impact_type=f"{region}_impact",
                            striking_limb=limb_name,
                            velocity=directed_vel,
                            extension_ratio=extension_ratio,
                            contact_region=region,
                            aggressor_id=striker_id,
                            receiver_id=receiver_id,
                        )

        return best

    # ── Velocity helpers ─────────────────────────────────────────────────────

    def _get_velocity_vector(
        self,
        track_id: int,
        kp_idx: int,
        current_pos: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """
        2D displacement vector (px/frame) for a keypoint.
        Clamped to MAX_WRIST_VELOCITY to suppress tracker-ID-switch spikes.
        """
        zero = np.zeros(2, dtype=float)
        if track_id not in self._prev:
            return zero
        prev = self._prev[track_id]
        dt = frame_idx - prev["frame"]
        if dt <= 0:
            return zero
        prev_pos = prev["keypoints"][kp_idx]
        if np.allclose(prev_pos, 0):
            return zero
        raw = (current_pos.astype(float) - prev_pos.astype(float)) / dt
        mag = float(np.linalg.norm(raw))
        if mag > MAX_WRIST_VELOCITY:
            raw = raw * (MAX_WRIST_VELOCITY / mag)
        return raw

    def _get_body_velocity(
        self,
        track_id: int,
        kp_s: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """
        Estimate whole-body translation from hip centroid (fallback: shoulders).
        Subtracted from wrist velocity so footwork is not confused with punching.
        """
        zero = np.zeros(2, dtype=float)
        if track_id not in self._prev:
            return zero
        prev = self._prev[track_id]
        dt = frame_idx - prev["frame"]
        if dt <= 0:
            return zero
        prev_kp = prev["keypoints"]

        def centroid_vel(indices: list[int]) -> np.ndarray | None:
            curr_pts, prev_pts = [], []
            for i in indices:
                c, p = kp_s[i], prev_kp[i]
                if not np.allclose(c, 0) and not np.allclose(p, 0):
                    curr_pts.append(c.astype(float))
                    prev_pts.append(p.astype(float))
            if not curr_pts:
                return None
            return (np.mean(curr_pts, axis=0) - np.mean(prev_pts, axis=0)) / dt

        vel = centroid_vel([KP["left_hip"], KP["right_hip"]])
        if vel is None:
            vel = centroid_vel([KP["left_shoulder"], KP["right_shoulder"]])
        return vel if vel is not None else zero


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)
