"""
2D/3D geometry utilities for impact detection.
Depth is approximated from bounding-box area (larger bbox ≈ closer to camera).
"""
import numpy as np


def box_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=float)


def box_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    union = box_area(a) + box_area(b) - inter
    return inter / (union + 1e-6)


def estimate_depth_proxy(bbox: np.ndarray, frame_area: float) -> float:
    """
    Mono-depth proxy: larger bounding box → person is closer.
    Returns a normalized depth value in [0, 1] where 1 = closest.
    """
    return float(box_area(bbox)) / (frame_area + 1e-6)


def keypoint_velocity(
    current: np.ndarray,
    previous: np.ndarray,
    dt: int,
) -> np.ndarray:
    """Per-keypoint velocity vector (pixels/frame)."""
    if dt <= 0:
        return np.zeros_like(current)
    return (current - previous) / dt


def wrist_to_body_distance(
    wrist_pos: np.ndarray,
    body_keypoints: np.ndarray,
    body_confs: np.ndarray,
    kp_indices: list[int],
    conf_threshold: float = 0.25,
) -> tuple[float, int]:
    """
    Minimum distance from wrist to any visible body keypoint.
    Returns (min_distance, best_target_index).
    """
    best_dist = float("inf")
    best_idx = -1
    for idx in kp_indices:
        if body_confs[idx] < conf_threshold:
            continue
        pos = body_keypoints[idx]
        if np.allclose(pos, 0):
            continue
        dist = float(np.linalg.norm(wrist_pos - pos))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_dist, best_idx


def smooth_keypoints(
    current: np.ndarray,
    previous: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Exponential moving average smoothing for keypoint positions."""
    return alpha * current + (1.0 - alpha) * previous
