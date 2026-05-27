"""
Optical flow estimator — provides dense motion information to support impact detection.
Lucas-Kanade sparse flow is used to track wrist/glove keypoints across frames.
"""
import cv2
import numpy as np

from config import OPTICAL_FLOW_WINSIZE, STRIKING_KPS_INFO


_LK_PARAMS = dict(
    winSize=(OPTICAL_FLOW_WINSIZE, OPTICAL_FLOW_WINSIZE),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


class OpticalFlowEstimator:
    """
    Lucas-Kanade sparse optical flow for tracking wrist keypoints.
    Also computes a lightweight dense flow magnitude heatmap for visualisation.
    """

    def __init__(self):
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None   # (N, 1, 2) float32

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Update flow estimator with new frame.
        Returns dense flow magnitude map (HxW float32).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros(gray.shape, dtype=np.float32)

        # Dense Farneback flow for heatmap
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)

        self._prev_gray = gray
        return magnitude

    def track_keypoints(
        self,
        frame_bgr: np.ndarray,
        keypoints: np.ndarray,
        confidences: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """
        Sparse LK tracking for striking keypoints.
        Returns dict {kp_idx: tracked_position (2,)} for high-confidence points.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        result = {}

        if self._prev_gray is None:
            self._prev_gray = gray
            return result

        pts = []
        indices = []
        for kp_idx, _ in STRIKING_KPS_INFO:
            if confidences[kp_idx] >= 0.3 and not np.allclose(keypoints[kp_idx], 0):
                pts.append(keypoints[kp_idx].astype(np.float32).reshape(1, 1, 2))
                indices.append(kp_idx)

        if not pts:
            self._prev_gray = gray
            return result

        prev_pts = np.concatenate(pts, axis=0)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **_LK_PARAMS
        )

        if next_pts is not None:
            for i, (idx, ok) in enumerate(zip(indices, status.flatten())):
                if ok:
                    result[idx] = next_pts[i, 0]

        self._prev_gray = gray
        return result

    def flow_magnitude_at_points(
        self,
        flow_map: np.ndarray,
        points: list[tuple[int, int]],
    ) -> list[float]:
        """Sample flow magnitude at given (x, y) pixel coordinates."""
        h, w = flow_map.shape
        return [
            float(flow_map[min(int(y), h - 1), min(int(x), w - 1)])
            for x, y in points
        ]
