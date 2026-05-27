"""
SMPL body state manager for the 3D impact visualization video.

Maintains current contact state and only re-renders the SMPL mesh when the
state changes — O(impacts) renders instead of O(frames), keeping CPU load low.
"""
import cv2
import numpy as np

from utils.smpl_mesh_viz import render_smpl_pair
from config import SMPL_HOLD_FRAMES


class SMPLVideoRenderer:
    """
    Call update() once per processed frame.
    Returns a BGR canvas the same dimensions as the real output video.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self._W = frame_width
        self._H = frame_height
        self._cache: np.ndarray | None = None

        self._regions_a: list[str] = []
        self._regions_b: list[str] = []
        self._label_a: str = "Fighter A"
        self._label_b: str = "Fighter B"
        self._contact_region: str = ""
        self._hold: int = 0
        self._n_impacts: int = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, new_impact=None, frame_idx: int = 0, fps: float = 30.0) -> np.ndarray:
        """Return the 3D visualization frame.  Pass ImpactEvent on impact, else None."""
        dirty = False

        if new_impact is not None:
            from utils.body_contact_viz import BodyContactDiagram
            hit_r, strike_r = BodyContactDiagram.regions_from_event(
                new_impact.contact_region, new_impact.striking_limb)
            self._regions_a = strike_r or ["torso"]
            self._regions_b = hit_r   or ["torso"]
            self._contact_region = new_impact.contact_region
            self._label_a = f"Fighter {new_impact.aggressor_id}"
            self._label_b = f"Fighter {new_impact.receiver_id}"
            self._hold = SMPL_HOLD_FRAMES
            self._n_impacts += 1
            dirty = True

        elif self._hold > 0:
            self._hold -= 1
            if self._hold == 0:
                self._regions_a = []
                self._regions_b = []
                self._contact_region = ""
                dirty = True

        if dirty or self._cache is None:
            self._cache = self._render()

        out = self._cache.copy()
        self._stamp_time(out, frame_idx, fps)
        return out

    # ── Private ───────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        body_h = self._H - 110
        body_w = max(200, (self._W - 50) // 2)

        pair = render_smpl_pair(
            contact_regions_a=self._regions_a,
            contact_regions_b=self._regions_b,
            label_a=self._label_a,
            label_b=self._label_b,
            width=body_w,
            height=body_h,
        )

        canvas = np.full((self._H, self._W, 3), 10, dtype=np.uint8)

        ph, pw = pair.shape[:2]
        y0 = 80 + (body_h - ph) // 2
        x0 = (self._W - pw) // 2
        y1, x1 = min(y0 + ph, self._H), min(x0 + pw, self._W)
        canvas[y0:y1, x0:x1] = pair[:y1 - y0, :x1 - x0]

        # Title bar
        cv2.putText(canvas, "SAM3D  |  3D Impact Visualization",
                    (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.80,
                    (0, 210, 255), 1, cv2.LINE_AA)

        # Column headers
        cv2.putText(canvas, "AGGRESSOR", (x0 + 10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, "RECEIVER",  (x0 + pw // 2 + 20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

        # Status
        if self._regions_b:
            region_str = self._contact_region.replace("_", " ").upper()
            status     = f"IMPACT  |  Contact: {region_str}"
            status_col = (30, 50, 255)
        else:
            status     = "Monitoring ..."
            status_col = (70, 70, 70)

        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.putText(canvas, status, (self._W - tw - 20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, status_col, 1, cv2.LINE_AA)

        # Impact counter (bottom-left)
        cv2.putText(canvas, f"Impacts detected: {self._n_impacts}",
                    (20, self._H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)

        # Legend
        cv2.circle(canvas, (self._W - 180, self._H - 20), 7, (65, 120, 255), -1)
        cv2.putText(canvas, "No contact", (self._W - 168, self._H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)
        cv2.circle(canvas, (self._W - 90, self._H - 20), 7, (0, 210, 255), -1)
        cv2.putText(canvas, "Contact", (self._W - 78, self._H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)

        return canvas

    def _stamp_time(self, canvas: np.ndarray, frame_idx: int, fps: float):
        ts = frame_idx / max(fps, 1e-6)
        m, s = divmod(int(ts), 60)
        cv2.rectangle(canvas, (self._W - 210, self._H - 35),
                      (self._W - 5, self._H - 5), (10, 10, 10), -1)
        cv2.putText(canvas, f"Time  {m:02d}:{s:02d}",
                    (self._W - 200, self._H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)
