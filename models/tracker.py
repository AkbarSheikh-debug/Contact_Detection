"""
Simple IoU-based person tracker (lightweight SORT-like).
Assigns persistent track IDs to detections across frames.
"""
import numpy as np
from dataclasses import dataclass, field


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    age: int = 0
    missed: int = 0
    history: list = field(default_factory=list)   # last N centroids


class PersonTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 20):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to existing tracks via greedy IoU matching.
        Assigns track_id in-place and returns the updated detection list.
        """
        boxes = [d["bbox"] for d in detections]

        # Build cost matrix (neg IoU)
        matched_tracks = set()
        matched_dets = set()

        if self._tracks and boxes:
            iou_matrix = np.zeros((len(self._tracks), len(boxes)))
            for ti, track in enumerate(self._tracks):
                for di, box in enumerate(boxes):
                    iou_matrix[ti, di] = _iou(track.bbox, box)

            while True:
                r, c = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[r, c] < self.iou_threshold:
                    break
                if r in matched_tracks or c in matched_dets:
                    iou_matrix[r, c] = 0
                    continue
                # Match
                self._tracks[r].bbox = boxes[c]
                self._tracks[r].age += 1
                self._tracks[r].missed = 0
                cx = (boxes[c][0] + boxes[c][2]) / 2
                cy = (boxes[c][1] + boxes[c][3]) / 2
                self._tracks[r].history.append((cx, cy))
                if len(self._tracks[r].history) > 30:
                    self._tracks[r].history.pop(0)
                detections[c]["track_id"] = self._tracks[r].track_id
                matched_tracks.add(r)
                matched_dets.add(c)
                iou_matrix[r, :] = 0
                iou_matrix[:, c] = 0

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_dets:
                track = Track(track_id=self._next_id, bbox=det["bbox"])
                self._next_id += 1
                self._tracks.append(track)
                det["track_id"] = track.track_id

        # Age out missed tracks
        alive_ids = {d["track_id"] for d in detections}
        new_tracks = []
        for t in self._tracks:
            if t.track_id in alive_ids:
                new_tracks.append(t)
            else:
                t.missed += 1
                if t.missed < self.max_missed:
                    new_tracks.append(t)
        self._tracks = new_tracks

        return detections

    def get_track(self, track_id: int) -> Track | None:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None
