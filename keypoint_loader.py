"""
Keypoint & Action Data Loader
==============================
Loads pre-extracted 2D/3D keypoints and ASFormer action recognition results
from JSON files into NumPy arrays for efficient frame-by-frame access.

Data format (from external SAM3D project):
  - 2d_points.json  → 70-joint 2D skeleton per frame  (resized 640×360)
  - 3d_points.json  → 70-joint 3D skeleton per frame  (world coordinates)
  - full_results.json → ASFormer action windows with confidence, speed, power
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameData2D:
    """2D keypoint data for a single frame."""
    frame: int
    track_id: str
    bbox: np.ndarray            # (4,) — [x1, y1, x2, y2] in original resolution
    joints_2d: np.ndarray       # (70, 2) — x, y in original resolution
    original_width: int
    original_height: int


@dataclass
class FrameData3D:
    """3D keypoint data for a single frame."""
    frame: int
    track_id: str
    bbox: np.ndarray                    # (4,)
    shared_space_coords: np.ndarray     # (70, 3) — world coordinates
    normalized_coords: np.ndarray       # (70, 3) — body-centered normalized
    focal_normalized_coords: Optional[np.ndarray] = None  # (70, 3)


@dataclass
class ActionEvent:
    """Single detected action from ASFormer."""
    fighter_type: str
    action: str
    confidence: float
    frame: int
    window_start: int
    window_end: int
    timestamp_seconds: float
    target: str
    is_significant: bool
    speed_kmh: float
    power_watts: float
    model_used: str


class KeypointLoader:
    """
    Loads and indexes keypoint data from JSON files.

    Usage:
        loader = KeypointLoader()
        frames_2d = loader.load_2d("2d_points.json")
        frames_3d = loader.load_3d("3d_points.json")
        actions   = loader.load_actions("full_results.json")
    """

    def load_2d(self, path: str) -> dict[int, FrameData2D]:
        """
        Load 2D keypoints, scaling from resized (640×360) to original resolution.

        Returns: dict mapping frame_number → FrameData2D
        """
        print(f"[Loader] Loading 2D keypoints from {path} …")
        with open(path, "r") as f:
            raw = json.load(f)

        frames: dict[int, FrameData2D] = {}
        total_entries = 0

        for track_id, entries in raw.items():
            for entry in entries:
                frame_num = entry["frame"]
                dims = entry.get("frame_dims", {})
                orig_w = dims.get("original_width", 1920)
                orig_h = dims.get("original_height", 1080)
                resized_w = dims.get("resized_width", 640)
                resized_h = dims.get("resized_height", 360)

                # Scale factors from resized → original
                sx = orig_w / resized_w
                sy = orig_h / resized_h

                joints_raw = np.array(entry["joints_2d"], dtype=np.float64)  # (70, 2)
                joints_orig = joints_raw.copy()
                joints_orig[:, 0] *= sx
                joints_orig[:, 1] *= sy

                bbox = np.array(entry["bbox"], dtype=np.float64)  # already original res

                frames[frame_num] = FrameData2D(
                    frame=frame_num,
                    track_id=track_id,
                    bbox=bbox,
                    joints_2d=joints_orig,
                    original_width=orig_w,
                    original_height=orig_h,
                )
                total_entries += 1

        print(f"[Loader] Loaded {total_entries} 2D frames "
              f"(frames {min(frames.keys())}–{max(frames.keys())})")
        return frames

    def load_3d(self, path: str) -> dict[int, FrameData3D]:
        """
        Load 3D keypoints from JSON.

        Returns: dict mapping frame_number → FrameData3D
        """
        print(f"[Loader] Loading 3D keypoints from {path} …")
        with open(path, "r") as f:
            raw = json.load(f)

        frames: dict[int, FrameData3D] = {}
        total_entries = 0

        for track_id, entries in raw.items():
            for entry in entries:
                frame_num = entry["frame"]
                bbox = np.array(entry["bbox"], dtype=np.float64)

                shared = np.array(entry["shared_space_coords"], dtype=np.float64)
                normalized = np.array(entry["normalized_coords"], dtype=np.float64)
                focal = None
                if "focal_normalized_coords" in entry:
                    focal = np.array(entry["focal_normalized_coords"], dtype=np.float64)

                frames[frame_num] = FrameData3D(
                    frame=frame_num,
                    track_id=track_id,
                    bbox=bbox,
                    shared_space_coords=shared,
                    normalized_coords=normalized,
                    focal_normalized_coords=focal,
                )
                total_entries += 1

        print(f"[Loader] Loaded {total_entries} 3D frames "
              f"(frames {min(frames.keys())}–{max(frames.keys())})")
        return frames

    def load_actions(self, path: str) -> list[ActionEvent]:
        """
        Load ASFormer action recognition results.

        Returns: list of ActionEvent sorted by frame number
        """
        print(f"[Loader] Loading action results from {path} …")
        with open(path, "r") as f:
            raw = json.load(f)

        actions: list[ActionEvent] = []
        for entry in raw.get("actions", []):
            speed = entry.get("speed_estimation", {}).get("estimated_speed_kmh", 0.0)
            power = entry.get("power_estimation", {}).get("estimated_power_watts", 0.0)

            actions.append(ActionEvent(
                fighter_type=entry.get("fighter_type", ""),
                action=entry["action"],
                confidence=entry["confidence"],
                frame=entry["frame"],
                window_start=entry["window_start"],
                window_end=entry["window_end"],
                timestamp_seconds=entry.get("timestamp_seconds", 0.0),
                target=entry.get("target", "Head"),
                is_significant=entry.get("is_significant", False),
                speed_kmh=speed,
                power_watts=power,
                model_used=entry.get("model_used", ""),
            ))

        actions.sort(key=lambda a: a.frame)
        print(f"[Loader] Loaded {len(actions)} action events "
              f"(types: {sorted(set(a.action for a in actions))})")
        return actions

    # ── Utility methods ──────────────────────────────────────────────────────

    @staticmethod
    def get_frame_range(
        frames: dict[int, object],
        start: int,
        end: int,
    ) -> list[int]:
        """Return sorted list of frame numbers available in [start, end]."""
        return sorted(f for f in frames if start <= f <= end)

    @staticmethod
    def extract_joint_trajectory(
        frames_3d: dict[int, FrameData3D],
        frame_range: list[int],
        joint_idx: int,
        use_normalized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract 3D trajectory for a specific joint across frames.

        Returns:
            frame_nums: (N,) array of frame numbers
            positions:  (N, 3) array of 3D positions
        """
        nums, positions = [], []
        for f in frame_range:
            if f not in frames_3d:
                continue
            data = frames_3d[f]
            coords = data.normalized_coords if use_normalized else data.shared_space_coords
            if joint_idx < len(coords):
                nums.append(f)
                positions.append(coords[joint_idx])

        return np.array(nums), np.array(positions) if positions else np.zeros((0, 3))

    @staticmethod
    def extract_joint_trajectory_2d(
        frames_2d: dict[int, FrameData2D],
        frame_range: list[int],
        joint_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D trajectory for a specific joint across frames.

        Returns:
            frame_nums: (N,) array of frame numbers
            positions:  (N, 2) array of 2D positions
        """
        nums, positions = [], []
        for f in frame_range:
            if f not in frames_2d:
                continue
            data = frames_2d[f]
            if joint_idx < len(data.joints_2d):
                nums.append(f)
                positions.append(data.joints_2d[joint_idx])

        return np.array(nums), np.array(positions) if positions else np.zeros((0, 2))
