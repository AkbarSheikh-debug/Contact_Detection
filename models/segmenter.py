"""
SAM (Segment Anything Model) segmenter.
Used as the contact decoder — same role as in Pi-HOC's contact decoder (Sec 3.4).
Takes bounding box prompts from YOLO detections and produces precise body masks.
Falls back to bounding-box fill when SAM checkpoint is unavailable.
"""
import os
import numpy as np

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False

from config import SAM_CHECKPOINT, SAM_MODEL_TYPE


class SAMSegmenter:
    """SAM ViT-B segmenter with bounding-box prompt interface."""

    def __init__(self, checkpoint_path: str = SAM_CHECKPOINT, model_type: str = SAM_MODEL_TYPE):
        self.available = False
        self.predictor = None

        if not _SAM_AVAILABLE:
            print("  [SAM] segment_anything not installed — skipping SAM.")
            return

        if not os.path.exists(checkpoint_path):
            print(f"  [SAM] Checkpoint not found at {checkpoint_path} — skipping SAM.")
            return

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [SAM] Loading {model_type} from {checkpoint_path} on {device} ...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.available = True
        self._current_image_set = False

    def set_image(self, frame_rgb: np.ndarray):
        """Encode image features once per frame (amortised over multiple queries)."""
        if not self.available:
            return
        self.predictor.set_image(frame_rgb)
        self._current_image_set = True

    def segment(self, bbox_xyxy: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        Segment person inside bbox.
        Returns (mask HxW bool, score) or (None, 0.0) on failure.
        """
        if not self.available or not self._current_image_set:
            return None, 0.0

        box = bbox_xyxy.astype(float)
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx], float(scores[best_idx])

    def reset(self):
        self._current_image_set = False
