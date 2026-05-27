"""
YOLOv8-pose based person detector.
Provides bounding boxes + 17 COCO keypoints per person.
"""
import numpy as np
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONF_THRESHOLD, YOLO_PERSON_CLASS


class PersonDetector:
    def __init__(self, model_path: str = YOLO_MODEL):
        print(f"  [Detector] Loading {model_path} ...")
        self.model = YOLO(model_path)
        self.model.fuse()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference and return list of person dicts:
          bbox       : [x1, y1, x2, y2]
          conf       : float
          keypoints  : {'points': (17,2) float32, 'confidence': (17,) float32}
                       or None if not available
          raw_id     : int  (detection index, replaced by tracker id downstream)
        """
        results = self.model(frame, verbose=False, classes=[YOLO_PERSON_CLASS])[0]
        persons = []

        if results.boxes is None:
            return persons

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            if conf < YOLO_CONF_THRESHOLD:
                continue

            bbox = box.xyxy[0].cpu().numpy().astype(int)

            keypoints = None
            if results.keypoints is not None and i < len(results.keypoints):
                kp_data = results.keypoints[i]
                pts = kp_data.xy[0].cpu().numpy()        # (17, 2)
                kp_conf = kp_data.conf[0].cpu().numpy()  # (17,)
                keypoints = {"points": pts, "confidence": kp_conf}

            persons.append({
                "bbox": bbox,
                "conf": conf,
                "keypoints": keypoints,
                "raw_id": i,
                "track_id": -1,   # filled by tracker
            })

        return persons
