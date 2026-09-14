"""YOLOv8 pedestrian detection for ROUTESENSE Member 1."""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


PERSON_CLASS = "person"


class PedestrianDetector:
    """Detect people in one image frame."""

    def __init__(self, model_path: str | Path | None = None, confidence: float = 0.25) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else project_root / "yolov8n.pt"
        self.confidence = confidence
        self.model = YOLO(str(self.model_path))
        self.person_class_id = next(
            class_id
            for class_id, class_name in self.model.names.items()
            if class_name == PERSON_CLASS
        )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return person detections in ``frame`` with pixel bounding boxes."""
        result = self.model.predict(
            source=frame,
            classes=[self.person_class_id],
            conf=self.confidence,
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                {
                    "class_name": PERSON_CLASS,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections
