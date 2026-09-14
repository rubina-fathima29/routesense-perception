"""ByteTrack wrapper for stable per-vehicle identities and position histories."""

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


VEHICLE_CLASS_TO_COCO_ID = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}
COCO_ID_TO_VEHICLE_CLASS = {
    class_id: class_name for class_name, class_id in VEHICLE_CLASS_TO_COCO_ID.items()
}


class _DetectionBatch:
    """Small adapter that provides ByteTrack with the detector's box data."""

    def __init__(self, xywh: np.ndarray, confidence: np.ndarray, class_ids: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = confidence
        self.cls = class_ids

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, indices: Any) -> "_DetectionBatch":
        return _DetectionBatch(self.xywh[indices], self.conf[indices], self.cls[indices])


class VehicleByteTracker:
    """Track vehicle detections with ByteTrack and retain their centroid history.

    The tracker is deliberately independent of YOLO: callers provide the
    dictionary detections emitted by ``VehicleDetector`` for each frame.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        settings = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self.tracker = BYTETracker(settings)
        self.position_history: dict[int, list[dict[str, Any]]] = {}

    def update(
        self,
        detections: Iterable[Mapping[str, object]],
        frame_number: int,
        timestamp: float | None = None,
    ) -> list[dict[str, Any]]:
        """Update ByteTrack for one frame and return its active vehicle tracks.

        ``frame_number`` and ``timestamp`` are stored alongside every centroid
        so each track's history can later be consumed by other ROUTESENSE
        components.
        """
        detection_batch = self._to_bytetrack_batch(detections)
        tracked_rows = self.tracker.update(detection_batch)

        tracks: list[dict[str, Any]] = []
        for row in tracked_rows:
            x1, y1, x2, y2, track_id, confidence, class_id, _ = row.tolist()
            class_name = COCO_ID_TO_VEHICLE_CLASS.get(int(class_id))
            if class_name is None:
                continue

            bbox = [int(x1), int(y1), int(x2), int(y2)]
            centroid = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            track_id = int(track_id)
            position = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "centroid": centroid,
            }
            self.position_history.setdefault(track_id, []).append(position)

            tracks.append(
                {
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 4),
                    "bbox": bbox,
                    "centroid": centroid,
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "position_history": list(self.position_history[track_id]),
                }
            )
        return tracks

    @staticmethod
    def _to_bytetrack_batch(
        detections: Iterable[Mapping[str, object]],
    ) -> _DetectionBatch:
        """Convert detector dictionaries to ByteTrack's ``xywh/conf/class`` form."""
        xywh_rows: list[list[float]] = []
        confidences: list[float] = []
        class_ids: list[float] = []

        for detection in detections:
            class_name = detection.get("class_name")
            if class_name not in VEHICLE_CLASS_TO_COCO_ID:
                continue
            x1, y1, x2, y2 = detection["bbox"]
            xywh_rows.append(
                [
                    (float(x1) + float(x2)) / 2,
                    (float(y1) + float(y2)) / 2,
                    float(x2) - float(x1),
                    float(y2) - float(y1),
                ]
            )
            confidences.append(float(detection["confidence"]))
            class_ids.append(float(VEHICLE_CLASS_TO_COCO_ID[class_name]))

        return _DetectionBatch(
            np.asarray(xywh_rows, dtype=np.float32).reshape(-1, 4),
            np.asarray(confidences, dtype=np.float32),
            np.asarray(class_ids, dtype=np.float32),
        )
