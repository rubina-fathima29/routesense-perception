"""Run ByteTrack vehicle tracking on the ROUTESENSE sample bus video."""

import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.byte_tracker import VEHICLE_CLASS_TO_COCO_ID, VehicleByteTracker


def find_sample_video() -> Path:
    """Find the supplied sample, accounting for a duplicated hidden extension."""
    input_directory = PROJECT_ROOT / "input"
    exact_path = input_directory / "sample_bus_video.mp4"
    if exact_path.is_file():
        return exact_path
    matching_paths = sorted(input_directory.glob("sample_bus_video.mp4*"))
    if matching_paths:
        return matching_paths[0]
    raise FileNotFoundError(f"Sample video not found in: {input_directory}")


def draw_tracks(frame, tracks):
    """Draw the active ByteTrack identity and confidence for each vehicle."""
    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        cx, cy = track["centroid"]
        label = (
            f"{track['class_name']} {track['confidence']:.2f} "
            f"ID:{track['track_id']}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)


def main() -> None:
    video_path = find_sample_video()
    output_video_path = PROJECT_ROOT / "output" / "tracking_frames" / "tracking_test.mp4"
    output_jsonl_path = PROJECT_ROOT / "output" / "tracking_results.jsonl"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create tracking video: {output_video_path}")

    detector = VehicleDetector()
    tracker = VehicleByteTracker()
    frame_number = 0
    unique_track_ids: set[int] = set()
    first_class_by_track_id: dict[int, str] = {}
    max_simultaneously_tracked = 0

    with output_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            timestamp = round((frame_number - 1) / fps, 3)
            vehicle_detections = detector.detect(frame)
            tracks = tracker.update(vehicle_detections, frame_number, timestamp)

            for track in tracks:
                unique_track_ids.add(track["track_id"])
                first_class_by_track_id.setdefault(track["track_id"], track["class_name"])
            max_simultaneously_tracked = max(max_simultaneously_tracked, len(tracks))

            frame_record = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "tracked_vehicles": tracks,
            }
            jsonl_file.write(json.dumps(frame_record) + "\n")

            annotated_frame = frame.copy()
            draw_tracks(annotated_frame, tracks)
            writer.write(annotated_frame)

    capture.release()
    writer.release()

    if frame_number == 0:
        raise RuntimeError("The sample video contains no readable frames.")

    histories_generated = bool(tracker.position_history) and all(
        history for history in tracker.position_history.values()
    )
    print(f"Total frames processed: {frame_number}")
    print(f"Total unique track IDs observed: {len(unique_track_ids)}")
    for class_name in VEHICLE_CLASS_TO_COCO_ID:
        track_count = sum(
            observed_class == class_name
            for observed_class in first_class_by_track_id.values()
        )
        print(f"Unique {class_name} tracks: {track_count}")
    print(f"Maximum simultaneously tracked vehicles: {max_simultaneously_tracked}")
    print(f"Position histories generated: {histories_generated}")
    print(f"Tracking video saved to: {output_video_path}")
    print(f"Tracking JSONL saved to: {output_jsonl_path}")


if __name__ == "__main__":
    main()
