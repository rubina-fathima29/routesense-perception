"""ROUTESENSE Member 1 perception entry point.

This script combines object detection, per-frame vehicle counting, and
ByteTrack vehicle tracking. It intentionally does not include Member 2 or
Member 3 intelligence, risk, or backend features.
"""

import json
from pathlib import Path

import cv2

from src.counting.vehicle_counter import VEHICLE_CLASSES, VehicleCounter
from src.detection.pedestrian_detector import PedestrianDetector
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.byte_tracker import VehicleByteTracker


PROJECT_ROOT = Path(__file__).resolve().parent


def find_input_video() -> Path:
    """Return the requested sample video, including a hidden-extension fallback."""
    input_directory = PROJECT_ROOT / "input"
    requested_path = input_directory / "sample_bus_video.mp4"
    if requested_path.is_file():
        return requested_path

    duplicate_extension_paths = sorted(input_directory.glob("sample_bus_video.mp4*"))
    if duplicate_extension_paths:
        return duplicate_extension_paths[0]

    raise FileNotFoundError(
        "Input video not found. Place sample_bus_video.mp4 in the input directory."
    )


def main() -> None:
    """Run the complete Member 1 perception pipeline over the sample video."""
    video_path = find_input_video()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    # Initialize each Member 1 component only once for the complete video.
    vehicle_detector = VehicleDetector()
    pedestrian_detector = PedestrianDetector()
    vehicle_counter = VehicleCounter()
    vehicle_tracker = VehicleByteTracker()

    print(f"Processing video: {video_path}")
    print(f"Video FPS: {fps}")

    frame_number = 0
    total_vehicle_detections = 0
    total_pedestrian_detections = 0
    class_visible_detection_totals = {vehicle_class: 0 for vehicle_class in VEHICLE_CLASSES}
    unique_track_ids: set[int] = set()
    max_simultaneously_tracked = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number += 1
        timestamp = round((frame_number - 1) / fps, 3)

        vehicle_detections = vehicle_detector.detect(frame)
        pedestrian_detections = pedestrian_detector.detect(frame)
        frame_vehicle_counts = vehicle_counter.count_current_visible(vehicle_detections)
        active_tracks = vehicle_tracker.update(
            vehicle_detections,
            frame_number=frame_number,
            timestamp=timestamp,
        )

        total_vehicle_detections += len(vehicle_detections)
        total_pedestrian_detections += len(pedestrian_detections)
        for vehicle_class, count in frame_vehicle_counts.items():
            class_visible_detection_totals[vehicle_class] += count
        unique_track_ids.update(track["track_id"] for track in active_tracks)
        max_simultaneously_tracked = max(max_simultaneously_tracked, len(active_tracks))

        if frame_number % 30 == 0:
            print(
                f"Frame {frame_number}: vehicles={len(vehicle_detections)}, "
                f"pedestrians={len(pedestrian_detections)}, "
                f"active_tracks={len(active_tracks)}"
            )

    capture.release()

    if frame_number == 0:
        raise RuntimeError("The input video contains no readable frames.")

    summary = {
        "input_video": str(video_path.relative_to(PROJECT_ROOT)),
        "total_frames_processed": frame_number,
        "total_vehicle_detections": total_vehicle_detections,
        "total_pedestrian_detections": total_pedestrian_detections,
        "vehicle_counts_by_class": class_visible_detection_totals,
        "vehicle_count_definition": "current visible vehicle detections in each frame",
        "total_unique_track_ids": len(unique_track_ids),
        "maximum_simultaneously_tracked_vehicles": max_simultaneously_tracked,
    }

    output_path = PROJECT_ROOT / "output" / "member1_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMember 1 perception summary")
    print(f"Total frames processed: {frame_number}")
    print(f"Total vehicle detections: {total_vehicle_detections}")
    print(f"Total pedestrian detections: {total_pedestrian_detections}")
    print(f"Vehicle counts by class: {class_visible_detection_totals}")
    print(f"Total unique track IDs: {len(unique_track_ids)}")
    print(f"Maximum simultaneously tracked vehicles: {max_simultaneously_tracked}")
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
