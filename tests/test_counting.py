"""Run per-frame visible vehicle counting on the supplied sample bus video."""

import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.counting.vehicle_counter import VEHICLE_CLASSES, VehicleCounter
from src.detection.vehicle_detector import VehicleDetector


def find_sample_video() -> Path:
    """Locate the supplied sample video, including a hidden-extension duplicate."""
    input_directory = PROJECT_ROOT / "input"
    exact_path = input_directory / "sample_bus_video.mp4"
    if exact_path.is_file():
        return exact_path

    matching_paths = sorted(input_directory.glob("sample_bus_video.mp4*"))
    if matching_paths:
        return matching_paths[0]
    raise FileNotFoundError(f"Sample video not found in: {input_directory}")


def empty_class_statistics() -> dict[str, dict[str, int]]:
    """Create counters for aggregate and maximum per-frame class visibility."""
    return {
        vehicle_class: {
            "total_visible_detections_across_frames": 0,
            "max_visible_detections_in_one_frame": 0,
        }
        for vehicle_class in VEHICLE_CLASSES
    }


def main() -> None:
    video_path = find_sample_video()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    detector = VehicleDetector()
    counter = VehicleCounter()
    class_statistics = empty_class_statistics()
    frame_count = 0
    visible_vehicle_total = 0
    max_visible_vehicle_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        vehicle_detections = detector.detect(frame)
        frame_counts = counter.count_current_visible(vehicle_detections)
        current_visible_total = sum(frame_counts.values())

        frame_count += 1
        visible_vehicle_total += current_visible_total
        max_visible_vehicle_count = max(max_visible_vehicle_count, current_visible_total)

        for vehicle_class, count in frame_counts.items():
            statistics = class_statistics[vehicle_class]
            statistics["total_visible_detections_across_frames"] += count
            statistics["max_visible_detections_in_one_frame"] = max(
                statistics["max_visible_detections_in_one_frame"], count
            )

    capture.release()

    if frame_count == 0:
        raise RuntimeError("The sample video contains no readable frames.")

    for statistics in class_statistics.values():
        statistics["average_visible_detections_per_frame"] = round(
            statistics["total_visible_detections_across_frames"] / frame_count, 3
        )

    summary = {
        "count_definition": "current visible vehicle detections in each frame",
        "note": "These are not unique vehicles across the full video.",
        "input_video": str(video_path.relative_to(PROJECT_ROOT)),
        "total_frames_processed": frame_count,
        "average_visible_vehicle_count_per_frame": round(
            visible_vehicle_total / frame_count, 3
        ),
        "maximum_visible_vehicle_count": max_visible_vehicle_count,
        "class_statistics": class_statistics,
    }

    output_path = PROJECT_ROOT / "output" / "vehicle_count_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Total frames processed: {frame_count}")
    print(
        "Average visible vehicle count per frame: "
        f"{summary['average_visible_vehicle_count_per_frame']}"
    )
    print(f"Maximum visible vehicle count: {max_visible_vehicle_count}")
    for vehicle_class, statistics in class_statistics.items():
        print(
            f"{vehicle_class}: total={statistics['total_visible_detections_across_frames']}, "
            f"average/frame={statistics['average_visible_detections_per_frame']}, "
            f"max/frame={statistics['max_visible_detections_in_one_frame']}"
        )
    print(f"JSON summary saved to: {output_path}")


if __name__ == "__main__":
    main()
