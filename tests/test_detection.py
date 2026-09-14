"""Run the ROUTESENSE object detectors on one local video or image input."""

from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.pedestrian_detector import PedestrianDetector
from src.detection.vehicle_detector import VehicleDetector


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def draw_detections(frame, detections, color):
    """Draw a labeled rectangle for every detection on an OpenCV frame."""
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def detect_in_frame(frame, vehicle_detector, pedestrian_detector):
    vehicles = vehicle_detector.detect(frame)
    pedestrians = pedestrian_detector.detect(frame)
    annotated_frame = frame.copy()
    draw_detections(annotated_frame, vehicles, (0, 255, 0))
    draw_detections(annotated_frame, pedestrians, (0, 165, 255))
    return annotated_frame, vehicles, pedestrians


def test_video(video_path, vehicle_detector, pedestrian_detector):
    output_path = PROJECT_ROOT / "output" / "annotated_video" / "detection_test.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create annotated video: {output_path}")

    frame_count = vehicle_count = pedestrian_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        annotated_frame, vehicles, pedestrians = detect_in_frame(
            frame, vehicle_detector, pedestrian_detector
        )
        writer.write(annotated_frame)
        frame_count += 1
        vehicle_count += len(vehicles)
        pedestrian_count += len(pedestrians)

    capture.release()
    writer.release()
    print(f"Annotated video saved to: {output_path}")
    return frame_count, vehicle_count, pedestrian_count


def test_image(image_path, vehicle_detector, pedestrian_detector):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    annotated_frame, vehicles, pedestrians = detect_in_frame(
        frame, vehicle_detector, pedestrian_detector
    )
    output_path = PROJECT_ROOT / "output" / "annotated_video" / "detection_test.jpg"
    if not cv2.imwrite(str(output_path), annotated_frame):
        raise RuntimeError(f"Could not save annotated image: {output_path}")
    print(f"Annotated image saved to: {output_path}")
    return 1, len(vehicles), len(pedestrians)


def main():
    input_directory = PROJECT_ROOT / "input"
    local_files = [path for path in input_directory.iterdir() if path.is_file()]
    videos = [path for path in local_files if path.suffix.lower() in VIDEO_SUFFIXES]
    images = [path for path in local_files if path.suffix.lower() in IMAGE_SUFFIXES]

    vehicle_detector = VehicleDetector()
    pedestrian_detector = PedestrianDetector()
    print("Vehicle and pedestrian detectors loaded successfully.")

    if videos:
        totals = test_video(videos[0], vehicle_detector, pedestrian_detector)
    elif images:
        totals = test_image(images[0], vehicle_detector, pedestrian_detector)
    else:
        print("No local video or image was found in input/.")
        print("Add a real input video to input/ to run object detection.")
        return

    frame_count, vehicle_count, pedestrian_count = totals
    print(f"Total frames processed: {frame_count}")
    print(f"Vehicle detections: {vehicle_count}")
    print(f"Pedestrian detections: {pedestrian_count}")


if __name__ == "__main__":
    main()
