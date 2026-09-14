"""Minimal check that the pretrained YOLOv8 nano model can be loaded."""

from ultralytics import YOLO


REQUIRED_CLASSES = {"car", "bus", "truck", "motorcycle", "person"}


def main() -> None:
    model = YOLO("yolov8n.pt")
    print("YOLOv8 nano model loaded successfully.")

    class_names = model.names
    print("Detected class names:")
    print(class_names)

    available_classes = set(class_names.values())
    missing_classes = REQUIRED_CLASSES - available_classes
    if missing_classes:
        raise RuntimeError(f"Required COCO classes are missing: {sorted(missing_classes)}")

    print(f"Required COCO classes are available: {sorted(REQUIRED_CLASSES)}")


if __name__ == "__main__":
    main()
