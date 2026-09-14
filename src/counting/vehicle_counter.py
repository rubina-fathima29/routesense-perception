"""Per-frame visible vehicle counting for ROUTESENSE Member 1."""

from collections.abc import Iterable, Mapping


VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle")


class VehicleCounter:
    """Count the vehicle detections currently visible in one frame.

    The returned counts are per-frame visible detections. They are not unique
    vehicle counts across a video; stable unique counts require tracking.
    """

    def count_current_visible(
        self, detections: Iterable[Mapping[str, object]]
    ) -> dict[str, int]:
        """Return one visible-detection count for each supported vehicle class."""
        counts = {vehicle_class: 0 for vehicle_class in VEHICLE_CLASSES}
        for detection in detections:
            class_name = detection.get("class_name")
            if class_name in counts:
                counts[class_name] += 1
        return counts
