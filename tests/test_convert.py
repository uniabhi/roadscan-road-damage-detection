"""Tests for data conversion module."""

from src.data.audit import KEEP_CLASSES, voc_to_yolo_lines


class TestVocToYoloLines:
    """Tests for VOC to YOLO conversion."""

    def test_all_four_classes(self):
        """All four damage classes should convert correctly."""
        W, H = 640, 480
        boxes = [
            (100, 100, 200, 200, "D00"),
            (200, 100, 300, 200, "D10"),
            (300, 100, 400, 200, "D20"),
            (400, 100, 500, 200, "D40"),
        ]

        lines = voc_to_yolo_lines(W, H, boxes)

        assert len(lines) == 4

        # Check class IDs are correct
        class_ids = [int(line.split()[0]) for line in lines]
        assert class_ids == [0, 1, 2, 3]

    def test_degenerate_box_filtered(self):
        """Degenerate boxes should be filtered out."""
        W, H = 640, 480
        boxes = [
            (100, 100, 200, 200, "D00"),  # Valid
            (100, 100, 101, 200, "D00"),  # Too narrow
            (100, 100, 200, 101, "D00"),  # Too short
        ]

        lines = voc_to_yolo_lines(W, H, boxes)

        assert len(lines) == 1

    def test_normalized_values_in_range(self):
        """All output values should be in [0, 1]."""
        W, H = 640, 480
        boxes = [(100, 100, 200, 200, "D00")]

        lines = voc_to_yolo_lines(W, H, boxes)
        parts = lines[0].split()

        for val in parts[1:]:  # Skip class ID
            v = float(val)
            assert 0.0 <= v <= 1.0


class TestKeepClasses:
    """Tests for class definitions."""

    def test_four_classes_defined(self):
        """Should have exactly 4 damage classes."""
        assert len(KEEP_CLASSES) == 4

    def test_class_ids_are_sequential(self):
        """Class IDs should be 0, 1, 2, 3."""
        assert KEEP_CLASSES == {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
