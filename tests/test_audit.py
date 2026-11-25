"""Tests for the audit module."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from src.data.audit import audit_one, compute_iou, voc_to_yolo_lines


class TestComputeIOU:
    """Tests for IoU computation."""

    def test_iou_same_box(self):
        """Same boxes should have IoU of 1."""
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0."""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_iou_half_overlap(self):
        """Test partially overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 0, 150, 100)
        # Overlap is 50x100=5000, union is 10000+10000-5000=15000
        # IoU = 5000/15000 = 0.333
        assert compute_iou(box1, box2) == pytest.approx(1 / 3, rel=0.01)


class TestVocToYoloLines:
    """Tests for VOC to YOLO conversion."""

    def test_basic_conversion(self):
        """Test basic box conversion."""
        W, H = 640, 480
        boxes = [(100, 100, 200, 200, "D00")]  # 100x100 box at (100,100)

        lines = voc_to_yolo_lines(W, H, boxes)

        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"  # D00 = class 0

        # Center should be at (150, 150) normalized
        cx, cy = float(parts[1]), float(parts[2])
        assert cx == pytest.approx(150 / 640, rel=0.01)
        assert cy == pytest.approx(150 / 480, rel=0.01)

    def test_skip_non_standard_class(self):
        """Non-standard classes should be skipped."""
        W, H = 640, 480
        boxes = [(100, 100, 200, 200, "Repair")]

        lines = voc_to_yolo_lines(W, H, boxes)
        assert len(lines) == 0

    def test_clip_out_of_bounds(self):
        """Boxes should be clipped to image boundaries."""
        W, H = 640, 480
        boxes = [(-10, -10, 650, 490, "D00")]

        lines = voc_to_yolo_lines(W, H, boxes, clip_boxes=True)
        assert len(lines) == 1


class TestAuditOne:
    """Tests for single image audit."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_missing_xml(self, temp_dir):
        """Missing XML should be flagged."""
        img_path = temp_dir / "test.jpg"
        xml_path = temp_dir / "test.xml"

        # Create dummy image
        Image.new("RGB", (640, 480)).save(img_path)

        result = audit_one(str(xml_path), str(img_path))

        assert "missing_xml" in result.issues
        assert not result.has_valid_annotation

    def test_missing_image(self, temp_dir):
        """Missing image should be flagged."""
        xml_path = temp_dir / "test.xml"

        # Create minimal XML
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "test.jpg"
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        result = audit_one(str(xml_path), str(temp_dir / "test.jpg"))

        assert "missing_image" in result.issues

    def test_valid_annotation(self, temp_dir):
        """Valid annotation should have no issues."""
        img_path = temp_dir / "test.jpg"
        xml_path = temp_dir / "test.xml"

        # Create image
        img = Image.new("RGB", (640, 480))
        img.save(img_path)

        # Create XML
        root = ET.Element("annotation")
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "640"
        ET.SubElement(size, "height").text = "480"

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "D00"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = "100"
        ET.SubElement(bb, "ymin").text = "100"
        ET.SubElement(bb, "xmax").text = "200"
        ET.SubElement(bb, "ymax").text = "200"

        tree = ET.ElementTree(root)
        tree.write(xml_path)

        result = audit_one(str(xml_path), str(img_path))

        assert len(result.issues) == 0
        assert len(result.boxes) == 1
