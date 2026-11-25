"""
Data audit module for RDD2022 dataset.

This module provides functions to audit the RDD2022 dataset for:
- Size mismatches between XML and image dimensions
- Out-of-bounds bounding boxes
- Degenerate boxes (too small)
- Duplicate boxes (high IoU overlap)
- Non-standard class labels
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image

# Standard RDD2022 challenge classes (4 damage types)
KEEP_CLASSES = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}

# Human-readable class names
CLASS_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole"
}


@dataclass
class AuditResult:
    """Results from auditing a single image-annotation pair."""
    image_path: str
    xml_path: str
    country: str
    issues: List[str] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float, str]] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)
    has_valid_annotation: bool = True


@dataclass
class AuditSummary:
    """Summary statistics for the entire dataset audit."""
    total_images: int = 0
    total_annotations: int = 0
    images_with_issues: int = 0
    issue_counts: Dict[str, int] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)
    country_counts: Dict[str, int] = field(default_factory=dict)
    country_class_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    other_classes_found: Set[str] = field(default_factory=set)


def compute_iou(box_a: Tuple[float, float, float, float],
                box_b: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: (x1, y1, x2, y2) format
        box_b: (x1, y1, x2, y2) format

    Returns:
        IoU value between 0 and 1
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    inter_area = inter_width * inter_height

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def audit_one(xml_path: str, img_path: str,
              iou_threshold: float = 0.95,
              min_box_size: int = 2) -> AuditResult:
    """
    Audit a single image-annotation pair for issues.

    Args:
        xml_path: Path to Pascal VOC XML annotation file
        img_path: Path to corresponding image file
        iou_threshold: IoU threshold for detecting duplicate boxes
        min_box_size: Minimum width/height for valid boxes

    Returns:
        AuditResult containing issues found and box information
    """
    result = AuditResult(
        image_path=str(img_path),
        xml_path=str(xml_path),
        country=_extract_country_from_path(img_path)
    )

    # Check if files exist
    if not Path(xml_path).exists():
        result.issues.append("missing_xml")
        result.has_valid_annotation = False
        return result

    if not Path(img_path).exists():
        result.issues.append("missing_image")
        result.has_valid_annotation = False
        return result

    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        result.issues.append(f"xml_parse_error:{str(e)}")
        result.has_valid_annotation = False
        return result

    # Get dimensions from XML
    size_elem = root.find("size")
    if size_elem is None:
        result.issues.append("missing_size_in_xml")
        xml_width, xml_height = 0, 0
    else:
        xml_width = int(size_elem.findtext("width", "0"))
        xml_height = int(size_elem.findtext("height", "0"))

    # Verify image dimensions
    try:
        with Image.open(img_path) as im:
            actual_width, actual_height = im.size
            result.image_size = (actual_width, actual_height)

            if (xml_width, xml_height) != (actual_width, actual_height):
                if xml_width > 0 and xml_height > 0:
                    result.issues.append("size_mismatch")
    except Exception as e:
        result.issues.append(f"image_read_error:{str(e)}")
        return result

    # Use actual image dimensions for validation
    W, H = result.image_size

    # Parse all objects/boxes
    boxes = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None:
            result.issues.append("missing_class_name")
            continue

        name = name_elem.text.strip() if name_elem.text else ""

        bb = obj.find("bndbox")
        if bb is None:
            result.issues.append(f"missing_bbox:{name}")
            continue

        try:
            x1 = float(bb.findtext("xmin", "0"))
            y1 = float(bb.findtext("ymin", "0"))
            x2 = float(bb.findtext("xmax", "0"))
            y2 = float(bb.findtext("ymax", "0"))
        except ValueError:
            result.issues.append(f"invalid_bbox_coords:{name}")
            continue

        # Check if class is in our keep list
        if name not in KEEP_CLASSES:
            result.issues.append(f"other_class:{name}")

        # Check for out-of-bounds boxes
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            result.issues.append("out_of_bounds")

        # Check for degenerate boxes (too small)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < min_box_size or box_height < min_box_size:
            result.issues.append("degenerate_box")

        boxes.append((x1, y1, x2, y2, name))

    result.boxes = boxes

    # Check for duplicate boxes (same class, high IoU)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][4] == boxes[j][4]:  # Same class
                iou = compute_iou(boxes[i][:4], boxes[j][:4])
                if iou > iou_threshold:
                    result.issues.append("duplicate_box")
                    break

    return result


def _extract_country_from_path(path: str) -> str:
    """Extract country name from file path."""
    path_parts = Path(path).parts
    for part in path_parts:
        if part in ["Japan", "India", "Czech", "Norway", "United_States",
                    "China_MotorBike", "China_Drone"]:
            return part
    return "unknown"


def voc_to_yolo_lines(W: int, H: int,
                      boxes: List[Tuple[float, float, float, float, str]],
                      clip_boxes: bool = True,
                      min_box_size: int = 2) -> List[str]:
    """
    Convert Pascal VOC bounding boxes to YOLO format.

    YOLO format: <class_id> <cx> <cy> <width> <height>
    where all values are normalized to [0, 1].
    """
    if W <= 0 or H <= 0:
        return []

    lines = []

    for x1, y1, x2, y2, name in boxes:
        # Skip non-standard classes
        if name not in KEEP_CLASSES:
            continue

        # Clip boxes to image boundaries
        if clip_boxes:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

        # Skip degenerate boxes
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue

        # Convert to YOLO format (normalized center, width, height)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H

        # Ensure values are in valid range
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        class_id = KEEP_CLASSES[name]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines
