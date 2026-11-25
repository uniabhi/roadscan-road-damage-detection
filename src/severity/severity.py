"""
Severity scoring module for road damage detection.

Computes a severity score based on detected damage types, sizes, and confidence.
This is a transparent heuristic, not an engineering standard.
"""

from typing import Dict, List, Literal

import numpy as np

# Severity weights per damage class (design choice, not engineering standard)
SEVERITY_WEIGHTS = {
    "D40": 3.0,  # Pothole - highest severity
    "D20": 2.0,  # Alligator crack
    "D10": 1.0,  # Transverse crack
    "D00": 1.0,  # Longitudinal crack
}

# Class ID to name mapping
ID_TO_CLASS = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def compute_severity(detections: List[Dict],
                     img_w: int,
                     img_h: int,
                     conf_min: float = 0.30,
                     weights: Dict[str, float] = None) -> float:
    """
    Compute severity score from detections.

    The score considers:
    - Damage type (potholes weighted higher than cracks)
    - Damage area (larger = more severe)
    - Detection confidence

    Args:
        detections: List of detection dicts with keys:
                   - cls: class name ("D00", "D10", "D20", "D40") or class_id
                   - conf: confidence score
                   - x1, y1, x2, y2: bounding box coordinates
        img_w: Image width in pixels
        img_h: Image height in pixels
        conf_min: Minimum confidence to include
        weights: Custom severity weights per class

    Returns:
        Severity score in [0, 1]
    """
    if weights is None:
        weights = SEVERITY_WEIGHTS

    if img_w <= 0 or img_h <= 0:
        return 0.0

    total_score = 0.0
    img_area = img_w * img_h

    for det in detections:
        # Get confidence
        conf = det.get("conf", 0.0)
        if conf < conf_min:
            continue

        # Get class name
        cls = det.get("cls", None)
        if cls is None:
            cls_id = det.get("class_id", det.get("cls_id", None))
            if cls_id is not None and cls_id in ID_TO_CLASS:
                cls = ID_TO_CLASS[cls_id]

        if cls is None or cls not in weights:
            continue

        # Get bounding box
        x1 = det.get("x1", det.get("xmin", 0))
        y1 = det.get("y1", det.get("ymin", 0))
        x2 = det.get("x2", det.get("xmax", 0))
        y2 = det.get("y2", det.get("ymax", 0))

        # Compute normalized area
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        normalized_area = box_area / img_area

        # Add weighted contribution
        weight = weights.get(cls, 1.0)
        total_score += weight * normalized_area * conf

    # Normalize to [0, 1]
    return min(total_score, 1.0)


def get_severity_band(score: float,
                      low_threshold: float = 0.05,
                      high_threshold: float = 0.15) -> Literal["low", "medium", "high"]:
    """
    Convert severity score to band.

    Args:
        score: Severity score from compute_severity()
        low_threshold: Threshold for low/medium boundary
        high_threshold: Threshold for medium/high boundary

    Returns:
        Severity band: "low", "medium", or "high"
    """
    if score < low_threshold:
        return "low"
    elif score < high_threshold:
        return "medium"
    else:
        return "high"


def compute_percentile_thresholds(scores: List[float],
                                  low_percentile: float = 60,
                                  high_percentile: float = 90) -> Dict[str, float]:
    """
    Compute severity band thresholds from percentiles of validation scores.

    Args:
        scores: List of severity scores from validation set
        low_percentile: Percentile for low/medium boundary
        high_percentile: Percentile for medium/high boundary

    Returns:
        Dict with "low" and "high" thresholds
    """
    if not scores:
        return {"low": 0.05, "high": 0.15}

    scores_array = np.array(scores)
    low_threshold = float(np.percentile(scores_array, low_percentile))
    high_threshold = float(np.percentile(scores_array, high_percentile))

    return {
        "low": low_threshold,
        "high": high_threshold
    }


def summarize_detections(detections: List[Dict],
                         conf_min: float = 0.30) -> Dict:
    """
    Summarize detections by class.

    Args:
        detections: List of detection dicts
        conf_min: Minimum confidence to include

    Returns:
        Dict with counts per class and total
    """
    counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    total = 0

    for det in detections:
        conf = det.get("conf", 0.0)
        if conf < conf_min:
            continue

        cls = det.get("cls", None)
        if cls is None:
            cls_id = det.get("class_id", det.get("cls_id", None))
            if cls_id is not None and cls_id in ID_TO_CLASS:
                cls = ID_TO_CLASS[cls_id]

        if cls and cls in counts:
            counts[cls] += 1
            total += 1

    return {
        "counts": counts,
        "total": total,
        "has": {cls: count > 0 for cls, count in counts.items()}
    }


if __name__ == "__main__":
    # Example usage
    example_detections = [
        {"cls": "D40", "conf": 0.85, "x1": 100, "y1": 200, "x2": 200, "y2": 300},
        {"cls": "D20", "conf": 0.72, "x1": 300, "y1": 400, "x2": 450, "y2": 500},
    ]

    img_w, img_h = 640, 480
    score = compute_severity(example_detections, img_w, img_h)
    band = get_severity_band(score)
    summary = summarize_detections(example_detections)

    print(f"Severity score: {score:.4f}")
    print(f"Severity band: {band}")
    print(f"Detection summary: {summary}")
