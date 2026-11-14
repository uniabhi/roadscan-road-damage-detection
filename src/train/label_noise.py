"""
Label noise detection using model predictions.

This module identifies potential label errors by comparing model predictions
with ground truth annotations.
"""

import json
from pathlib import Path
from typing import Dict, List
import random

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def compute_iou_yolo(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two YOLO format boxes (cx, cy, w, h)."""
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def find_missing_labels(model_path: str, images_dir: str, labels_dir: str,
                        conf_threshold: float = 0.6, iou_threshold: float = 0.1) -> List[Dict]:
    """Find potential missing labels (Flag A) - high confidence predictions with no GT overlap."""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed")
    
    model = YOLO(model_path)
    flagged = []
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in image_extensions]
    
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        gt_boxes = []
        
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append([float(x) for x in parts[1:5]])
        
        results = model.predict(str(img_file), conf=conf_threshold, verbose=False)
        
        for result in results:
            for box in result.boxes:
                pred_box = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                
                max_iou = 0.0
                for gt in gt_boxes:
                    iou = compute_iou_yolo(pred_box, gt)
                    max_iou = max(max_iou, iou)
                
                if max_iou < iou_threshold:
                    flagged.append({
                        "image": str(img_file),
                        "type": "missing_label",
                        "confidence": conf,
                        "box": pred_box,
                        "class_id": int(box.cls[0])
                    })
    
    return flagged


def find_hard_labels(model_path: str, images_dir: str, labels_dir: str,
                     conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> List[Dict]:
    """Find potential wrong or hard labels (Flag B) - GT boxes with no prediction match."""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed")
    
    model = YOLO(model_path)
    flagged = []
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in image_extensions]
    
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
        
        gt_boxes = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    gt_boxes.append({
                        "box": [float(x) for x in parts[1:5]],
                        "class_id": int(parts[0])
                    })
        
        results = model.predict(str(img_file), conf=conf_threshold, verbose=False)
        
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                pred_boxes.append({
                    "box": box.xywhn[0].tolist(),
                    "conf": float(box.conf[0]),
                    "class_id": int(box.cls[0])
                })
        
        for gt in gt_boxes:
            found_match = False
            for pred in pred_boxes:
                iou = compute_iou_yolo(gt["box"], pred["box"])
                if iou >= iou_threshold and pred["conf"] >= conf_threshold:
                    found_match = True
                    break
            
            if not found_match:
                flagged.append({
                    "image": str(img_file),
                    "type": "hard_or_wrong_label",
                    "gt_box": gt["box"],
                    "gt_class": gt["class_id"]
                })
    
    return flagged


def sample_for_review(flagged_a: List[Dict], flagged_b: List[Dict],
                      n_samples: int = 100, seed: int = 42) -> Dict:
    """Sample flagged items for manual review."""
    random.seed(seed)
    combined = flagged_a + flagged_b
    n_samples = min(n_samples, len(combined))
    samples = random.sample(combined, n_samples) if combined else []
    
    return {
        "total_flagged_a": len(flagged_a),
        "total_flagged_b": len(flagged_b),
        "sample_size": n_samples,
        "samples": samples
    }