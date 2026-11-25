"""
Evaluation module for RoadScan.

Provides functions for evaluating trained models on various splits
and generating per-country performance metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def evaluate_model(model_path: str, data_yaml: str, split: str = "val",
                   imgsz: int = 640, conf: float = 0.001,
                   iou: float = 0.6, device: int = 0) -> Dict:
    """Evaluate a YOLO model on a dataset split."""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed")

    model = YOLO(model_path)
    results = model.val(data=data_yaml, split=split, imgsz=imgsz,
                        conf=conf, iou=iou, device=device, verbose=True)

    metrics = {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "per_class_map50_95": [float(x) for x in results.box.maps],
    }

    per_class = {}
    for i, name in CLASS_NAMES.items():
        per_class[name] = {"map50_95": metrics["per_class_map50_95"][i] if i < len(metrics["per_class_map50_95"]) else 0.0}
    metrics["per_class"] = per_class

    return metrics


def evaluate_per_country(model_path: str, data_dir: str,
                         countries: List[str], output_path: Optional[str] = None,
                         imgsz: int = 640, device: int = 0) -> pd.DataFrame:
    """Evaluate model performance per country."""
    results = []

    for country in countries:
        yaml_path = Path(data_dir) / f"test_{country}.yaml"
        if not yaml_path.exists():
            continue

        try:
            metrics = evaluate_model(model_path, str(yaml_path), split="val",
                                     imgsz=imgsz, device=device)
            results.append({
                "country": country,
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
            })
        except Exception as e:
            print(f"Error evaluating {country}: {e}")

    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
    return df


def generate_metrics_report(metrics: Dict, output_path: str) -> None:
    """Generate a JSON file with metrics for CI metric gate."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
