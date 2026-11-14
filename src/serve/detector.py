"""
Detection service for RoadScan.

Handles model loading and inference using ONNX Runtime or Ultralytics.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Class mapping
ID_TO_CLASS = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


class Detector:
    """Road damage detector using YOLO model."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize detector.
        
        Args:
            model_path: Path to model weights (.pt or .onnx)
            conf_threshold: Default confidence threshold
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.model_version = Path(model_path).stem
    
    def predict(self, image: np.ndarray,
                conf: Optional[float] = None,
                iou: float = 0.45) -> Tuple[List[Dict], float]:
        """
        Run inference on an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold for NMS
        
        Returns:
            Tuple of (list of detection dicts, latency in ms)
        """
        if conf is None:
            conf = self.conf_threshold
        
        start_time = time.time()
        
        results = self.model.predict(image, conf=conf, iou=iou, verbose=False)
        
        latency_ms = (time.time() - start_time) * 1000
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = ID_TO_CLASS.get(cls_id, f"unknown_{cls_id}")
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                
                detections.append({
                    "cls": cls_name,
                    "conf": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": cls_id
                })
        
        return detections, latency_ms
    
    def predict_from_file(self, image_path: str,
                          conf: Optional[float] = None,
                          iou: float = 0.45) -> Tuple[List[Dict], float, Tuple[int, int]]:
        """
        Run inference on an image file.
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            iou: IoU threshold for NMS
        
        Returns:
            Tuple of (detections, latency_ms, (width, height))
        """
        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))
            width, height = img.size
        
        detections, latency = self.predict(image, conf, iou)
        return detections, latency, (width, height)