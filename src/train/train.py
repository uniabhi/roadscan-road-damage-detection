"""
Training module for RoadScan.

This module provides functions for training YOLO models on the RDD2022 dataset
with experiment tracking via MLflow.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
import json
import yaml
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def train_model(data_yaml: str,
                model_name: str = "yolo11s.pt",
                epochs: int = 60,
                imgsz: int = 640,
                batch: int = 16,
                device: int = 0,
                project: str = "runs",
                name: Optional[str] = None,
                patience: int = 15,
                seed: int = 42,
                resume: bool = False,
                **kwargs) -> Dict:
    """
    Train a YOLO model.
    
    Args:
        data_yaml: Path to data.yaml file
        model_name: Model to use (e.g., "yolo11s.pt", "yolo11n.pt")
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        device: GPU device index
        project: Project directory for runs
        name: Run name
        patience: Early stopping patience
        seed: Random seed
        resume: Whether to resume from last checkpoint
        **kwargs: Additional arguments for model.train()
    
    Returns:
        Dict with training results
    """
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    # Generate run name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"train_{timestamp}"
    
    # Initialize model
    if resume:
        # Resume from checkpoint
        checkpoint_path = Path(project) / name / "weights" / "last.pt"
        model = YOLO(str(checkpoint_path))
    else:
        # Load pretrained model
        model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        seed=seed,
        resume=resume,
        workers=kwargs.get("workers", 2),
        amp=kwargs.get("amp", True),
        **{k: v for k, v in kwargs.items() if k not in ["workers", "amp"]}
    )
    
    # Return results
    return {
        "model_path": str(Path(project) / name / "weights" / "best.pt"),
        "last_checkpoint": str(Path(project) / name / "weights" / "last.pt"),
        "results": results.results_dict if hasattr(results, "results_dict") else {}
    }


def export_to_onnx(model_path: str, 
                   output_path: Optional[str] = None,
                   imgsz: int = 640,
                   simplify: bool = True) -> str:
    """
    Export YOLO model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model (.pt)
        output_path: Output path for ONNX model
        imgsz: Image size
        simplify: Whether to simplify ONNX model
    
    Returns:
        Path to exported ONNX model
    """
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed")
    
    model = YOLO(model_path)
    
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify
    )
    
    # Move to output path if specified
    if output_path and str(onnx_path) != output_path:
        import shutil
        shutil.move(str(onnx_path), output_path)
        return output_path
    
    return str(onnx_path)


def setup_mlflow(tracking_uri: str, 
                 username: str, 
                 password: str) -> None:
    """
    Setup MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI (e.g., DagsHub URL)
        username: Username for authentication
        password: Password/token for authentication
    """
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("data_yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Model name")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--project", default="runs", help="Project directory")
    parser.add_argument("--name", help="Run name")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    
    args = parser.parse_args()
    
    results = train_model(
        data_yaml=args.data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: {results['model_path']}")
    
    if args.export_onnx:
        onnx_path = export_to_onnx(results['model_path'])
        print(f"ONNX model: {onnx_path}")
