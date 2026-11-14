"""
FastAPI application for RoadScan.

Provides endpoints for road damage detection and LLM-powered report generation.
"""

import time
import io
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import DetectResponse, ReportResponse, HealthResponse, Detection
from .detector import Detector
from ..severity.severity import compute_severity, get_severity_band, summarize_detections


# Global state
_detector: Optional[Detector] = None
_start_time: float = time.time()


def get_detector() -> Detector:
    """Get or create detector instance."""
    global _detector
    if _detector is None:
        model_path = Path(__file__).parent.parent.parent / "models" / "best.onnx"
        if not model_path.exists():
            model_path = Path(__file__).parent.parent.parent / "models" / "best.pt"
        if not model_path.exists():
            raise RuntimeError("No model found. Place best.onnx or best.pt in models/")
        _detector = Detector(str(model_path))
    return _detector


def create_app(model_path: Optional[str] = None, conf_threshold: float = 0.25) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_path: Path to model weights
        conf_threshold: Default confidence threshold
    
    Returns:
        FastAPI application
    """
    global _detector
    
    app = FastAPI(
        title="RoadScan API",
        description="Road damage detection and maintenance report generation",
        version="0.1.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize detector if path provided
    if model_path:
        _detector = Detector(model_path, conf_threshold)
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        detector = get_detector()
        return HealthResponse(
            status="healthy",
            model_version=detector.model_version,
            uptime_seconds=time.time() - _start_time
        )
    
    @app.post("/detect", response_model=DetectResponse)
    async def detect(file: UploadFile = File(...), detector: Detector = Depends(get_detector)):
        """
        Detect road damage in an image.
        
        Args:
            file: Image file (JPG or PNG)
        
        Returns:
            Detection results with severity score
        """
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")
        
        # Read image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_array = np.array(image)
            width, height = image.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Run inference
        try:
            detections, latency = detector.predict(image_array)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        # Compute severity
        severity = compute_severity(detections, width, height)
        band = get_severity_band(severity)
        
        # Format response
        detection_objs = [
            Detection(
                cls=d["cls"],
                conf=d["conf"],
                x1=d["x1"],
                y1=d["y1"],
                x2=d["x2"],
                y2=d["y2"]
            )
            for d in detections
        ]
        
        return DetectResponse(
            image_w=width,
            image_h=height,
            detections=detection_objs,
            severity=severity,
            band=band,
            model_version=detector.model_version,
            latency_ms=latency
        )
    
    return app


# Default app instance
app = create_app()