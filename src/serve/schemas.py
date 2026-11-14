"""
Pydantic schemas for the RoadScan API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single detection result."""
    cls: Literal["D00", "D10", "D20", "D40"]
    conf: float = Field(ge=0, le=1, description="Confidence score")
    x1: float = Field(description="Bounding box x1 (pixels)")
    y1: float = Field(description="Bounding box y1 (pixels)")
    x2: float = Field(description="Bounding box x2 (pixels)")
    y2: float = Field(description="Bounding box y2 (pixels)")


class DetectResponse(BaseModel):
    """Response for /detect endpoint."""
    image_w: int = Field(description="Image width")
    image_h: int = Field(description="Image height")
    detections: List[Detection] = Field(default_factory=list)
    severity: float = Field(ge=0, le=1, description="Severity score")
    band: Literal["low", "medium", "high"] = Field(description="Severity band")
    model_version: str = Field(description="Model version")
    latency_ms: float = Field(description="Inference latency in milliseconds")


class ReportResponse(BaseModel):
    """Response for /report endpoint."""
    detect_result: DetectResponse
    summary: str = Field(description="LLM-generated summary")
    priority: Literal["low", "medium", "high"]
    rationale: str = Field(description="Reasoning for priority")
    recommended_action: str = Field(description="Recommended action")
    cited_sources: List[str] = Field(default_factory=list)
    grounded: bool = Field(description="Whether report passed grounding check")
    grounding_errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: Literal["healthy", "unhealthy"]
    model_version: str
    uptime_seconds: float