"""Report generation endpoint using LLM with grounding checks."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..llm.client import MockLLMClient
from ..llm.grounding import GroundingChecker
from ..llm.prompts import format_detection_prompt
from ..llm.rag import RAGRetriever
from ..severity.severity import summarize_detections

logger = logging.getLogger(__name__)
router = APIRouter()


class ReportRequest(BaseModel):
    """Request for report generation."""
    image_id: str
    detections: List[dict]
    image_width: int
    image_height: int


class ReportResponse(BaseModel):
    """Generated report with grounding verification."""
    summary: str
    priority: str
    rationale: str
    recommended_action: str
    grounded: bool
    grounding_errors: List[str]
    sources: List[str]


def create_report(
    detections: List[dict],
    image_width: int,
    image_height: int,
    llm_client=None
) -> dict:
    """
    Generate a grounded report from detections.

    Args:
        detections: List of detection dicts with class, confidence, bbox
        image_width: Image width in pixels
        image_height: Image height in pixels
        llm_client: LLM client (uses MockLLMClient if None)

    Returns:
        Report dict with summary, priority, and grounding status
    """
    # Calculate severity
    severity = summarize_detections(detections, image_width, image_height)

    # Prepare facts for grounding
    facts = {
        "detections": detections,
        "severity_score": severity["severity_score"],
        "severity_band": severity["severity_band"],
        "class_counts": severity["class_counts"],
        "has": {d["class_name"] for d in detections}
    }

    # Initialize LLM client if not provided
    if llm_client is None:
        llm_client = MockLLMClient()

    # Retrieve relevant documents
    retriever = RAGRetriever()
    retrieved_docs = retriever.retrieve(str(facts))

    # Format prompt
    prompt = format_detection_prompt(
        detections=detections,
        severity_band=severity["severity_band"],
        severity_score=severity["severity_score"],
        context="\n".join([doc["text"] for doc in retrieved_docs[:3]])
    )

    # Generate report
    try:
        raw_report = llm_client.generate(prompt)
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}, using template fallback")
        raw_report = generate_template_report(facts)

    # Parse raw report into structured format
    parsed = parse_report(raw_report)

    # Grounding check
    checker = GroundingChecker()
    grounding_result = checker.check(parsed, facts, retrieved_docs)

    return {
        "summary": parsed.get("summary", ""),
        "priority": parsed.get("priority", severity["severity_band"]),
        "rationale": parsed.get("rationale", ""),
        "recommended_action": parsed.get("recommended_action", ""),
        "grounded": grounding_result["passed"],
        "grounding_errors": grounding_result["errors"],
        "sources": [doc["id"] for doc in retrieved_docs[:3]]
    }


def generate_template_report(facts: dict) -> str:
    """Generate a simple template-based report."""
    counts = facts["class_counts"]
    band = facts["severity_band"]
    score = facts["severity_score"]

    summary_parts = []
    for class_name, count in counts.items():
        if count > 0:
            summary_parts.append(f"{count} {class_name}")

    summary = f"Detected {' and '.join(summary_parts)}."
    priority = band
    rationale = f"Severity score: {score:.2f} (band: {band})"

    if band == "high":
        action = "Immediate inspection recommended within 24 hours."
    elif band == "medium":
        action = "Schedule repair within 1-2 weeks."
    else:
        action = "Monitor during routine maintenance."

    return f"""
SUMMARY: {summary}
PRIORITY: {priority}
RATIONALE: {rationale}
RECOMMENDED_ACTION: {action}
"""


def parse_report(raw_text: str) -> dict:
    """Parse raw LLM output into structured report."""
    result = {}
    lines = raw_text.strip().split("\n")

    for line in lines:
        if line.startswith("SUMMARY:"):
            result["summary"] = line.replace("SUMMARY:", "").strip()
        elif line.startswith("PRIORITY:"):
            result["priority"] = line.replace("PRIORITY:", "").strip()
        elif line.startswith("RATIONALE:"):
            result["rationale"] = line.replace("RATIONALE:", "").strip()
        elif line.startswith("RECOMMENDED_ACTION:"):
            result["recommended_action"] = line.replace("RECOMMENDED_ACTION:", "").strip()

    return result


@router.post("/report", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest):
    """
    Generate a grounded maintenance report from detections.

    Uses LLM with RAG context and validates output against detection facts.
    """
    try:
        report = create_report(
            detections=request.detections,
            image_width=request.image_width,
            image_height=request.image_height
        )
        return ReportResponse(**report)
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
