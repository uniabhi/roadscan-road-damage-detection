"""
Prompt templates for LLM report generation.
"""

SYSTEM_PROMPT = """You are a road maintenance assistant. Generate maintenance reports based ONLY on the detection facts provided. Never invent numbers or damage types not in the facts.

Output valid JSON with these keys:
- summary: 1-2 sentence description of the detected damage
- priority: "low", "medium", or "high" (must match the provided severity band)
- rationale: explanation for the priority level
- recommended_action: specific maintenance recommendation
- cited_sources: list of source IDs from the provided context (empty list if none)

Do not convert or compute any numbers. Only restate the exact values provided."""

REPORT_PROMPT_TEMPLATE = """Generate a road maintenance report based on these detection facts:

DETECTION FACTS (use EXACTLY these values, do not modify):
{facts_json}

REPAIR GUIDANCE CONTEXT:
{context}

Output JSON with keys: summary, priority, rationale, recommended_action, cited_sources."""


def format_facts(detections_summary: dict, severity: float, band: str, 
                 model_version: str) -> str:
    """Format detection facts as JSON string for the prompt."""
    import json
    facts = {
        "counts": detections_summary.get("counts", {}),
        "total": detections_summary.get("total", 0),
        "severity_score": round(severity, 4),
        "severity_band": band,
        "model_version": model_version,
        "has": detections_summary.get("has", {})
    }
    return json.dumps(facts, indent=2)


def build_report_prompt(detections_summary: dict, severity: float, band: str,
                        model_version: str, context: str = "") -> str:
    """Build the full prompt for report generation."""
    facts_json = format_facts(detections_summary, severity, band, model_version)
    return REPORT_PROMPT_TEMPLATE.format(facts_json=facts_json, context=context)