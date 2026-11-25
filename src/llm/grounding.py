"""
Groundedness checker for LLM reports.

Ensures that LLM-generated reports only contain information
that can be traced back to the detection facts and retrieved context.
"""

import re
from typing import Dict, List, Set


def check_report(report: Dict, facts: Dict, retrieved_ids: Set[str]) -> List[str]:
    """
    Check if a report is grounded in the provided facts.

    Args:
        report: LLM-generated report dict with keys:
                summary, priority, rationale, recommended_action, cited_sources
        facts: Detection facts dict with counts, total, severity, etc.
        retrieved_ids: Set of valid document IDs that were retrieved

    Returns:
        List of grounding errors (empty if all checks pass)
    """
    errors = []

    # Combine all text for number checking
    text = " ".join([
        report.get("summary", ""),
        report.get("rationale", ""),
        report.get("recommended_action", "")
    ])

    # Build allowed numbers from facts
    allowed_numbers = set()
    allowed_numbers.add(str(facts.get("total", 0)))

    counts = facts.get("counts", {})
    for count in counts.values():
        allowed_numbers.add(str(count))

    # Add severity score (allow some flexibility with rounding)
    severity = facts.get("severity_score", 0)
    allowed_numbers.add(str(round(severity, 4)))
    allowed_numbers.add(str(round(severity, 2)))
    allowed_numbers.add(str(round(severity, 1)))

    # Check for numbers in text that aren't in facts
    # A proposed maintenance deadline is not a detector measurement.
    action = re.sub(r"\bwithin\s+\d+\s+days?\b", "", report.get("recommended_action", ""), flags=re.I)
    claims = " ".join([report.get("summary", ""), report.get("rationale", ""), action])
    numbers_in_text = re.findall(r'\b\d+(?:\.\d+)?\b', claims)
    for num in numbers_in_text:
        if num not in allowed_numbers:
            errors.append(f"number_not_in_facts:{num}")

    # Check class mentions
    has = facts.get("has", {})
    class_keywords = {
        "D40": ["pothole", "potholes"],
        "D20": ["alligator", "alligator crack", "alligator cracking"],
        "D10": ["transverse", "transverse crack"],
        "D00": ["longitudinal", "longitudinal crack"]
    }

    text_lower = text.lower()
    for class_id, keywords in class_keywords.items():
        if not has.get(class_id, False):
            for kw in keywords:
                if kw in text_lower:
                    errors.append(f"class_not_detected:{class_id}")
                    break

    # Check priority matches severity band
    report_priority = report.get("priority", "").lower()
    facts_band = facts.get("severity_band", "").lower()
    if report_priority != facts_band:
        errors.append(f"priority_mismatch:report={report_priority},facts={facts_band}")

    # Check cited sources
    cited = report.get("cited_sources", [])
    for source in cited:
        if source not in retrieved_ids and source not in ["doc_1", "doc_2", "doc_3", "doc_4"]:
            errors.append(f"citation_not_retrieved:{source}")

    return errors


def is_grounded(report: Dict, facts: Dict, retrieved_ids: Set[str]) -> bool:
    """Check if a report passes all grounding checks."""
    return len(check_report(report, facts, retrieved_ids)) == 0


def validate_report_schema(report: Dict) -> List[str]:
    """Validate that report has required fields."""
    errors = []
    required_fields = ["summary", "priority", "rationale", "recommended_action", "cited_sources"]

    for field in required_fields:
        if field not in report:
            errors.append(f"missing_field:{field}")

    if "priority" in report and report["priority"] not in ["low", "medium", "high"]:
        errors.append(f"invalid_priority:{report.get('priority')}")

    if "cited_sources" in report and not isinstance(report["cited_sources"], list):
        errors.append("cited_sources_not_list")

    return errors
