"""Tests for the grounding module."""

from src.llm.grounding import check_report, is_grounded, validate_report_schema


class TestCheckReport:
    """Tests for report grounding validation."""

    def test_grounded_report_passes(self):
        """A properly grounded report should pass."""
        report = {
            "summary": "2 damages detected with severity 0.15",
            "priority": "high",
            "rationale": "Pothole detected requiring immediate attention",
            "recommended_action": "Schedule repair within 7 days",
            "cited_sources": ["doc_1"]
        }

        facts = {
            "total": 2,
            "counts": {"D00": 1, "D10": 0, "D20": 0, "D40": 1},
            "severity_score": 0.15,
            "severity_band": "high",
            "has": {"D00": True, "D10": False, "D20": False, "D40": True}
        }

        retrieved_ids = {"doc_1", "doc_2"}

        errors = check_report(report, facts, retrieved_ids)
        assert len(errors) == 0

    def test_invented_number_fails(self):
        """Report with invented number should fail."""
        report = {
            "summary": "5 damages detected",
            "priority": "medium",
            "rationale": "Multiple damages found",
            "recommended_action": "Schedule inspection",
            "cited_sources": []
        }

        facts = {
            "total": 2,
            "counts": {"D00": 1, "D10": 0, "D20": 0, "D40": 1},
            "severity_score": 0.08,
            "severity_band": "medium",
            "has": {"D00": True, "D10": False, "D20": False, "D40": True}
        }

        errors = check_report(report, facts, set())
        assert any("number_not_in_facts" in e for e in errors)

    def test_wrong_priority_fails(self):
        """Wrong priority should fail."""
        report = {
            "summary": "1 damage detected",
            "priority": "high",
            "rationale": "Critical issue",
            "recommended_action": "Immediate repair",
            "cited_sources": []
        }

        facts = {
            "total": 1,
            "counts": {"D00": 1, "D10": 0, "D20": 0, "D40": 0},
            "severity_score": 0.02,
            "severity_band": "low",
            "has": {"D00": True, "D10": False, "D20": False, "D40": False}
        }

        errors = check_report(report, facts, set())
        assert any("priority_mismatch" in e for e in errors)

    def test_class_not_detected_fails(self):
        """Mentioning undetected class should fail."""
        report = {
            "summary": "Pothole detected requiring urgent repair",
            "priority": "medium",
            "rationale": "Pothole is dangerous",
            "recommended_action": "Fill pothole",
            "cited_sources": []
        }

        facts = {
            "total": 1,
            "counts": {"D00": 1, "D10": 0, "D20": 0, "D40": 0},
            "severity_score": 0.05,
            "severity_band": "medium",
            "has": {"D00": True, "D10": False, "D20": False, "D40": False}
        }

        errors = check_report(report, facts, set())
        assert any("class_not_detected" in e for e in errors)


class TestIsGrounded:
    """Tests for is_grounded helper."""

    def test_grounded_returns_true(self):
        """Grounded report should return True."""
        report = {"summary": "1 damage", "priority": "low", "rationale": "", "recommended_action": "", "cited_sources": []}
        facts = {"total": 1, "counts": {}, "severity_score": 0.01, "severity_band": "low", "has": {}}

        assert is_grounded(report, facts, set())


class TestValidateReportSchema:
    """Tests for report schema validation."""

    def test_valid_schema(self):
        """Valid report should have no errors."""
        report = {
            "summary": "Test",
            "priority": "medium",
            "rationale": "Test",
            "recommended_action": "Test",
            "cited_sources": ["doc_1"]
        }

        errors = validate_report_schema(report)
        assert len(errors) == 0

    def test_missing_field(self):
        """Missing field should error."""
        report = {
            "summary": "Test",
            "priority": "medium"
        }

        errors = validate_report_schema(report)
        assert any("missing_field" in e for e in errors)

    def test_invalid_priority(self):
        """Invalid priority should error."""
        report = {
            "summary": "Test",
            "priority": "critical",
            "rationale": "Test",
            "recommended_action": "Test",
            "cited_sources": []
        }

        errors = validate_report_schema(report)
        assert any("invalid_priority" in e for e in errors)
