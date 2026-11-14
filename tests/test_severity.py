"""Tests for the severity module."""

import pytest
from src.severity.severity import (
    compute_severity, get_severity_band, summarize_detections,
    SEVERITY_WEIGHTS
)


class TestComputeSeverity:
    """Tests for severity computation."""
    
    def test_zero_detections(self):
        """No detections should give zero severity."""
        score = compute_severity([], 640, 480)
        assert score == 0.0
    
    def test_monotonic_more_damage(self):
        """More damage should never decrease severity."""
        detections_1 = [
            {"cls": "D00", "conf": 0.8, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ]
        detections_2 = [
            {"cls": "D00", "conf": 0.8, "x1": 100, "y1": 100, "x2": 200, "y2": 200},
            {"cls": "D40", "conf": 0.9, "x1": 300, "y1": 300, "x2": 400, "y2": 400}
        ]
        
        score_1 = compute_severity(detections_1, 640, 480)
        score_2 = compute_severity(detections_2, 640, 480)
        
        assert score_2 > score_1
    
    def test_pothole_weighted_higher(self):
        """Potholes should be weighted higher than cracks."""
        img_w, img_h = 640, 480
        
        crack_det = [{"cls": "D00", "conf": 0.8, "x1": 100, "y1": 100, "x2": 200, "y2": 200}]
        pothole_det = [{"cls": "D40", "conf": 0.8, "x1": 100, "y1": 100, "x2": 200, "y2": 200}]
        
        crack_score = compute_severity(crack_det, img_w, img_h)
        pothole_score = compute_severity(pothole_det, img_w, img_h)
        
        assert pothole_score > crack_score
    
    def test_confidence_filter(self):
        """Low confidence detections should be filtered."""
        detections = [
            {"cls": "D40", "conf": 0.2, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ]
        
        score = compute_severity(detections, 640, 480, conf_min=0.3)
        assert score == 0.0


class TestGetSeverityBand:
    """Tests for severity band classification."""
    
    def test_low_band(self):
        """Low scores should be 'low' band."""
        assert get_severity_band(0.0) == "low"
        assert get_severity_band(0.02) == "low"
    
    def test_medium_band(self):
        """Medium scores should be 'medium' band."""
        assert get_severity_band(0.08) == "medium"
    
    def test_high_band(self):
        """High scores should be 'high' band."""
        assert get_severity_band(0.5) == "high"
        assert get_severity_band(1.0) == "high"


class TestSummarizeDetections:
    """Tests for detection summarization."""
    
    def test_summarize_empty(self):
        """Empty detections should give zeros."""
        summary = summarize_detections([])
        
        assert summary["total"] == 0
        assert all(v == 0 for v in summary["counts"].values())
    
    def test_summarize_mixed(self):
        """Test summarizing mixed detections."""
        detections = [
            {"cls": "D00", "conf": 0.8, "x1": 100, "y1": 100, "x2": 200, "y2": 200},
            {"cls": "D00", "conf": 0.7, "x1": 200, "y1": 200, "x2": 300, "y2": 300},
            {"cls": "D40", "conf": 0.9, "x1": 300, "y1": 300, "x2": 400, "y2": 400}
        ]
        
        summary = summarize_detections(detections)
        
        assert summary["total"] == 3
        assert summary["counts"]["D00"] == 2
        assert summary["counts"]["D40"] == 1
        assert summary["has"]["D00"] == True
        assert summary["has"]["D40"] == True