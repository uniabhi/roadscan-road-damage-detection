"""Tests for the API endpoints."""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

from src.serve.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (640, 480), color="gray")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_schema(self, client):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_version" in data
        assert "uptime_seconds" in data


class TestDetectEndpoint:
    """Tests for /detect endpoint."""
    
    def test_detect_valid_image(self, client, sample_image):
        """Valid image should return detection results."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "image_w" in data
        assert "image_h" in data
        assert "detections" in data
        assert "severity" in data
        assert "band" in data
    
    def test_detect_invalid_content_type(self, client):
        """Non-image file should return 400."""
        response = client.post(
            "/detect",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_detect_invalid_image(self, client):
        """Corrupted image should return 400."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(b"not a real image"), "image/jpeg")}
        )
        
        assert response.status_code == 400


class TestDetectResponse:
    """Tests for detection response schema."""
    
    def test_response_has_detections_list(self, client, sample_image):
        """Response should have detections as list."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        data = response.json()
        assert isinstance(data["detections"], list)
    
    def test_severity_in_valid_range(self, client, sample_image):
        """Severity should be between 0 and 1."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        data = response.json()
        assert 0.0 <= data["severity"] <= 1.0
    
    def test_band_is_valid(self, client, sample_image):
        """Band should be low, medium, or high."""
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        data = response.json()
        assert data["band"] in ["low", "medium", "high"]