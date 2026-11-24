"""Tests for the API endpoints."""

import pytest
import io
from importlib import import_module
from PIL import Image
from fastapi.testclient import TestClient

from src.serve.app import create_app


@pytest.fixture
def client():
    """Exercise the API without depending on local checkpoint files."""
    app = create_app()
    app.state.detector = DetectorStub()
    return TestClient(app)


class DetectorStub:
    model_version = "test"

    def predict(self, image, conf):
        assert image.shape[2] == 3
        box = dict(cls="D40", conf=0.8, x1=0, y1=0, x2=100, y2=100)
        return ([box] if conf <= 0.8 else []), 1.0


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


def test_threshold_changes_detections_and_score(client, sample_image):
    def detect(conf):
        return client.post("/detect", params={"conf": conf},
                           files={"file": ("test.jpg", sample_image.getvalue(), "image/jpeg")})

    low = detect(0.5).json()
    assert (low["image_w"], low["image_h"]) == (640, 480)
    assert len(low["detections"]) == 1
    assert low["severity"] == pytest.approx(3 * 10000 * 0.8 / (640 * 480))
    high = detect(0.9).json()
    assert high["detections"] == [] and high["severity"] == 0
    assert detect(1.1).status_code == 422


def test_examples_and_review_page(client, tmp_path, monkeypatch, sample_image):
    module = import_module("src.serve.app")
    monkeypatch.setattr(module, "EXAMPLES", tmp_path)
    (tmp_path / "road.jpg").write_bytes(sample_image.getvalue())
    (tmp_path / "notes.txt").write_text("private")
    assert client.get("/examples").json() == ["road.jpg"]
    assert client.get("/examples/road.jpg").content == sample_image.getvalue()
    assert client.get("/examples/notes.txt").status_code == 404
    assert "Review queue" in client.get("/").text
    assert client.get("/assets/review.js").status_code == 200


def test_upload_limits(client, monkeypatch, sample_image):
    module = import_module("src.serve.app")
    file = {"file": ("test.jpg", sample_image.getvalue(), "image/jpeg")}
    monkeypatch.setattr(module, "MAX_PIXELS", 10)
    assert client.post("/detect", files=file).status_code == 413
    monkeypatch.setattr(module, "MAX_BYTES", 2)
    assert client.post("/detect", files=file).status_code == 413


def test_missing_checkpoint_keeps_page_available(monkeypatch, tmp_path):
    monkeypatch.setattr(import_module("src.serve.app"), "ROOT", tmp_path)
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 503
