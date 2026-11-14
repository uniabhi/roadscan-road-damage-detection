"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")


# Mock detector fixture for API tests
class MockDetector:
    """Mock detector for testing without model weights."""
    
    def predict(self, image):
        """Return mock detections."""
        return [
            {
                "class_id": 0,
                "class_name": "D00",
                "confidence": 0.85,
                "bbox": [100, 100, 200, 200]
            }
        ]
    
    @property
    def class_names(self):
        return ["D00", "D10", "D20", "D40"]