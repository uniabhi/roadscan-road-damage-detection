"""Tests for data splitting module."""

import pytest
from src.data.split import create_country_splits, verify_no_leakage


class TestCreateCountrySplits:
    """Tests for country-level splitting."""
    
    def test_basic_split(self):
        """Test basic split creation."""
        image_paths = [f"image_{i}.jpg" for i in range(100)]
        group_ids = list(range(100))  # Each image is its own group
        
        splits = create_country_splits(image_paths, group_ids, test_size=0.1, val_size=0.1)
        
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
    
    def test_total_count_preserved(self):
        """Total images should be preserved after split."""
        image_paths = [f"image_{i}.jpg" for i in range(100)]
        group_ids = list(range(100))
        
        splits = create_country_splits(image_paths, group_ids, test_size=0.1, val_size=0.1)
        
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 100


class TestVerifyNoLeakage:
    """Tests for leakage verification."""
    
    def test_no_leakage_passes(self):
        """Non-overlapping groups should pass."""
        splits = {
            "train": ["a.jpg", "b.jpg"],
            "val": ["c.jpg", "d.jpg"],
            "test": ["e.jpg", "f.jpg"]
        }
        image_to_group = {
            "a.jpg": 1, "b.jpg": 2,
            "c.jpg": 3, "d.jpg": 4,
            "e.jpg": 5, "f.jpg": 6
        }
        
        assert verify_no_leakage(splits, image_to_group) == True
    
    def test_leakage_detected(self):
        """Overlapping groups should be detected."""
        splits = {
            "train": ["a.jpg", "b.jpg"],
            "val": ["c.jpg", "a.jpg"],  # a.jpg in both train and val
            "test": ["e.jpg"]
        }
        image_to_group = {
            "a.jpg": 1, "b.jpg": 2,
            "c.jpg": 3, "e.jpg": 5
        }
        
        assert verify_no_leakage(splits, image_to_group) == False