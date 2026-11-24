"""Tests for data splitting module."""

import json
from importlib import import_module

import pytest
from src.data.split import create_country_splits, create_transfer_splits, verify_no_leakage


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


def test_groups_follow_original_image_order(monkeypatch):
    group = import_module("src.data.group")
    paths = ["c.jpg", "a.jpg", "b.jpg"]
    hashes = [group.imagehash.hex_to_hash(h) for h in ["0000", "ffff", "ffff"]]
    monkeypatch.setattr(group, "compute_hashes_batch", lambda *a, **kw: hashes)
    c, a, b = group.group_images(paths)
    assert c == b and a != b


def test_transfer_splits_keep_groups_and_indian_holdout(tmp_path):
    groups = {}
    for offset, country in enumerate(("India", "Japan", "Czech")):
        folder = tmp_path / "images" / country
        folder.mkdir(parents=True)
        for i in range(20):
            photo = folder / f"{i}.jpg"
            photo.touch()
            groups[photo.relative_to(tmp_path).as_posix()] = offset * 10 + i // 2
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps({"image_to_group": groups}))
    output = tmp_path / "splits"
    create_transfer_splits(str(tmp_path), str(manifest), str(output))

    def read(setup, partition):
        return set((output / f"{setup}_{partition}.txt").read_text().splitlines())

    target = {p for p in groups if p.startswith("images/India/")}
    test = read("s1", "test")
    assert test and test < target
    for setup in ("s1", "s2", "s3", "s4"):
        partitions = {k: read(setup, k) for k in ("train", "val", "test")}
        assert all(partitions.values())
        assert partitions["test"] == test
        assert verify_no_leakage(partitions, groups)
    assert read("s2", "train") | read("s2", "val") == groups.keys() - target
    assert read("s3", "train") | read("s3", "val") | test == target
    assert read("s4", "train") == read("s3", "train")
    assert read("s4", "val") == read("s3", "val")

    groups.pop(next(iter(groups)))
    manifest.write_text(json.dumps({"image_to_group": groups}))
    with pytest.raises(ValueError, match="Missing group"):
        create_transfer_splits(str(tmp_path), str(manifest), str(output))
