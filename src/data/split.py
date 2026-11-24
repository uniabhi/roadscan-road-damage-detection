"""
Leakage-safe dataset splitting.

Creates train/val/test splits while keeping near-duplicate images
in the same split. Uses GroupShuffleSplit to prevent data leakage.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict

COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", 
             "China_MotorBike", "China_Drone"]


def load_group_mapping(groups_path: str) -> Dict[str, int]:
    """Load image-to-group mapping from JSON file."""
    with open(groups_path, 'r') as f:
        data = json.load(f)
    return {k: int(v) if isinstance(v, str) else v for k, v in data["image_to_group"].items()}


def create_country_splits(image_paths: List[str],
                          group_ids: List[int],
                          test_size: float = 0.10,
                          val_size: float = 0.10,
                          random_state: int = 42) -> Dict[str, List[str]]:
    """Create train/val/test splits for a single country."""
    # First split: separate test set
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss1.split(image_paths, groups=group_ids))
    
    # Second split: separate train and val
    trainval_paths = [image_paths[i] for i in trainval_idx]
    trainval_groups = [group_ids[i] for i in trainval_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
    train_idx_local, val_idx_local = next(gss2.split(trainval_paths, groups=trainval_groups))
    
    train_idx = [trainval_idx[i] for i in train_idx_local]
    val_idx = [trainval_idx[i] for i in val_idx_local]
    
    return {
        "train": [image_paths[i] for i in train_idx],
        "val": [image_paths[i] for i in val_idx],
        "test": [image_paths[i] for i in test_idx]
    }


def verify_no_leakage(splits: Dict[str, List[str]], image_to_group: Dict[str, int]) -> bool:
    """Verify that no group appears in multiple splits."""
    split_groups = {}
    for split_name, paths in splits.items():
        groups_in_split = set(image_to_group[p] for p in paths if p in image_to_group)
        split_groups[split_name] = groups_in_split
    
    all_groups = list(split_groups.values())
    for i in range(len(all_groups)):
        for j in range(i + 1, len(all_groups)):
            overlap = all_groups[i] & all_groups[j]
            if overlap:
                print(f"LEAKAGE DETECTED between splits")
                return False
    return True


def create_splits(yolo_data_dir: str, groups_path: str, output_dir: str,
                  test_size: float = 0.10, val_size: float = 0.10,
                  random_state: int = 42) -> Dict:
    """Create global splits across all countries."""
    data_path = Path(yolo_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_to_group = load_group_mapping(groups_path)
    
    country_images = defaultdict(list)
    country_groups = defaultdict(list)
    
    images_dir = data_path / "images"
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    for country_dir in images_dir.iterdir():
        if not country_dir.is_dir():
            continue
        
        country = country_dir.name
        for img_file in country_dir.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue
            
            img_path = str(img_file)
            if img_path in image_to_group:
                country_images[country].append(img_path)
                country_groups[country].append(image_to_group[img_path])
    
    all_splits = {"train": [], "val": [], "test": []}
    stats = {"countries": {}}
    
    for country in country_images:
        images = country_images[country]
        groups = country_groups[country]
        
        if len(images) == 0:
            continue
        
        splits = create_country_splits(images, groups, test_size, val_size, random_state)
        
        if not verify_no_leakage(splits, image_to_group):
            raise ValueError(f"Leakage detected in {country} splits!")
        
        for split_name in ["train", "val", "test"]:
            all_splits[split_name].extend(splits[split_name])
        
        stats["countries"][country] = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": len(images)
        }
    
    for split_name, paths in all_splits.items():
        split_file = output_path / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for p in paths:
                f.write(f"{p}\n")
        stats[f"total_{split_name}"] = len(paths)
    
    stats["total_images"] = sum(len(p) for p in all_splits.values())
    
    return stats


def create_transfer_splits(yolo_data_dir: str, groups_path: str, output_dir: str,
                           test_country: str = "India",
                           test_size: float = 0.10, val_size: float = 0.10,
                           random_state: int = 42) -> Dict:
    """
    Create transfer learning experiment splits (S1-S4).
    
    S1: Multi-country training (all countries in train)
    S2: Zero-shot (train on all except test_country)
    S3: Target-only (train only on test_country)
    S4: Pretrain + fine-tune (S2 weights fine-tuned on test_country)
    
    All share the same held-out test set from test_country.
    
    Args:
        yolo_data_dir: Path to YOLO format data
        groups_path: Path to groups.json
        output_dir: Output directory for split files
        test_country: Country to use as held-out test
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
    
    Returns:
        Dict with split statistics
    """
    data_path = Path(yolo_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group manifests may use dataset-relative paths or the paths passed to group_images.
    group_map = {Path(key.replace("\\", "/")): group
                 for key, group in load_group_mapping(groups_path).items()}
    image_to_group = {}
    target_images = []
    other_images = []
    images_dir = data_path / "images"
    for photo in sorted(images_dir.glob("*/*")):
        if not photo.is_file() or photo.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        relative = photo.relative_to(data_path)
        for key in (relative, photo, photo.resolve()):
            if key in group_map:
                break
        else:
            raise ValueError(f"Missing group for {relative}")
        path = relative.as_posix()
        image_to_group[path] = group_map[key]
        if photo.parent.name == test_country:
            target_images.append(path)
        else:
            other_images.append(path)

    if not target_images:
        raise ValueError(f"No images found for test country: {test_country}")
    target = create_country_splits(
        target_images, [image_to_group[p] for p in target_images],
        test_size, val_size, random_state,
    )
    other_groups = [image_to_group[p] for p in other_images]
    if len(set(other_groups)) < 2:
        raise ValueError("Zero-shot validation requires at least two non-target groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(other_images, groups=other_groups))
    source_train = [other_images[i] for i in train_idx]
    source_val = [other_images[i] for i in val_idx]

    test = target["test"]
    splits_config = {
        "s1": (other_images + target["train"], target["val"], test),
        "s2": (source_train, source_val, test),
        "s3": (target["train"], target["val"], test),
        # S4 starts from the selected S2 checkpoint, then fine-tunes on the target.
        "s4": (target["train"], target["val"], test),
    }

    for setup, partitions in splits_config.items():
        if not verify_no_leakage(dict(zip(("train", "val", "test"), partitions)), image_to_group):
            raise ValueError(f"A group crosses partitions in {setup}; check the group manifest")

    stats = {"setups": {}}
    
    for setup_name, (train, val, test) in splits_config.items():
        for split_name, paths in [("train", train), ("val", val), ("test", test)]:
            split_file = output_path / f"{setup_name}_{split_name}.txt"
            with open(split_file, 'w') as f:
                for p in paths:
                    f.write(f"{p}\n")
        
        stats["setups"][setup_name] = {
            "train": len(train),
            "val": len(val),
            "test": len(test)
        }
    
    stats["test_country"] = test_country
    stats["total_other_countries"] = len(other_images)
    
    return stats