"""
Near-duplicate image grouping using perceptual hashing.

This module groups similar images from video-like sequences to prevent
data leakage between train/val/test splits.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


def phash(image_path: str, hash_size: int = 8) -> imagehash.ImageHash:
    """
    Compute perceptual hash for an image.

    Args:
        image_path: Path to image file
        hash_size: Size of the hash (larger = more precise)

    Returns:
        ImageHash object that supports subtraction for distance
    """
    try:
        with Image.open(image_path).convert("L") as img:
            img_resized = img.resize((64, 64), Image.LANCZOS)
            return imagehash.phash(img_resized, hash_size=hash_size)
    except Exception:
        return imagehash.random_hash()


def compute_hashes_batch(image_paths: List[str],
                         hash_size: int = 8,
                         show_progress: bool = True) -> List[imagehash.ImageHash]:
    """Compute perceptual hashes for multiple images."""
    hashes = []
    iterator = tqdm(image_paths, desc="Computing hashes") if show_progress else image_paths
    for path in iterator:
        hashes.append(phash(path, hash_size))
    return hashes


def group_by_sequential_comparison(hashes: List[imagehash.ImageHash],
                                   paths: List[str],
                                   window: int = 8,
                                   max_distance: int = 10) -> List[int]:
    """
    Group images using sequential comparison with a sliding window.
    Uses Union-Find (Disjoint Set Union) for efficient grouping.
    """
    n = len(hashes)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        """Find root with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        """Union by rank."""
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    for i in range(n):
        for j in range(i + 1, min(i + 1 + window, n)):
            distance = hashes[i] - hashes[j]
            if distance <= max_distance:
                union(i, j)

    return [find(i) for i in range(n)]


def group_images(image_paths: List[str],
                 window: int = 8,
                 max_distance: int = 10,
                 assume_sequential: bool = True,
                 show_progress: bool = True) -> List[int]:
    """Group near-duplicate images."""
    sorted_indices = sorted(range(len(image_paths)), key=lambda i: image_paths[i])
    sorted_paths = [image_paths[i] for i in sorted_indices]

    hashes = compute_hashes_batch(sorted_paths, show_progress=show_progress)

    comparison_window = window if assume_sequential else len(hashes)
    sorted_groups = group_by_sequential_comparison(
        hashes, sorted_paths, comparison_window, max_distance
    )

    groups = [0] * len(image_paths)
    for sorted_idx, orig_idx in enumerate(sorted_indices):
        groups[orig_idx] = sorted_groups[sorted_idx]

    return groups


def analyze_groups(groups: List[int]) -> Dict:
    """Analyze grouping results."""
    from collections import Counter
    group_sizes = Counter(groups)

    stats = {
        "total_images": len(groups),
        "num_groups": len(group_sizes),
        "avg_group_size": np.mean(list(group_sizes.values())),
        "max_group_size": max(group_sizes.values()),
        "singletons": sum(1 for s in group_sizes.values() if s == 1),
        "groups_of_2": sum(1 for s in group_sizes.values() if s == 2),
        "groups_of_3+": sum(1 for s in group_sizes.values() if s >= 3)
    }
    return stats


def save_groups(image_paths: List[str], groups: List[int], output_path: str) -> None:
    """Save grouping results to JSON file."""
    group_to_images = defaultdict(list)
    for path, group_id in zip(image_paths, groups):
        group_to_images[group_id].append(path)

    data = {
        "group_to_images": {str(k): v for k, v in group_to_images.items()},
        "image_to_group": {path: group_id for path, group_id in zip(image_paths, groups)}
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_groups(input_path: str) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """Load grouping results from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    image_to_group = {k: int(v) if isinstance(v, str) else v for k, v in data["image_to_group"].items()}
    group_to_images = {int(k): v for k, v in data["group_to_images"].items()}
    return image_to_group, group_to_images
