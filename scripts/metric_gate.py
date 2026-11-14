"""
Metric gate script for CI/CD.

Checks if model metrics meet minimum thresholds before deployment.
"""

import argparse
import json
import sys
from pathlib import Path


def check_metrics(metrics_path: str, min_map50_95: float, min_map50: float = 0.5) -> bool:
    """
    Check if metrics meet minimum thresholds.
    
    Args:
        metrics_path: Path to metrics JSON file
        min_map50_95: Minimum mAP@50-95 threshold
        min_map50: Minimum mAP@50 threshold
    
    Returns:
        True if all checks pass
    """
    metrics_file = Path(metrics_path)
    
    if not metrics_file.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}")
        return False
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    map50 = metrics.get("map50", 0)
    map50_95 = metrics.get("map50_95", 0)
    
    print(f"Model Metrics:")
    print(f"  mAP@50:    {map50:.4f} (minimum: {min_map50})")
    print(f"  mAP@50-95: {map50_95:.4f} (minimum: {min_map50_95})")
    
    passed = True
    
    if map50 < min_map50:
        print(f"FAIL: mAP@50 {map50:.4f} < {min_map50}")
        passed = False
    else:
        print(f"PASS: mAP@50 meets threshold")
    
    if map50_95 < min_map50_95:
        print(f"FAIL: mAP@50-95 {map50_95:.4f} < {min_map50_95}")
        passed = False
    else:
        print(f"PASS: mAP@50-95 meets threshold")
    
    return passed


def main():
    parser = argparse.ArgumentParser(description="CI metric gate")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON")
    parser.add_argument("--min-map50", type=float, default=0.5, help="Minimum mAP@50")
    parser.add_argument("--min-map50-95", type=float, default=0.35, help="Minimum mAP@50-95")
    
    args = parser.parse_args()
    
    if check_metrics(args.metrics, args.min_map50_95, args.min_map50):
        print("\n✓ Metric gate PASSED")
        sys.exit(0)
    else:
        print("\n✗ Metric gate FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()