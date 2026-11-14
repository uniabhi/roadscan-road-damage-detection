"""
CPU benchmark script for ONNX model inference.

Measures p50 and p95 latency for the detection model.
"""

import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    exit(1)


def benchmark_model(model_path: str, num_runs: int = 200, warmup: int = 10) -> dict:
    """
    Benchmark model inference latency.
    
    Args:
        model_path: Path to model (.pt or .onnx)
        num_runs: Number of inference runs
        warmup: Number of warmup runs
    
    Returns:
        Dict with latency statistics
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        model.predict(dummy_image, verbose=False)
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    
    for i in range(num_runs):
        start = time.time()
        model.predict(dummy_image, verbose=False)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs")
    
    latencies = np.array(latencies)
    
    results = {
        "model": Path(model_path).name,
        "num_runs": num_runs,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument("model_path", help="Path to model (.pt or .onnx)")
    parser.add_argument("--runs", type=int, default=200, help="Number of runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    results = benchmark_model(args.model_path, args.runs, args.warmup)
    
    print(f"\nResults for {results['model']}:")
    print(f"  Mean latency: {results['latency_ms']['mean']:.2f} ms")
    print(f"  p50 latency:  {results['latency_ms']['p50']:.2f} ms")
    print(f"  p95 latency:  {results['latency_ms']['p95']:.2f} ms")
    print(f"  p99 latency:  {results['latency_ms']['p99']:.2f} ms")
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()