"""Push notebook to Kaggle."""
import os
import json
from pathlib import Path

# Create kernel metadata - use EXISTING kernel ID
kernel_meta = {
    "id": "uniabhi/roadscan-yolo-training-on-rdd2022",
    "title": "RoadScan - YOLO Training on RDD2022",
    "code_file": "kaggle_train.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

# Write metadata to notebooks folder
notebooks_dir = Path("notebooks")
meta_path = notebooks_dir / "kernel-metadata.json"
with open(meta_path, 'w') as f:
    json.dump(kernel_meta, f, indent=2)

print(f"Created kernel-metadata.json")

# Push using kaggle CLI via subprocess
import subprocess
result = subprocess.run(
    ["python", "-m", "kaggle", "kernels", "push", "-p", str(notebooks_dir)],
    capture_output=True,
    text=True
)

print(f"stdout: {result.stdout}")
if result.stderr:
    print(f"stderr: {result.stderr}")
print(f"Return code: {result.returncode}")

if result.returncode == 0:
    print("\nSuccess! Check your notebook at: https://www.kaggle.com/code/uniabhi/roadscan-yolo-training-on-rdd2022")
