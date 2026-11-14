"""Generate Kaggle training notebook."""
import json
from pathlib import Path

cells = []

# Header
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# RoadScan YOLO Training on RDD2022"]})

# Setup
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "import os, json, shutil, subprocess, glob\n",
    "from pathlib import Path\n",
    "from tqdm import tqdm\n",
    "import torch\n",
    "print(f'CUDA: {torch.cuda.is_available()}')"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "!pip install -q ultralytics\n",
    "WORK_DIR = Path('/kaggle/working/roadscan')\n",
    "RAW_DIR = WORK_DIR / 'raw'\n",
    "YOLO_DIR = WORK_DIR / 'yolo'\n",
    "MODELS_DIR = WORK_DIR / 'models'\n",
    "for d in [WORK_DIR, RAW_DIR, YOLO_DIR, MODELS_DIR]: d.mkdir(parents=True, exist_ok=True)"
]})

# Download
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 1. Download RDD2022"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "!wget -q --show-progress -O {WORK_DIR}/rdd2022.zip https://figshare.com/ndownloader/articles/21431547/versions/1"
]})

# Extract
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "print('Extracting...')\n",
    "!unzip -q {WORK_DIR}/rdd2022.zip -d {WORK_DIR}/temp\n",
    "for zf in glob.glob(str(WORK_DIR / 'temp' / '*.zip')):\n",
    "    print(f'  {Path(zf).name}')\n",
    "    subprocess.run(['unzip', '-q', zf, '-d', str(RAW_DIR)], check=True)\n",
    "!rm -rf {WORK_DIR}/temp {WORK_DIR}/rdd2022.zip\n",
    "!ls {RAW_DIR}"
]})

# Convert
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 2. Convert to YOLO"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["KEEP_CLASSES = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}"]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "import xml.etree.ElementTree as ET\n",
    "from PIL import Image\n",
    "\n",
    "def voc_to_yolo(xml_path, w, h):\n",
    "    try:\n",
    "        tree = ET.parse(xml_path)\n",
    "        root = tree.getroot()\n",
    "    except:\n",
    "        return []\n",
    "    lines = []\n",
    "    for obj in root.findall('object'):\n",
    "        name = obj.find('name')\n",
    "        if name is None or name.text not in KEEP_CLASSES:\n",
    "            continue\n",
    "        bb = obj.find('bndbox')\n",
    "        if bb is None:\n",
    "            continue\n",
    "        x1 = float(bb.findtext('xmin', '0'))\n",
    "        y1 = float(bb.findtext('ymin', '0'))\n",
    "        x2 = float(bb.findtext('xmax', '0'))\n",
    "        y2 = float(bb.findtext('ymax', '0'))\n",
    "        xc = (x1 + x2) / 2 / w\n",
    "        yc = (y1 + y2) / 2 / h\n",
    "        bw = (x2 - x1) / w\n",
    "        bh = (y2 - y1) / h\n",
    "        lines.append(f\"{KEEP_CLASSES[name.text]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\")\n",
    "    return lines"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "def convert_country(country_dir, output_dir, name):\n",
    "    images_out = output_dir / 'images' / name\n",
    "    labels_out = output_dir / 'labels' / name\n",
    "    images_out.mkdir(parents=True, exist_ok=True)\n",
    "    labels_out.mkdir(parents=True, exist_ok=True)\n",
    "    ann_dir = country_dir / 'annotations' / 'xmls'\n",
    "    if not ann_dir.exists():\n",
    "        ann_dir = country_dir / 'Annotations'\n",
    "    img_dir = country_dir / 'images'\n",
    "    if not img_dir.exists():\n",
    "        img_dir = country_dir / 'JPEGImages'\n",
    "    if not ann_dir.exists() or not img_dir.exists():\n",
    "        return 0\n",
    "    count = 0\n",
    "    for xml_file in tqdm(list(ann_dir.glob('*.xml')), desc=name):\n",
    "        img_name = xml_file.stem\n",
    "        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:\n",
    "            img_path = img_dir / (img_name + ext)\n",
    "            if img_path.exists():\n",
    "                break\n",
    "        else:\n",
    "            continue\n",
    "        try:\n",
    "            with Image.open(img_path) as img:\n",
    "                w, h = img.size\n",
    "        except:\n",
    "            continue\n",
    "        yolo_lines = voc_to_yolo(str(xml_file), w, h)\n",
    "        shutil.copy(img_path, images_out / f\"{name}_{img_name}.jpg\")\n",
    "        with open(labels_out / f\"{name}_{img_name}.txt\", 'w') as f:\n",
    "            f.write('\\n'.join(yolo_lines))\n",
    "        count += 1\n",
    "    return count"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "print('Converting to YOLO...')\n",
    "total = 0\n",
    "for subdir in sorted(RAW_DIR.iterdir()):\n",
    "    if subdir.is_dir():\n",
    "        has_ann = (subdir / 'annotations' / 'xmls').exists() or (subdir / 'Annotations').exists()\n",
    "        if has_ann:\n",
    "            n = convert_country(subdir, YOLO_DIR, subdir.name)\n",
    "            total += n\n",
    "            print(f'  {subdir.name}: {n} images')\n",
    "print(f'Total: {total} images')"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "import random\n",
    "all_imgs = sorted(str(p) for p in (YOLO_DIR / 'images').rglob('*.jpg'))\n",
    "random.Random(42).shuffle(all_imgs)\n",
    "n_val = max(1, int(len(all_imgs) * 0.1))\n",
    "val_imgs, train_imgs = all_imgs[:n_val], all_imgs[n_val:]\n",
    "(YOLO_DIR / 'train.txt').write_text('\\n'.join(train_imgs))\n",
    "(YOLO_DIR / 'val.txt').write_text('\\n'.join(val_imgs))\n",
    "print(f'train: {len(train_imgs)}, val: {len(val_imgs)}')"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "data_yaml = f\"\"\"path: {YOLO_DIR}\n",
    "train: train.txt\n",
    "val: val.txt\n",
    "nc: 4\n",
    "names: ['D00', 'D10', 'D20', 'D40']\n",
    "\"\"\"\n",
    "with open(YOLO_DIR / 'data.yaml', 'w') as f:\n",
    "    f.write(data_yaml)\n",
    "print(data_yaml)"
]})

# Train
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 3. Train YOLO11"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "from ultralytics import YOLO\n",
    "model = YOLO('yolo11s.pt')\n",
    "EPOCHS = 30  # 1 = smoke test (validated), 30 = full run\n",
    "TIME_LIMIT_H = 10.5  # Kaggle session cap is 12h; stops training gracefully\n",
    "results = model.train(data=str(YOLO_DIR / 'data.yaml'), epochs=EPOCHS, time=TIME_LIMIT_H, imgsz=640, batch=32, device=0, workers=4, cache=False, patience=8, save_period=5, seed=42, project=str(MODELS_DIR), name='roadscan', exist_ok=True)"
]})

# Evaluate
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 4. Evaluate"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "model = YOLO(str(MODELS_DIR / 'roadscan' / 'weights' / 'best.pt'))\n",
    "metrics = model.val(data=str(YOLO_DIR / 'data.yaml'), device=0)\n",
    "print(f'mAP50: {metrics.box.map50:.4f}')\n",
    "print(f'mAP50-95: {metrics.box.map:.4f}')"
]})

# Export
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 5. Export"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "onnx_path = model.export(format='onnx', simplify=True)\n",
    "best_pt = MODELS_DIR / 'roadscan' / 'weights' / 'best.pt'\n",
    "if best_pt.exists():\n",
    "    shutil.copy(best_pt, MODELS_DIR / 'best.pt')\n",
    "    print('Saved best.pt')"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "metrics_json = {'mAP50': float(metrics.box.map50), 'mAP50-95': float(metrics.box.map)}\n",
    "with open(WORK_DIR / 'metrics.json', 'w') as f:\n",
    "    json.dump(metrics_json, f, indent=2)\n",
    "print(json.dumps(metrics_json, indent=2))"
]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "!cd {WORK_DIR} && zip -r results.zip models/ metrics.json\n",
    "print('Done!')"
]})

# Create notebook
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = Path(r'c:\Users\uniab\high-code\personal_project\roadscan\notebooks\kaggle_train.ipynb')
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)
print(f'Created notebook with {len(cells)} cells')