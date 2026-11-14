# RoadScan - Road Damage Detection

[![CI](https://github.com/YOUR_USERNAME/roadscan/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/roadscan/actions/workflows/ci.yml)

**RoadScan** detects road damage (potholes and cracks) from vehicle-camera images, scores severity, and generates grounded maintenance reports using LLMs.

## Problem

Manual road surveys are slow and expensive. RoadScan automates damage detection to help road engineers prioritize repairs.

## Features

- **Detection**: YOLO11-based detection of 4 damage classes (longitudinal cracks, transverse cracks, alligator cracks, potholes)
- **Severity Scoring**: Transparent heuristic based on damage type, size, and confidence
- **LLM Reports**: Grounded maintenance reports with programmatic verification
- **Production Ready**: Dockerized FastAPI service with CI/CD pipeline

## Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/roadscan.git
cd roadscan

# Install dependencies
pip install -r requirements-serve.txt

# Run API server
uvicorn src.serve.app:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/detect` | POST | Detect road damage in image |

Example usage:
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -F "file=@road_image.jpg"
```

## Dataset

Uses **RDD2022** - a public road damage dataset with 47,420 images from 6 countries.

- **Source**: [Figshare](https://figshare.com/articles/dataset/RDD2022/21431547)
- **License**: CC BY-SA 4.0
- **Classes**: D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), D40 (pothole)

### Data Processing Pipeline

1. **Audit**: Check for annotation issues (out-of-bounds boxes, duplicates, class noise)
2. **Convert**: Pascal VOC XML to YOLO format
3. **Group**: Near-duplicate frames grouped using perceptual hashing
4. **Split**: Leakage-safe train/val/test splits at group level

## Model Training

Training is done on Kaggle GPU (free tier). See `notebooks/kaggle_train.ipynb`.

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(data="configs/data.yaml", epochs=60, imgsz=640)
```

## Results

| Setup | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| S1 Multi-country | TBD | TBD |
| S2 Zero-shot India | TBD | TBD |
| S3 India-only | TBD | TBD |
| S4 Fine-tuned | TBD | TBD |

See `reports/per_country.csv` for detailed per-country metrics.

## Architecture

```
RDD2022 Dataset
      |
      v
Data Audit & Conversion --> Near-duplicate Grouping --> Leakage-safe Splits
      |
      v
YOLO11 Training (Kaggle GPU) --> MLflow Tracking (DagsHub)
      |
      v
Evaluation: mAP per class, per country, small-object slice
      |
      v
FastAPI Service (ONNX Runtime) --> Severity Score
      |
      v
LLM Report Generation (with grounding check)
      |
      v
Docker --> GitHub Actions CI/CD --> AWS ECS Fargate
```

## Project Structure

```
roadscan/
├── configs/           # Data YAML files, training configs
├── data/              # DVC-tracked data (not in Git)
├── notebooks/         # Kaggle notebooks
├── src/
│   ├── data/          # Audit, convert, group, split
│   ├── train/         # Training, evaluation, label noise
│   ├── severity/      # Severity scoring
│   ├── serve/         # FastAPI app, schemas, detector
│   └── llm/           # LLM client, RAG, grounding
├── tests/             # Unit tests
├── scripts/           # Utility scripts
├── reports/           # Metrics, failure gallery
└── infra/             # AWS infrastructure
```

## LLM Report Grounding

The LLM report generator is constrained to only restate detection facts:

- **Input**: Detection counts, severity score, image size
- **Output**: Summary, priority, rationale, recommended action
- **Grounding Check**: Programmatic verification that:
  - Numbers in report match detection facts
  - Mentioned damage types were actually detected
  - Priority matches severity band
  - Cited sources were retrieved from RAG

## Deployment

### Docker

```bash
docker build -t roadscan .
docker run -p 8000:8000 roadscan
```

### AWS (ECS Fargate)

See `infra/aws_setup.md` for deployment instructions.

**Cost Control**:
- Use Fargate (no idle EC2 costs)
- Set AWS budget alerts
- Teardown after demo period

## Limitations

- **Not a certified engineering tool**: Severity scores are heuristics, not standards
- **No GPS mapping**: Location-based prioritization not implemented
- **No video support**: Single-frame detection only
- **Privacy**: Images may contain faces/license plates - do not store user uploads

## Reproduce

```bash
# 1. Download RDD2022 from Figshare
# 2. Run audit and conversion
python -m src.data.audit /path/to/rdd2022

# 3. Create splits
python -m src.data.split /path/to/yolo_data groups.json splits/

# 4. Train on Kaggle GPU
# See notebooks/kaggle_train.ipynb

# 5. Evaluate
python -m src.train.evaluate best.pt configs/data.yaml

# 6. Run tests
pytest tests/
```

## Statement

This project uses public data only. Not affiliated with any employer.

## License

MIT License. Dataset (RDD2022) is CC BY-SA 4.0.