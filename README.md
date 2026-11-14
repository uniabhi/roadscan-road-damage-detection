# RoadScan: Road Damage Detection

![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![Model](https://img.shields.io/badge/model-YOLO11s-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

RoadScan finds potholes and cracks in ordinary vehicle-camera images, scores how severe the damage is, and drafts a maintenance report for a road engineer. The report writer is only allowed to restate numbers the detector produced, and a programmatic check enforces that.

## Problem

Manual road surveys are slow and expensive, and damaged roads cause accidents. Cheap smartphone or dashcam images can be screened automatically, and an engineer then reviews only the flagged ones. RoadScan is built for that user: a road engineer who needs a ranked list of what to fix first, not just boxes on images.

## Project status

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset audit, VOC to YOLO conversion (4 classes) | Done | `src/data/`, Kaggle notebook in [`docs/`](docs/) |
| Near-duplicate grouping, grouped splits and held-out test set | Done | `src/data/group.py`, `split.py` |
| YOLO11s training on Kaggle GPU | Done | 58 epochs, results below |
| Transfer experiments S1 to S4 (India vs other countries) | Done | See transfer table below |
| Error analysis and failure gallery | Done | `reports/` |
| ONNX export | Done | `best.onnx`, 36 MB (not stored in git) |
| Severity score | Done | Transparent heuristic, see below |
| FastAPI service, Docker, CI | Done | `src/serve/`, `Dockerfile`, `.github/` |
| LLM maintenance report with grounding check | Done | `src/llm/` |
| AWS deployment (ECS Fargate) | Done | Kept live only for the demo period; see `infra/` |

## Baseline results

Model: `yolo11s.pt` (COCO-pretrained), 9.4 M parameters, trained on a Kaggle Tesla T4. Evaluated on the validation split with the best checkpoint.

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **All** | 0.633 | 0.552 | **0.596** | **0.314** |
| D00 longitudinal crack | 0.624 | 0.554 | 0.587 | 0.331 |
| D10 transverse crack | 0.594 | 0.572 | 0.589 | 0.302 |
| D20 alligator crack | 0.661 | 0.635 | 0.683 | 0.382 |
| D40 pothole | 0.652 | 0.449 | 0.523 | 0.238 |

Inference took about 5.2 ms per image at 640 px on the T4 (Ultralytics validation log, PyTorch weights). CPU latency of the ONNX model is measured separately with `scripts/`.

> **How to read these numbers.** They come from the Kaggle training notebook, which shuffles all labelled images at random and uses 15% for validation. Frames from a moving vehicle are near-duplicates, so a random split can put almost identical images in both train and validation, and these scores are probably optimistic. The validation set was also used for early stopping and for choosing the checkpoint, so this is a baseline, not an untouched test score. The grouped-split pipeline in `src/data/` is the leakage-safe version.

Pothole recall is the weakest result (0.449), so many potholes are missed at the default thresholds.

Full training log and code: [`docs/roadscan-yolo11-training-kaggle.md`](docs/roadscan-yolo11-training-kaggle.md).

## Dataset

[RDD2022](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) is the public Road Damage Dataset released for the CRDDC2022 challenge: 47,420 images from six countries, with Pascal VOC bounding boxes. Paper: [arXiv 2209.08538](https://arxiv.org/abs/2209.08538).

- **Classes kept:** D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. Other label types are dropped.
- **Test images are unlabelled** in the public release, so only the 38,385 labelled training images are used, split into train and validation.
- **Licence:** listed as CC BY-SA 4.0 by a third-party dataset site. Check the Figshare page for the authoritative terms. The dataset is **not redistributed** in this repo.

| Subset | Labelled images used |
|--------|---------------------|
| Japan | 10,506 |
| Norway | 8,161 |
| India | 7,706 |
| United States | 4,805 |
| Czech | 2,829 |
| China (drone) | 2,401 |
| China (motorbike) | 1,977 |
| **Total** | **38,385** |

| Split | Images | Of which no damage | D00 | D10 | D20 | D40 |
|-------|--------|--------------------|-----|-----|-----|-----|
| Train | 32,628 | 12,454 | 21,933 | 9,998 | 9,055 | 5,573 |
| Val | 5,757 | 2,164 | 4,083 | 1,832 | 1,561 | 971 |

Class counts show an imbalance: longitudinal cracks are the most common class and potholes the rarest.

## Training setup

| Setting | Value |
|---------|-------|
| Model | `yolo11s.pt`, Ultralytics 8.4.204 |
| Image size / batch | 640 / 16 |
| Time cap / patience | 9 hours / 10 epochs |
| Seed | 42 |
| Hardware | Kaggle Tesla T4 (one GPU) |
| Result | 58 epochs in 9.02 hours |

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16,
            patience=10, seed=42, time=9.0)   # time (hours) overrides epochs
```

## Transfer evaluation: India

All four setups are scored on the same India test set.

| Setup | Train on | mAP@50 | mAP@50-95 |
|-------|----------|--------|-----------|
| S1 multi-country | All countries | TBD | TBD |
| S2 zero-shot India | All countries except India | TBD | TBD |
| S3 India-only | India only | TBD | TBD |
| S4 fine-tuned | S2 weights, then India data | TBD | TBD |

## Architecture

```text
RDD2022 (Figshare)
   |
   v
Label audit -> VOC to YOLO (4 classes) -> near-duplicate grouping -> grouped splits
   |
   v
YOLO11 training on Kaggle GPU -> best.pt -> best.onnx
   |
   v
Evaluation: mAP overall, per class, per country, small-object slice
   |
   v
FastAPI + ONNX Runtime (CPU) -> detections + severity band
   |
   v
LLM report (facts only) + grounding check
   |
   v
Docker -> GitHub Actions -> AWS ECS Fargate
```

## Quick start (API)

The model weights are not stored in git. Train your own with the notebook, or place a trained `best.onnx` at `models/best.onnx`.

```bash
git clone https://github.com/uniabhi/roadscan-road-damage-detection.git
cd roadscan-road-damage-detection

pip install -r requirements-serve.txt
uvicorn src.serve.app:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model version |
| `/detect` | POST | Detections and severity band for one image |
| `/report` | POST | Detections plus a draft LLM report with grounding status |

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -F "file=@road_image.jpg"
```

Docker:

```bash
docker build -t roadscan .
docker run -p 8000:8000 roadscan
```

## Severity score

A transparent heuristic, not an engineering standard. Each detection above a confidence threshold adds `class weight x box area share x confidence`, with weights chosen by design (pothole 3, alligator crack 2, other cracks 1), capped at 1. The low / medium / high bands are set from validation-set percentiles. Calibrating them against real engineer priorities is future work.

## LLM report and grounding check

The report generator receives only a facts object (counts per class, severity, band, image size) plus a few retrieved repair-guidance passages. It returns a summary, priority, rationale, recommended action and cited sources. A programmatic check then verifies that:

- every number in the text appears in the detection facts,
- every damage type mentioned was actually detected,
- the stated priority matches the severity band,
- every cited source was actually retrieved.

If the check fails, the generator retries once and then falls back to a plain template. Reports are drafts for an engineer to approve.

## Project structure

```text
roadscan-road-damage-detection/
├── configs/       # data YAML files, training configs
├── data/          # DVC-tracked data (not in git)
├── docs/          # project guide and Kaggle training notes
├── infra/         # AWS setup
├── models/        # weights (not in git)
├── notebooks/     # Kaggle notebooks
├── reports/       # metrics, failure gallery
├── scripts/       # utility scripts
├── src/
│   ├── data/      # audit, convert, group, split
│   ├── train/     # training, evaluation, label noise
│   ├── severity/  # severity scoring
│   ├── serve/     # FastAPI app, schemas, detector
│   └── llm/       # LLM client, RAG, grounding
└── tests/         # unit tests
```

## Reproduce the baseline

1. Open the notebook in `notebooks/` on Kaggle with the accelerator set to GPU (T4) and Internet on.
2. Run all cells. The notebook downloads RDD2022 from Figshare, converts it to YOLO format, trains, evaluates and exports `best.pt` and `best.onnx`.
3. Download `results.zip` from the notebook's Output tab.
4. Run the tests locally with `pytest tests/`.

## Limitations

- **Optimistic baseline metrics:** the Kaggle notebook uses a random train/val split, see the note above.
- **Not a certified engineering tool:** severity scores are heuristics.
- **No GPS mapping or video tracking:** single-frame detection only.
- **Class imbalance and label noise:** potholes are the rarest class and are missed most often.
- **Privacy:** road images can contain faces and licence plates. Uploaded images should not be stored.

## Licence and credits

- Code: GPL-3.0, see [LICENSE](LICENSE).
- Dataset: RDD2022 (see its Figshare page for terms), not redistributed here.
- Built with [Ultralytics YOLO11](https://docs.ultralytics.com), which is licensed under AGPL-3.0.
- Built on public data only. Not affiliated with any employer.

## Author

Abhishek Kumar

GitHub: http://github.com/uniabhi
LinkedIn: http://linkedin.com/in/abhikr56
