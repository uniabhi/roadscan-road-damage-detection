# RoadScan: Road Damage Detection

![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![Model](https://img.shields.io/badge/model-YOLO11s-orange)
RoadScan runs a YOLO11s road-damage detector on local CPU photos. The browser draws the model's boxes and ranks photos for human review using a severity heuristic. Training runs on a Kaggle NVIDIA T4.

## Run the local demo

In the VS Code **PowerShell** terminal on this machine:

```powershell
cd "$HOME/high-code/personal_project/roadscan"
$env:YOLO_CONFIG_DIR = "$PWD/.pytest_cache/ultralytics"
& "$HOME/anaconda3/envs/roadscan/python.exe" -m uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
```

Open [RoadScan on localhost](http://127.0.0.1:8000/). Select **Upload photos** for one or more JPG/PNG files, or choose **Review all examples**. Moving the confidence slider and releasing it runs inference again for the current batch. Select any queue row to inspect that photo's boxes, class names and scores. Tied scores preserve input order. Uploads are limited to 10 MiB and 24 megapixels per photo.

The app loads `models/best.pt` first and can also load `models/best.onnx`. The checkpoint and examples are local assets. For another machine, install `requirements-serve.txt`, supply the checkpoint, and run the same Uvicorn command with that environment's Python. Add JPG/PNG files to `data/demo/` to populate the example menu. To rebuild the four examples from this run:

```powershell
& "$HOME/anaconda3/envs/roadscan/python.exe" scripts/prepare_demo.py "$HOME/Downloads/results.zip"
```

The examples are unmarked crops from RDD2022 training and validation contact sheets in the Kaggle export. The browser requests new predictions for each image. Cropping changes framing and can change both detections and severity. These photos illustrate the interface.

## Preliminary random-split validation

The Kaggle notebook shuffled the labelled images and reserved 15% for validation. It used that split for checkpoint selection and evaluation. Nearly identical frames may appear on both sides and inflate the metrics. These numbers describe the exported checkpoint on that validation split. Grouped India-transfer test results are still pending. [Saved metrics](reports/metrics.json) | [Kaggle training notebook](notebooks/abhi-roadscan-yolo-training-on-rdd2022-fixed%20%281%29.ipynb)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **All** | 0.633 | 0.552 | **0.596** | **0.313** |
| D00 longitudinal crack | 0.624 | 0.554 | 0.587 | 0.331 |
| D10 transverse crack | 0.594 | 0.572 | 0.589 | 0.302 |
| D20 alligator crack | 0.661 | 0.635 | 0.683 | 0.382 |
| D40 pothole | 0.652 | 0.449 | 0.523 | 0.238 |

Potholes had the lowest recall (0.449). The model missed more than half of the labelled potholes at the reported validation operating point.

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

## Planned India transfer evaluation

The transfer splitter generates manifests for four experiments. It keeps the same held-out Indian groups for the final test in every setup. The grouped evaluation still requires Kaggle training, checkpoint selection, and scoring on those test images.

| Setup | Training | Validation for checkpoint selection | India test metrics |
|-------|----------|-------------------------------------|--------------------|
| S1 multi-country | Non-India images and India train groups | India validation groups | Unmeasured |
| S2 zero-shot | Non-India train groups | Separate non-India groups | Unmeasured |
| S3 India-only | India train groups | India validation groups | Unmeasured |
| S4 fine-tuned | Start from S2 weights and fine-tune on S3 training groups | India validation groups | Unmeasured |

The grouping code compares perceptual hashes in a sliding window of filename-sorted images. Split integrity depends on which near-duplicates grouping finds. Regenerate the group manifest and splits before running these experiments. The measured notebook run used its own random split. [Grouping code](src/data/group.py) | [Transfer split code](src/data/split.py)

## Local API

| Method | Path | Response |
|--------|------|----------|
| GET | `/` | Browser review interface |
| GET | `/examples`, `/examples/{name}` | Local example filenames or image |
| GET | `/health` | Model availability, version and uptime |
| POST | `/detect?conf=0.25` | Multipart field `file`; image dimensions, boxes, class/confidence, severity, band and inference time |

The batch view calls `/detect` once per photo and sorts the returned scores from highest to lowest. If one file fails, the queue shows its error and continues. To inspect a single response from PowerShell:

```powershell
curl.exe -F "file=@road_image.jpg" "http://127.0.0.1:8000/detect?conf=0.4"
```

## Severity score

Each retained box with confidence of at least 0.30 adds `class weight * box area / image area * confidence`. The weights are 3 for potholes, 2 for alligator cracks and 1 for the other crack classes. The sum is capped at 1. Selecting a detection threshold below 0.30 can display boxes that contribute zero to the score.

The default bands are low (below 0.05), medium (0.05 to below 0.15) and high (0.15 or above). These are design choices. Camera framing, overlapping boxes and missed detections affect the ranking. An engineer would need to calibrate these scores against actual road conditions before using them for maintenance decisions. [Severity implementation](src/severity/severity.py)

## Reproduce the baseline

1. Open the linked training notebook on Kaggle with the T4 accelerator and Internet enabled.
2. Run the notebook to prepare RDD2022, train YOLO11s, evaluate the selected checkpoint on the random validation split, and export the artifacts.
3. Download the output archive. Copy `roadscan/models/best.pt` from the archive into the local `models/` folder.
4. For local regression tests, use the project Python environment with pytest installed and run `python -m pytest tests`.

## Licence and credits

- Code: GPL-3.0, see [LICENSE](LICENSE).
- Dataset: RDD2022 (see its Figshare page for terms), not redistributed here.
- Built with [Ultralytics YOLO11](https://docs.ultralytics.com), which is licensed under AGPL-3.0.
- Built on public data only. Not affiliated with any employer.

## Author

Abhishek Kumar

GitHub: http://github.com/uniabhi
LinkedIn: http://linkedin.com/in/abhikr56
