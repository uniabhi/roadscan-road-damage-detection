"""Prepare local RoadScan example crops from the matching Kaggle output archive."""

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"
# Pixel coordinates select individual road views, inside the contact-sheet annotations.
SHEETS = {
    "train_batch81601.jpg": [
        ("Lane markings.jpg", (205, 292, 468, 399)),
        ("Pavement detail.jpg", (1609, 1294, 1824, 1407)),
    ],
    "train_batch81602.jpg": [("Urban intersection.jpg", (1448, 1014, 1910, 1265))],
    "val_batch0_labels.jpg": [("Rural highway.jpg", (975, 44, 1419, 232))],
}


def prepare(archive: Path) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        for sheet, crops in SHEETS.items():
            with Image.open(BytesIO(bundle.read(f"roadscan/models/roadscan/{sheet}"))) as image:
                for name, bounds in crops:
                    image.crop(bounds).convert("RGB").save(DEMO_DIR / name, quality=95)
                    print(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the Kaggle results.zip")
    prepare(parser.parse_args().archive)
