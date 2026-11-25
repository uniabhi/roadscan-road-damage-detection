"""
Dataset conversion module.

Converts RDD2022 from Pascal VOC XML format to YOLO format.
Handles train/test splits and preserves the folder structure.
"""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from .audit import KEEP_CLASSES, voc_to_yolo_lines


def convert_voc_to_yolo(xml_path: str, img_width: int, img_height: int,
                        keep_classes: Dict[str, int] = None) -> List[str]:
    """
    Convert a single VOC XML annotation to YOLO format lines.

    Args:
        xml_path: Path to VOC XML file
        img_width: Image width
        img_height: Image height
        keep_classes: Dict mapping class names to IDs (default: KEEP_CLASSES)

    Returns:
        List of YOLO format annotation strings
    """
    if keep_classes is None:
        keep_classes = KEEP_CLASSES

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    boxes = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue

        name = name_elem.text.strip()

        bb = obj.find("bndbox")
        if bb is None:
            continue

        try:
            x1 = float(bb.findtext("xmin", "0"))
            y1 = float(bb.findtext("ymin", "0"))
            x2 = float(bb.findtext("xmax", "0"))
            y2 = float(bb.findtext("ymax", "0"))
        except ValueError:
            continue

        boxes.append((x1, y1, x2, y2, name))

    return voc_to_yolo_lines(img_width, img_height, boxes)


def convert_country_dataset(source_dir: Path, output_dir: Path,
                            country: str,
                            split: str = "train",
                            resize_max: Optional[int] = None,
                            copy_images: bool = True) -> Dict:
    """
    Convert a single country's dataset to YOLO format.

    Args:
        source_dir: Source directory with VOC format data
        output_dir: Output directory for YOLO format
        country: Country name (e.g., "Japan", "India")
        split: "train" or "test"
        resize_max: Optional max dimension for resizing images
        copy_images: Whether to copy images (if False, only create labels)

    Returns:
        Dict with conversion statistics
    """
    stats = {
        "country": country,
        "split": split,
        "images_processed": 0,
        "labels_created": 0,
        "boxes_converted": 0,
        "images_skipped": 0,
        "errors": []
    }

    # Source paths
    source_country = source_dir / country / split
    images_src = source_country / "images"
    annotations_src = source_country / "annotations" / "xmls"

    if not images_src.exists():
        stats["errors"].append(f"Images directory not found: {images_src}")
        return stats

    # Output paths (YOLO structure: images/labels at same level)
    images_dst = output_dir / "images" / country
    labels_dst = output_dir / "labels" / country

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    # Process all images
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in images_src.iterdir()
                   if f.suffix.lower() in image_extensions]

    for img_file in tqdm(image_files, desc=f"Converting {country}/{split}"):
        stats["images_processed"] += 1

        # Get image dimensions
        try:
            with Image.open(img_file) as im:
                width, height = im.size
        except Exception as e:
            stats["errors"].append(f"Cannot read image {img_file}: {e}")
            stats["images_skipped"] += 1
            continue

        # Handle resize if specified
        if resize_max and (width > resize_max or height > resize_max):
            scale = resize_max / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height

        # Copy image
        if copy_images:
            dst_img = images_dst / img_file.name
            if resize_max and (width > resize_max or height > resize_max):
                # Resize and save
                with Image.open(img_file) as im:
                    im_resized = im.resize((new_width, new_height), Image.LANCZOS)
                    im_resized.save(dst_img, quality=95)
            else:
                shutil.copy2(img_file, dst_img)

        # Convert annotations
        xml_file = annotations_src / f"{img_file.stem}.xml"
        label_file = labels_dst / f"{img_file.stem}.txt"

        if xml_file.exists():
            yolo_lines = convert_voc_to_yolo(str(xml_file), new_width, new_height)

            if yolo_lines:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                stats["labels_created"] += 1
                stats["boxes_converted"] += len(yolo_lines)

    return stats


def convert_dataset(source_dir: str, output_dir: str,
                    countries: Optional[List[str]] = None,
                    splits: Optional[List[str]] = None,
                    resize_max: Optional[int] = None,
                    copy_images: bool = True) -> Dict:
    """Convert entire RDD2022 dataset to YOLO format."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if countries is None:
        countries = ["Japan", "India", "Czech", "Norway", "United_States",
                     "China_MotorBike", "China_Drone"]

    if splits is None:
        splits = ["train"]

    all_stats = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "resize_max": resize_max,
        "countries": {},
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0
    }

    for country in countries:
        for split in splits:
            stats = convert_country_dataset(
                source_path, output_path, country, split,
                resize_max=resize_max, copy_images=copy_images
            )

            key = f"{country}_{split}"
            all_stats["countries"][key] = stats
            all_stats["total_images"] += stats["images_processed"]
            all_stats["total_labels"] += stats["labels_created"]
            all_stats["total_boxes"] += stats["boxes_converted"]

    report_path = output_path / "conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    return all_stats


def create_data_yaml(output_dir: str,
                     train_list: str = "splits/train.txt",
                     val_list: str = "splits/val.txt",
                     test_list: Optional[str] = None,
                     names: Dict[int, str] = None) -> str:
    """Create data.yaml file for Ultralytics YOLO training."""
    import yaml

    if names is None:
        names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

    data = {
        "path": str(output_dir),
        "train": train_list,
        "val": val_list,
        "names": names
    }

    if test_list:
        data["test"] = test_list

    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    return str(yaml_path)
