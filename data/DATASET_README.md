# Dataset README

## RDD2022 - Road Damage Dataset

### Source
- **Official Source**: [Figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)
- **Paper**: [arXiv:2209.08538](https://arxiv.org/abs/2209.08538)
- **Challenge**: [CRDDC2022](https://crddc2022.sekilab.global/)

### Statistics (from paper)
- **Total Images**: 47,420
- **Countries**: 6 (Japan, India, Czech, Norway, United States, China)
- **Annotated Objects**: 55,000+

### Countries
| Country | Images |
|---------|--------|
| Japan | 13,133 |
| Norway | 10,201 |
| India | 9,665 |
| United States | 6,005 |
| Czech | 3,538 |
| China_MotorBike | 2,477 |
| China_Drone | 2,401 |

### Damage Classes (Challenge Standard)
| Class ID | Name | Description |
|----------|------|-------------|
| D00 | Longitudinal Crack | Crack running along road direction |
| D10 | Transverse Crack | Crack across road direction |
| D20 | Alligator Crack | Network of interconnected cracks |
| D40 | Pothole | Road surface hole/void |

### Annotation Format
- Pascal VOC XML format
- Location: `country/train/annotations/xmls/`

### Splits
- **Train**: 38,385 images (with annotations)
- **Test**: 9,035 images (without public annotations)

**Note**: The official test set has no public annotations. We create our own held-out test from train images.

### License
CC BY-SA 4.0 - See Figshare page for full license text.

### Citation
```bibtex
@article{arya2022RDD2022,
  title={RDD2022: A multi-national road damage dataset},
  author={Arya, Deeksha and Maeda, Hiroya and others},
  journal={arXiv preprint arXiv:2209.08538},
  year={2022}
}
```

### Known Issues
1. **Near-duplicate frames**: Consecutive frames from video are similar
2. **Label noise**: Some annotations may be incorrect
3. **Class imbalance**: Potholes (D40) are less common
4. **Non-standard classes**: Some files have other labels like "Repair"

### Download Instructions
```bash
# Download from Figshare (run on Kaggle or machine with good internet)
wget https://figshare.com/ndownloader/articles/21431547/versions/1

# Extract
unzip rdd2022.zip
```

### Data Card
- **Collection**: Smartphone and drone cameras mounted on vehicles
- **Purpose**: Road damage detection research
- **Sensitive Content**: May contain license plates, faces (blur if needed)