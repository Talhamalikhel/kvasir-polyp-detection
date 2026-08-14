# Polyp Detection in GI Endoscopy — YOLO11 on Kvasir-SEG

Object detection model that locates polyps in gastrointestinal endoscopy frames, fine-tuned with transfer learning on the Kvasir-SEG dataset.

## Problem

Early polyp detection during colonoscopy is directly tied to preventing colorectal cancer — polyps that go unnoticed during a live procedure can develop into malignancies. This project builds an automated detector that draws a bounding box around any polyp present in a single endoscopy frame, as a proof-of-concept for real-time detection assistance.

## Dataset

[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) — 1,000 color endoscopy images from Simula Research Lab, each with radiologist-derived polyp annotations. The dataset ships with segmentation masks and a `kavsir_bboxes.json` file containing pixel-coordinate bounding boxes.

- Single class: `polyp`
- 55 images (5.5%) contain more than one polyp
- 333 distinct image resolutions across the 1,000 images
- Split 70/15/15 → 700 train / 150 val / 150 test, no stratification needed (single class)

Bounding boxes were converted from pixel corner format (`xmin, ymin, xmax, ymax`) to YOLO's normalized center format (`x_center, y_center, width, height`, all as fractions of image size) to match Ultralytics' expected label format.

## Approach

**Model**: YOLO11n (nano), pretrained on COCO, fine-tuned via transfer learning.

YOLO11 was chosen over YOLOv8 for its attention-based feature extraction (C2PSA blocks) and improved localization loss (DFL), reasoned to help with small/subtle-feature detection. The nano variant was chosen given the small dataset size (700 training images), to reduce overfitting risk.

**Augmentation**: relied on YOLO11's built-in augmentation pipeline (mosaic, flips, HSV jitter, Albumentations-based blur/CLAHE) rather than a custom pipeline, after reasoning that flips and rotation are safe for this domain — endoscopy footage has no fixed anatomical orientation, unlike e.g. chest X-rays where flipping can invert clinically meaningful anatomy.

**Training**: 50 epochs, image size 640, batch size 16, early stopping patience 15 (never triggered — model was still improving through epoch 50).

## Results

Evaluated on the held-out test set (150 images, 160 polyp instances, never used in training or checkpoint selection):

| Metric | Score |
|---|---|
| Precision | 0.949 |
| Recall | 0.929 |
| mAP50 | 0.971 |
| mAP50-95 | 0.848 |

Test performance exceeded validation performance (val mAP50-95: 0.754), indicating the model generalizes well without overfitting to the validation set.

## Repository structure

```
├── kvasir_polyp_detection.ipynb   # full pipeline: data prep, training, evaluation
├── app.py                          # Gradio deployment demo
├── best.pt                         # trained model weights
├── data.yaml                       # YOLO dataset config
├── requirements.txt
└── results/
    └── sample_predictions/         # example inference outputs
```

## Running it

```bash
pip install -r requirements.txt
```

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('path/to/endoscopy_frame.jpg', conf=0.5)
```

Or launch the demo app:
```bash
python app.py
```

## Limitations

- Trained on a single, relatively small dataset (1,000 images) from one source — generalization to footage from different endoscopy hardware/settings is untested.
- Single-class detection (polyp presence only) — does not classify polyp type or malignancy risk.
- Not validated for clinical use; a research/portfolio demonstration only.
