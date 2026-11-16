# Polyp Detection in GI Endoscopy — YOLO11

An object detection project using **YOLO11n** to detect and localize polyps in gastrointestinal endoscopy images. Fine-tuned on the Kvasir-SEG dataset using transfer learning with PyTorch and Ultralytics.

## Dataset

- **Source:** [Kvasir-SEG – Simula Research Laboratory](https://datasets.simula.no/kvasir-seg/)
- **Images:** 1,000 endoscopy images
- **Classes:** 1 (`polyp`)
- **Split:** 700 training / 150 validation / 150 testing
- **Annotations:** Bounding boxes converted to YOLO format

## Model & Training

- **Architecture:** YOLO11n (Nano)
- **Pretrained weights:** COCO
- **Image size:** 640 × 640
- **Batch size:** 16
- **Epochs:** 50
- **Augmentation:** YOLO11 built-in augmentation pipeline

## Results

Test-set evaluation on 150 images:

| Metric | Score |
|---|---:|
| Precision | 0.949 |
| Recall | 0.929 |
| mAP@50 | 0.971 |
| mAP@50-95 | 0.848 |

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Gradio demo:

```bash
python app.py
```

Upload an endoscopy image to visualize predicted polyp bounding boxes.

## Repository Structure

```text
kvasir-polyp-detection/
├── app.py
├── data.yaml
├── .gitignore
├── kvasir.ipynb
├── kvasir_best.pt
├── requirements.txt
└── README.md
```

## Limitations

- Trained on a relatively small dataset from a single source.
- Detects polyps but does not classify polyp types or malignancy.
- Not clinically validated; intended for research and educational demonstration only.
