# VisionSpec QC — AI-Powered PCB Defect Detection

VisionSpec QC is an end-to-end deep learning project built to automate Printed Circuit Board (PCB) quality inspection using computer vision.

The system classifies PCB images into:

* **PASS** — No visible defect
* **DEFECT** — Fault detected

## Key Highlights

* Built using **PyTorch** with transfer learning
* Trained **MobileNetV2 / ResNet50** models
* Achieved **96.68% test accuracy** and **0.995 AUC-ROC**
* Implemented **Grad-CAM** for model interpretability
* Developed **Flask inference API** for real-time predictions
* Designed modern **Glassmorphism Dashboard UI**
* Achieved **~35 ms average latency** and **~28 FPS**

## Features

* Upload PCB images for instant inspection
* PASS / DEFECT prediction with confidence score
* Real-time latency display
* Inspection history tracking
* Responsive web dashboard

## Tech Stack

* Python
* PyTorch
* OpenCV
* Flask
* HTML / CSS / JavaScript
* Git / GitHub

## Run Project

```bash
pip install -r requirements.txt
python src/inference.py --weights outputs/logs/week2/mobilenetv2_head_best.pth --arch mobilenetv2
```

Open:

```text
http://localhost:5000
```

## Applications

PCB manufacturing, smart factory automation, defect screening, electronics quality assurance.


