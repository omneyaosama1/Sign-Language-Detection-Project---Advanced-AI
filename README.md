# ASL Sign Language Recognition System

A real-time American Sign Language recognition system using YOLOv8, Roboflow, and OpenCV.

## Structure
- `training/`: Notebooks to train YOLOv8 models using Roboflow.
- `detection/`: Python script to run real-time letter detection.
- `writing/`: Real-time detection + letter-to-word builder.
- `models/`: Trained YOLOv8 models.
- `utils/`: Optional helpers for smoothing, speech, etc.

## Setup
```bash
pip install -r requirements.txt
