# YOLOv11 Custom Object Detection Pipeline

This folder provides a complete workflow for training and running a **custom YOLOv11** object detection model, including dataset splitting, training, and inference.

## Step 1 — Split Dataset (train/val/test)

Make sure your rawdata/images and rawdata/labels follow YOLO format.

Run the script:

```bash
python split_data.py
```

This will randomly split your dataset into train / val / test according to your predefined ratios, and store them under:

```txt
dataset/
├── train/
├── val/
└── test/
```

## Step 2 — Train the YOLOv11 Custom Model

Run:

```bash
python train_yolo.py
```

After training, the best-performing weights will be saved automatically to:

```txt
weights/best.pt
```

You may also directly use our pre-trained best.pt.

## Step 3 — Run Inference

Set the custom model path inside detection_model/, for example:

```txt
model_path = "weights/best.pt"
```

Then run:

```bash
python yolo_run.py
```

It will load your YOLOv11 model and perform object detection.