# Faster R-CNN for Medical Object Detection

High-quality Faster R-CNN implementation for training and evaluating custom medical-equipment detection models (e.g., ventilator, monitor, IV stand, infusion pump, bed, person).

## Step 1 — Environment Setup

Run the setup script to automatically create the environment and install all required dependencies:

```bash
bash setup.sh
```

## Step 2 — Download Pretrained Backbone or Custom Model

Option A — Download pretrained backbone (recommended for training)


Option B — Use our trained custom model (skip training)

## Step 3 — Prepare Dataset

The script voc_annotation.py is used to generate the dataset split text files and training lists:

Running the script:

```bash
python voc_annotation.py
```

This will automatically create:

Training list files which are read by ```train.py```

```txt
2007_train.txt
2007_val.txt
```

## Step 4 — Train the Faster R-CNN Model

run:

```python
python train.py 
```

Trained weights will be saved to:

```txt
./train_data/model_data/frcnn_custom.pth
```


## Step 4 — Run Inference

Use ```frcnn_run.py``` for image, or video detection.

