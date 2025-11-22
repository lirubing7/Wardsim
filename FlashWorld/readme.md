# FlashWorld + YOLO Auto-Detection Integration

This project adds a fully automatic pipeline:

## FlashWorld Scene Generation → YOLO Detection → Output Annotated Video

After this update, FlashWorld can automatically run YOLO on every generated video—no manual steps required.

### 1. Replace the original ```cli.py``` with the updated version

The new ```cli.py``` will automatically call YOLO after FlashWorld finishes generating a scene video.

### 2. Add the YOLO detection files into the same project directory

Required files:

```txt
yolo_run.py
detection_model.py
depth_model.py
bbox3d_utils.py
```
These files must be placed in the same module path so ```cli.py``` can import them.

## Usage

Run FlashWorld as usual:

```bash
python cli.py \
  --input_dir my_inputs \
  --output_dir my_outputs \
  --video \
  --yolo
```
The system will generate:

```txt
video.mp4        # FlashWorld original output
video_yolo.mp4   # YOLO detection results (automatic)
```
