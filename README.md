# Wardsim
WardSim is a generative simulation platform designed to support the development and training of humanoid nurse robots in realistic hospital environments.
This project unifies multiple state-of-the-art models—including image editing, 3D scene generation, 2D object detection, depth estimation, and 3D detection—into a single pipeline that supports training and benchmarking of humanoid nurse robots.

![Pipeline Diagram](assets/pipeline.jpg)

## 🔧 Modular Deployment

WardSim is intentionally modularized.
To set up or run any component:

👉 Refer to the README inside each folder.

For example:

```yolo/README.md```for YOLOv11 training and inference

```fasterrcnn/README.md``` for Faster R-CNN

## 🌐 External Dependencies (3D Scene Modules)

The following modules rely on official GitHub repositories and should follow their installation steps:

### HunyuanWorld

High-fidelity 3D scene and panorama generator

### FlashWorld

Video-centric 3D and panorama generation system

These components are included as submodules/wrappers inside WardSim, but installation and environment setup follow the original authors’ instructions.

## 🔮 Future Work

- Unified launch script for full end-to-end pipeline

- Integration with ROS2 for robot-side execution

- More medical equipment categories

- Improved 3D perception via pseudo-LiDAR or NeRF methods

## 🙏 Acknowledgements

WardSim integrates multiple open-source frameworks:

- HunyuanWorld (Tencent ARC Lab)

- FlashWorld (Tencent ARC Lab)

- YOLOv11 (Ultralytics)

- Faster R-CNN (PyTorch Vision)

- Depth Anything V2

Their contributions make this research possible.