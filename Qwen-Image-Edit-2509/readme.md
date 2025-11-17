# Qwen-Image-Edit-2509 Local Execution Guide

This document describes how to set up and run **Qwen-Image-Edit-2509** on a local machine.

## 🖥️ Hardware & Software Environment

The model is executed on the following environment:

- **GPU**: NVIDIA RTX 5090  
- **PyTorch**: 2.7.0  
- **Python**: 3.12  
- **CUDA Toolkit**: 12.8  

Ensure that your environment meets or exceeds these specifications for stable performance.

---

## Step 1 — Download Qwen-Image-Edit-2509 Locally

First, navigate to your desired model directory:

```bash
cd model_path
```

Use ModelScope to download the model:

```bash
modelscope download --model Qwen/Qwen-Image-Edit-2509 --local_dir ./Qwen-Image-Edit-2509
```

## Step 2 — Set Up Environment

Run the setup script to install all required dependencies:

```bash
bash Qwen-Image-setup.sh
```

## Step 3 — Configure Prompt and Run the Editor

Modify the prompt inside Qwen_image_edit_run.py, for example:

```python
prompt = "Replace the hospital bed with a modern ICU bed, keep lighting consistent."
```

After updating your edit instructions, run:

```bash
python Qwen_image_edit_run.py
```

The script will load Qwen-Image-Edit-2509 and generate the edited image based on your prompt.