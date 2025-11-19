import os
import torch
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    """
    Depth estimator based on Depth Anything v2, supporting multiple model sizes and device adaptation
    """
    def __init__(self, model_size='small', device=None):
        """
        Initialize depth estimator
        
        Args:
            model_size (str): model size ('small', 'base', 'large'); larger = higher accuracy but slower
            device (str): execution device ('cuda', 'cpu', 'mps'); automatically inferred if None
        """
        # Automatically infer device (prefer GPU, then MPS, then CPU)
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # MPS compatibility handling (pipeline may not support some operations → fallback to CPU)
        if self.device == 'mps':
            print("MPS detected; due to compatibility issues, depth estimation pipeline will use CPU")
            self.pipe_device = 'cpu'
        else:
            self.pipe_device = self.device
        
        #print(f"Depth estimator device setup: main {self.device}, pipeline {self.pipe_device}")
        
        # Mapping of model sizes to Hugging Face model names
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])
        
        # Initialize depth estimation pipeline
        try:
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                device=self.pipe_device,
                trust_remote_code=True  # load custom model code
            )
            print(f"Successfully loaded Depth Anything v2 {model_size} model")
        except Exception as e:
            print(f"Model loading failed: {e}, falling back to CPU...")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                device=self.pipe_device,
                trust_remote_code=True
            )
            print(f"Depth Anything v2 {model_size} model loaded on CPU")
        
        # Cache original depth range (for later denormalization)
        self.raw_depth_min = None
        self.raw_depth_max = None

    def estimate_depth(self, image, return_raw=False):
        """
        Estimate depth map from input image
        
        Args:
            image (numpy.ndarray): input image (BGR format, OpenCV default)
            return_raw (bool): whether to return unnormalized depth
        
        Returns:
            numpy.ndarray: depth map (normalized to 0–1, or raw values)
        """
        #print(f"Starting depth estimation - input shape: {image.shape}")
        
        # Convert BGR → RGB (model expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run inference
        with torch.no_grad():  # disable gradients for faster inference
            depth_result = self.pipe(pil_image)
        
        # Extract depth map and convert to numpy
        depth_map = depth_result["depth"]
        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map, dtype=np.float32)
        elif isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy().astype(np.float32)

        #print(f"Depth map generated - shape: {depth_map.shape}, range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    
        # Save raw depth range
        self.raw_depth_min = depth_map.min()
        self.raw_depth_max = depth_map.max()
        
        if return_raw:
            return depth_map  # raw depth values
        
        else:
            # Normalize to 0–1 range for visualization & processing
            if self.raw_depth_max > self.raw_depth_min:
                return (depth_map - self.raw_depth_min) / (self.raw_depth_max - self.raw_depth_min)
                print(f"Depth map normalized - range: [{norm_depth.min():.4f}, {norm_depth.max():.4f}]")
            return np.zeros_like(depth_map)

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Apply color map to depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): normalized depth (0–1)
            cmap (int): OpenCV color map (e.g., INFERNO, JET)
        
        Returns:
            numpy.ndarray: colored depth map (BGR format)
        """
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, cmap)
        return colored

    def get_depth_at_point(self, depth_map, x, y, normalized=True):
        """
        Get depth value at a specific pixel
        
        Args:
            depth_map (numpy.ndarray): depth map
            x (int): x coordinate
            y (int): y coordinate
            normalized (bool): whether depth_map is normalized (0–1)
        
        Returns:
            float: depth value (normalized or raw)
        """
        h, w = depth_map.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        val = depth_map[y, x]
        # Convert back to raw depth if needed
        if normalized and self.raw_depth_min is not None:
            return val * (self.raw_depth_max - self.raw_depth_min) + self.raw_depth_min
        return val

    def get_depth_in_bbox(self, depth_map, bbox, method='median', normalized=True):
        """
        Compute depth inside a bounding box (for 3D positioning)
        
        Args:
            depth_map (numpy.ndarray): depth map
            bbox (list/tuple): [x1, y1, x2, y2]
            method (str): aggregation method ('median', 'mean', 'max')
            normalized (bool): whether depth_map is normalized
        
        Returns:
            float: aggregated depth value
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = depth_map.shape[:2]
        
        # Clip bbox to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        
        # Aggregate
        if method == 'median':
            val = np.median(region)
        elif method == 'mean':
            val = np.mean(region)
        elif method == 'max':
            val = np.max(region)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        # Convert back to raw depth if needed
        if normalized and self.raw_depth_min is not None:
            return val * (self.raw_depth_max - self.raw_depth_min) + self.raw_depth_min
        return val

    def denormalize_depth(self, normalized_depth):
        """
        Convert normalized depth back to raw depth values.
        Must call estimate_depth() first.
        
        Args:
            normalized_depth (numpy.ndarray): normalized depth (0–1)
        
        Returns:
            numpy.ndarray: raw depth values
        """
        if self.raw_depth_min is None or self.raw_depth_max is None:
            raise RuntimeError("Please call estimate_depth() first to obtain raw depth range")
        return normalized_depth * (self.raw_depth_max - self.raw_depth_min) + self.raw_depth_min
