import os
import torch
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    """
    基于 Depth Anything v2 的深度估计器，支持多种模型尺寸和设备适配
    """
    def __init__(self, model_size='small', device=None):
        """
        初始化深度估计器
        
        Args:
            model_size (str): 模型尺寸 ('small', 'base', 'large')，越大精度越高但速度越慢
            device (str): 运行设备 ('cuda', 'cpu', 'mps')，自动推断默认设备
        """
        # 自动推断设备（优先GPU，其次MPS，最后CPU）
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # MPS设备兼容处理（部分操作可能不支持，强制使用CPU管道）
        if self.device == 'mps':
            print("检测到MPS设备，由于兼容性问题，深度估计管道将使用CPU")
            self.pipe_device = 'cpu'
        else:
            self.pipe_device = self.device
        
        #print(f"深度估计设备配置：主设备 {self.device}，管道设备 {self.pipe_device}")
        
        # 模型尺寸与Hugging Face模型名映射
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])
        
        # 初始化深度估计管道
        try:
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                device=self.pipe_device,
                trust_remote_code=True  # 加载自定义模型代码
            )
            print(f"成功加载 Depth Anything v2 {model_size} 模型")
        except Exception as e:
            print(f"模型加载失败：{e}，尝试降级为CPU加载...")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                device=self.pipe_device,
                trust_remote_code=True
            )
            print(f"已在CPU上加载 Depth Anything v2 {model_size} 模型")
        
        # 缓存原始深度范围（用于后续反归一化）
        self.raw_depth_min = None
        self.raw_depth_max = None

    def estimate_depth(self, image, return_raw=False):
        """
        从输入图像估计深度图
        
        Args:
            image (numpy.ndarray): 输入图像（BGR格式，OpenCV默认格式）
            return_raw (bool): 是否返回原始深度值（未归一化）
        
        Returns:
            numpy.ndarray: 深度图（归一化到0-1，或原始值）
        """
        #print(f"开始深度估计 - 输入图像尺寸: {image.shape}")  # 新增
        
        # 转换BGR→RGB（模型输入要求RGB格式）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 推理获取深度图
        with torch.no_grad():  # 禁用梯度计算，加速推理
            depth_result = self.pipe(pil_image)
        
        # 提取深度图并转换为numpy数组
        depth_map = depth_result["depth"]
        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map, dtype=np.float32)
        elif isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy().astype(np.float32)

        #print(f"深度图生成成功 - 尺寸: {depth_map.shape}, 范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")  # 新增
    
        # 保存原始深度范围（用于后续计算实际距离）
        self.raw_depth_min = depth_map.min()
        self.raw_depth_max = depth_map.max()
        
        if return_raw:
            return depth_map  # 原始深度值（无单位，相对值）
        else:
            # 归一化到0-1范围（便于可视化和后续处理）
            if self.raw_depth_max > self.raw_depth_min:
                return (depth_map - self.raw_depth_min) / (self.raw_depth_max - self.raw_depth_min)
                print(f"深度图归一化完成 - 范围: [{norm_depth.min():.4f}, {norm_depth.max():.4f}]")  # 新增
            return np.zeros_like(depth_map)

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        将深度图彩色化，便于可视化
        
        Args:
            depth_map (numpy.ndarray): 归一化的深度图（0-1）
            cmap (int): OpenCV颜色映射（如COLORMAP_INFERNO、COLORMAP_JET）
        
        Returns:
            numpy.ndarray: 彩色深度图（BGR格式，可直接用OpenCV显示）
        """
        # 转换为8位整数（0-255）
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        # 应用颜色映射并转换为BGR（OpenCV显示格式）
        colored = cv2.applyColorMap(depth_uint8, cmap)
        return colored

    def get_depth_at_point(self, depth_map, x, y, normalized=True):
        """
        获取指定像素点的深度值
        
        Args:
            depth_map (numpy.ndarray): 深度图（归一化或原始）
            x (int): 像素x坐标（列）
            y (int): 像素y坐标（行）
            normalized (bool): depth_map是否为归一化（0-1）
        
        Returns:
            float: 深度值（归一化值或原始相对值）
        """
        # 检查坐标有效性
        h, w = depth_map.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        val = depth_map[y, x]
        # 若输入是归一化图，可转换为原始值
        if normalized and self.raw_depth_min is not None:
            return val * (self.raw_depth_max - self.raw_depth_min) + self.raw_depth_min
        return val

    def get_depth_in_bbox(self, depth_map, bbox, method='median', normalized=True):
        """
        计算边界框区域内的深度值（用于目标的3D定位）
        
        Args:
            depth_map (numpy.ndarray): 深度图（归一化或原始）
            bbox (list/tuple): 边界框 [x1, y1, x2, y2]（左上角和右下角坐标）
            method (str): 聚合方法 ('median' 中位数, 'mean' 均值, 'max' 最大值)
            normalized (bool): depth_map是否为归一化（0-1）
        
        Returns:
            float: 区域内的深度值（抗噪性：median > mean > max）
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = depth_map.shape[:2]
        
        # 裁剪边界框到图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        
        # 提取区域深度
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        
        # 按指定方法聚合
        if method == 'median':
            val = np.median(region)
        elif method == 'mean':
            val = np.mean(region)
        elif method == 'max':
            val = np.max(region)
        else:
            raise ValueError(f"不支持的聚合方法：{method}")
        
        # 转换为原始深度值（若输入是归一化图）
        if normalized and self.raw_depth_min is not None:
            return val * (self.raw_depth_max - self.raw_depth_min) + self.raw_depth_min
        return val

    def denormalize_depth(self, normalized_depth):
        """
        将归一化深度图转换为原始深度值（需先调用过estimate_depth）
        
        Args:
            normalized_depth (numpy.ndarray): 归一化深度图（0-1）
        
        Returns:
            numpy.ndarray: 原始深度值
        """
        if self.raw_depth_min is None or self.raw_depth_max is None:
            raise RuntimeError("请先调用estimate_depth获取原始深度范围")
        return normalized_depth * (self.raw_depth_max - self.raw_depth_min) + self.raw_depth_min