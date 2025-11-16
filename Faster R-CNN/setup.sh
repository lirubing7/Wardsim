# 1. 创建虚拟环境（可选但推荐）

conda create -n fasterrcnn_3d python=3.8
conda init bash
source ~/.bashrc
conda activate fasterrcnn_3d
pip install transformers
pip install timm torchvision
pip install filterpy

# 2. 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # 适配CUDA 11.8，根据显卡调整
pip install opencv-python  # 视频读取与帧处理
pip install numpy pillow matplotlib  # 数据处理与可视化
pip install pycocotools  # 目标检测评估（可选，若需验证2D检测效果）
#pip install open3d  # 可选，用于3D点云与边框可视化
