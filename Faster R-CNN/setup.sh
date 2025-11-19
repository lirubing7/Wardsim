conda create -n fasterrcnn_3d python=3.8
conda init bash
source ~/.bashrc
conda activate fasterrcnn_3d
pip install transformers
pip install timm torchvision
pip install filterpy


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install opencv-python  
pip install numpy pillow matplotlib  
pip install pycocotools  
#pip install open3d  
