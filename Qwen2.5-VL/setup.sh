
conda create -n qwen-vl python=3.10 -y
conda activate qwen-vl

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
 
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
 
pip install numpy==1.26.2   
pip install accelerate
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils==0.0.10
