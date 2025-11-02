conda create -n Qwen-Image python=3.12
conda init bash
source ~/.bashrc
conda activate Qwen-Image

export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo
unzip Qwen-Image-main.zip

git clone https://github.com/QwenLM/Qwen-Image.git
cd Qwen-Image 
pip install -r requirements.txt
