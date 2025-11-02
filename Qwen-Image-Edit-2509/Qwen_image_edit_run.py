import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# 引入黑图诊断工具（假设已保存为fix_black_image_issue.py）
#from fix_black_image_issue import BlackImageFixer

input_image_path = "./test/test.jpg"
output_image_path = "./test/output_qwen_image_test.jpg"
local_model_path="./Qwen-Image/Qwen-Image/model_path/Qwen-Image-Edit-2509"


# 检查输入文件
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"输入文件不存在：{os.path.abspath(input_image_path)}")

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️  注意：使用CPU推理，速度会很慢，建议使用GPU")

# 初始化修复工具
#fixer = BlackImageFixer()

# ------------------- 模型加载 -------------------
# 从本地加载模型（关键修改：将远程pretrained 名称改为本地路径）
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    local_model_path,  # 使用本地模型路径
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,  # 加载自定义代码（Qwen模型可能需要）
    device_map="balanced"  # 核心：自动分配设备（GPU+CPU），避免全量占GPU
)
print("pipeline loaded from local path:", local_model_path)    
print("正在加载Qwen-Image-Edit-2509模型...")

#pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
#print("正在加载Qwen-Image-Edit模型...")
#pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=False)

# 启用CPU offloading和内存高效注意力
#pipeline.enable_model_cpu_offload()
#pipeline.enable_xformers_memory_efficient_attention()

# 启用梯度检查点（牺牲20%速度，节省40%显存）
#pipeline.enable_gradient_checkpointing()
# 启用内存高效注意力（适合Transformer模型）
#pipeline.enable_xformers_memory_efficient_attention()  # 需要安装xformers
        
print("✅ 模型加载完成")

# 预处理图像
image = Image.open(input_image_path).convert("RGB")
target_size = (1024, 1024)  # 调整为适合显存的尺寸
image = image.resize(target_size, Image.Resampling.LANCZOS)
print(f"✅ 图像预处理完成（尺寸：{image.size}）")

negative_prompt = "blurry, low quality, distorted perspective, unnatural shadow"
Prompt = "Add one more IV stand next to the bed: silver-gray metal, 3 hooks, base on floor, shadow matches the environment."

inputs = {
    "image": image,
    "prompt": Prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 3.5,
    "negative_prompt": negative_prompt.strip(),
    "num_inference_steps": 20,
    "guidance_scale": 1  # 新增该参数，与true_cfg_scale配合增强引导
}

# 应用修复工具优化参数
#fixed_inputs = fixer.apply_all_fixes(pipeline, inputs)

# 执行编辑
print("正在生成合成得图像...")
with torch.inference_mode():
    try:
        output = pipeline(**inputs)
        output_image = output.images[0]

        # 保存结果
        output_image.save(output_image_path)
        print(output_image.size, output_image.mode)
        output_image.save(output_image_path)
        print(f"✅ 结果保存至：{os.path.abspath(output_image_path)}")
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("❌ 显存不足！请减小target_size（如512x512）") from e
        else:
            raise RuntimeError(f"编辑失败：{str(e)}") from e
 
