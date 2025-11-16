import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Import black image diagnostic tool (assuming saved as fix_black_image_issue.py)
#from fix_black_image_issue import BlackImageFixer

input_image_path = "../test/test.jpg"
output_image_path = "../test/output_qwen_image_test.jpg"
local_model_path = "./Qwen-Image/Qwen-Image/model_path/Qwen-Image-Edit-2509"


# Check input file
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Input file does not exist: {os.path.abspath(input_image_path)}")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Note: Inference using CPU will be very slow. It is recommended to use a GPU.")

# Initialize repair tool
#fixer = BlackImageFixer()

# ------------------- Model Loading -------------------
# Load model from local path (key modification: change remote pretrained name to local path)
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    local_model_path,  # Use local model path
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,  # Load custom code (may be required for Qwen models)
    device_map="balanced"  # Core: automatically allocate devices (GPU+CPU) to avoid full GPU occupation
)
print("Pipeline loaded from local path:", local_model_path)    
print("Loading Qwen-Image-Edit-2509 model...")

#pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
#print("Loading Qwen-Image-Edit model...")
#pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=False)

# Enable CPU offloading and memory-efficient attention
#pipeline.enable_model_cpu_offload()
#pipeline.enable_xformers_memory_efficient_attention()

# Enable gradient checkpointing (sacrifices 20% speed, saves 40% memory)
#pipeline.enable_gradient_checkpointing()
# Enable memory-efficient attention (suitable for Transformer models)
#pipeline.enable_xformers_memory_efficient_attention()  # Requires xformers installation
        
print("Model loaded successfully")

# Preprocess image
image = Image.open(input_image_path).convert("RGB")
target_size = (1024, 1024)  # Adjust to a size suitable for memory
image = image.resize(target_size, Image.Resampling.LANCZOS)
print(f"Image preprocessing completed (size: {image.size})")

negative_prompt = "blurry, low quality, distorted perspective, unnatural shadow"
prompt = "Add one more IV stand next to the bed: silver-gray metal, 3 hooks, base on floor, shadow matches the environment."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 3.5,
    "negative_prompt": negative_prompt.strip(),
    "num_inference_steps": 20,
    "guidance_scale": 1  # Add this parameter to enhance guidance with true_cfg_scale
}

# Apply repair tool to optimize parameters
#fixed_inputs = fixer.apply_all_fixes(pipeline, inputs)

# Perform editing
print("Generating synthetic image...")
with torch.inference_mode():
    try:
        output = pipeline(** inputs)
        output_image = output.images[0]

        # Save result
        output_image.save(output_image_path)
        print(output_image.size, output_image.mode)
        output_image.save(output_image_path)
        print(f"Result saved to: {os.path.abspath(output_image_path)}")
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("Out of memory! Please reduce target_size (e.g., 512x512)") from e
        else:
            raise RuntimeError(f"Editing failed: {str(e)}") from e
