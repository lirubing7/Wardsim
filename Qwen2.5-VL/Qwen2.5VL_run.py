from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw  # Used for drawing bounding boxes
import os
import json
import re
import time  # Added: time module for timing

# Added: record total start time
total_start_time = time.time()

# 1. Specify model save path (replace with your target path)
model_path = "./Qwen/model_path"  # e.g., "./local_models/Qwen2.5-VL-3B-Instruct"
output_img = "../test/output_qwen_detected_image5.jpg"  # Output annotated image path

# 1. Load model components
# 2. Load multimodal model using AutoModel
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # Updated here!
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir=model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir=model_path,
    trust_remote_code=True
)

img_path = "../test/image_5.jpg"  
question = "Detect monitor and IV_stand in the image and return their locations in the form of coordinates. The format of output should be like {“bbox”: [x1, y1, x2, y2], “label”: the name of this object in Englis}"

"""
Please label all monitors in the image with bounding boxes. Follow these rules:
1. Format: [x1,y1,x2,y2], multiple boxes separated by commas (e.g. [100,200,300,400],[500,600,700,800]);
2. Use ONLY English half-width characters! Comma “,” and brackets “[]”; no Chinese punctuation;
3. Do NOT add any text, spaces, or newlines; only output bounding boxes;
4. Coordinates may be integers or floats.
"""

# 3. Load and verify image
if not os.path.exists(img_path):
    print(f"Error: image file does not exist - {img_path}")
    exit(1)
try:
    image = Image.open(img_path).convert("RGB")
except Exception as e:
    print(f"Failed to load image: {e}")
    exit(1)

# 4. Build conversation messages (explicit bounding box request)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    }
]

# 5. Process input
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device)

# 6. Inference (limit output format)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,  # bounding box output is short
    temperature=0.1,  # reduce randomness for stable format
    do_sample=False  # deterministic output
)

# 7. Decode output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0].strip()

# 8. Print raw model output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0].strip()

print("="*50)
print("Raw model output:")
print(f"{output_text}")
print("="*50)

# 8. Extract bounding boxes and labels
try:
    # Remove JSON code block marks
    output_clean = re.sub(r'```json|```', '', output_text).strip()
    # Parse list containing dicts of bbox_2d + label
    detection_array = json.loads(output_clean)
    print("Parsed array (bbox_2d, label):")
    for item in detection_array:
        print(f"bbox_2d: {item['bbox_2d']}, label: {item['label']}")
except Exception as e:
    print(f"Parsing failed: {e}")
    total_elapsed = round(time.time() - total_start_time, 2)
    print(f"Elapsed time until failure: {total_elapsed} seconds")
    exit(1)

# 2. Draw bounding boxes and labels
try:
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Draw each object
    for i, item in enumerate(detection_array):
        bbox = item["bbox_2d"]
        label = item["label"]
        # Convert to int
        x1, y1, x2, y2 = map(int, bbox)
        # Clamp to image boundaries
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        # Draw bounding box (red)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        # Draw label above box
        draw.text((x1, y1 - 15), f"{label}_{i+1}", fill="red")

    # Save result
    image.save(output_img)
    print(f"Bounding boxes drawn and saved to: {output_img}")

    total_elapsed = round(time.time() - total_start_time, 2)
    print("="*50)
    print(f"Annotation completed! Total runtime: {total_elapsed} seconds")
    print("="*50)

except Exception as e:
    print(f"Drawing failed: {e}")
    total_elapsed = round(time.time() - total_start_time, 2)
    print("="*50)
    print(f"Drawing failed, total runtime: {total_elapsed} seconds")
    print("="*50)


