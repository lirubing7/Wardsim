from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
#from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw  # 用于绘制边界框
import os
import json
import re
import time  # 新增：导入计时模块

# 新增：记录程序总起始时间（计时起点）
total_start_time = time.time()

# 1. 指定模型保存路径（替换为你的目标路径）
model_path = "./Qwen/model_path"  # 例如："./local_models/Qwen2.5-VL-3B-Instruct"
output_img = "./test/output_qwen_detected_image5.jpg"  # 输出标注结果图片路径

# 1. 加载模型组件
# 2. 加载模型组件（关键：用 AutoModel 加载多模态模型）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # 这里改了！
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

img_path = "./test/image_5.jpg"  # 修正扩展名（.jpg 而非 .jgp）
question = "Detect monitor and IV_stand in the image and return their locations in the form of coordinates. The format of output should be like {“bbox”: [x1, y1, x2, y2], “label”: the name of this object in Englis}"

"""
请请标记图片中所有显示器（monitor）的边界框，严格遵循以下格式：
1. 每个格式：[x1,y1,x2,y2]，多个边界框用英文逗号分隔（例如 [100,200,300,400],[500,600,700,800]）；
2. 必须使用英文半角符号！逗号用“,”，括号用“[]”，绝对禁止使用中文逗号“，”或其他中文符号；
3. 不要添加任何文字、空格或换行，只输出边界框；
4. 坐标可以是整数或小数。
"""

# 3. 加载图像并检查
if not os.path.exists(img_path):
    print(f"错误：图片文件不存在 - {img_path}")
    exit(1)
try:
    image = Image.open(img_path).convert("RGB")
except Exception as e:
    print(f"图像加载失败：{e}")
    exit(1)

# 4. 构建对话消息（明确要求边界框输出）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    }
]

# 5. 处理输入
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device)

# 6. 推理（限制输出格式，避免冗余内容）
generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,  # 坐标信息很短，无需太长
    temperature=0.1,  # 降低随机性，确保输出格式稳定
    do_sample=False  # 确定性生成，优先保证格式正确
)

# 7. 解析输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0].strip()

# 8. 解析输出结果（新增打印原始输出）
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0].strip()

# 新增：打印模型原始输出（关键！看模型到底返回了什么）
print("="*50)
print("模型原始输出内容：")
print(f"{output_text}")  # 用【】包裹，清晰看到是否有多余字符
print("="*50)

# 8. 提取边界框并绘制到图片上
# 替换原解析部分
# 1. 解析模型输出，提取 bbox_2d 和 label 到组合数组
try:
    # 清除 JSON 代码块标记
    output_clean = re.sub(r'```json|```', '', output_text).strip()
    # 解析为 Python 列表（每个元素是包含 bbox_2d 和 label 的字典）
    detection_array = json.loads(output_clean)
    print("组合数组（bbox_2d, label）：")
    for item in detection_array:
        print(f"bbox_2d: {item['bbox_2d']}, label: {item['label']}")
except Exception as e:
    print(f"解析失败：{e}")
    # 新增：异常时计算并打印已耗时
    total_elapsed = round(time.time() - total_start_time, 2)
    print(f"截至解析失败，程序总耗时：{total_elapsed} 秒")
    exit(1)

# 2. 绘制边框和标签（假设已加载图片）
# 替换为你的图片路径
try:
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 遍历组合数组，绘制每个目标
    for i, item in enumerate(detection_array):
        bbox = item["bbox_2d"]
        label = item["label"]
        # 提取坐标并转换为整数
        x1, y1, x2, y2 = map(int, bbox)
        # 修正坐标范围（避免超出图片）
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        # 绘制边框（红色，线宽3）
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        # 绘制标签（红色文字，位于边框上方）
        draw.text((x1, y1 - 15), f"{label}_{i+1}", fill="red")

    # 保存绘制结果
    image.save(output_img)
    print(f"边框绘制完成，结果保存至：{output_img}")

    # 新增：程序正常完成时，计算并打印总耗时
    total_elapsed = round(time.time() - total_start_time, 2)
    print("="*50)
    print(f"生成标注图片完成！程序总耗时：{total_elapsed} 秒")
    print("="*50)

except Exception as e:
    print(f"绘制失败：{e}")
    # 新增：绘制失败时，计算并打印总耗时
    total_elapsed = round(time.time() - total_start_time, 2)
    print("="*50)
    print(f"绘制失败，程序总耗时：{total_elapsed} 秒")
    print("="*50)

"""
try:
    # 从输出中提取坐标（格式：[x1, y1, x2, y2]）
    bboxes = eval(output_text)
    # 若为单边界框（非列表套列表），转为列表套列表格式统一处理
    if not isinstance(bboxes[0], list):
        bboxes = [bboxes]
    
    # 验证每个边界框格式并绘制
    draw = ImageDraw.Draw(image)
    monitor_count = 0
    for bbox in bboxes:
        assert len(bbox) == 4, f"边界框格式错误：{bbox}（需包含4个坐标）"
        x1, y1, x2, y2 = bbox
        # 坐标范围修正（避免超出图片尺寸）
        width, height = image.size
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        # 绘制红色边界框和标注文字
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 15), f"monitor_{monitor_count + 1}", fill="red", font_size=12)
        monitor_count += 1
    
    # 保存标注结果
    image.save(output_img_path)
    print(f"成功标注 {monitor_count} 个显示器，结果保存至：{output_img_path}")

except Exception as e:
    print(f"边界框解析或绘制失败：{e}")
    # 保存原始图片（避免无输出）
    image.save(output_img_path.replace(".jpg", "_raw.jpg"))
    print(f"原始图片已保存至：{output_img_path.replace('.jpg', '_raw.jpg')}")
"""