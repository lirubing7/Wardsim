import os

def check_image_label_matching(img_dir, label_dir):
    """
    检查图片和标签文件是否存在且文件名匹配（忽略扩展名）
    :param img_dir: 图片文件夹路径
    :param label_dir: 标签文件夹路径
    :return: 检查结果（True/False）和不匹配的文件信息
    """
    # 检查文件夹是否存在
    if not os.path.isdir(img_dir):
        return False, f"图片文件夹不存在: {img_dir}"
    if not os.path.isdir(label_dir):
        return False, f"标签文件夹不存在: {label_dir}"
    
    # 获取所有图片文件（过滤常见图片格式）
    img_extensions = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    if not img_files:
        return False, f"图片文件夹中未找到任何图片文件: {img_dir}"
    
    # 获取所有标签文件（假设为YOLO格式的txt文件）
    label_extension = '.txt'
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(label_extension)]
    if not label_files:
        return False, f"标签文件夹中未找到任何标签文件: {label_dir}"
    
    # 提取文件名（不含扩展名）用于匹配
    img_basenames = {os.path.splitext(f)[0] for f in img_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}
    
    # 检查不匹配的文件
    missing_labels = img_basenames - label_basenames  # 有图片但无标签
    missing_images = label_basenames - img_basenames  # 有标签但无图片
    
    # 生成检查报告
    report = []
    if missing_labels:
        report.append(f"以下图片缺少对应标签: {', '.join(missing_labels)}")
    if missing_images:
        report.append(f"以下标签缺少对应图片: {', '.join(missing_images)}")
    
    if report:
        return False, "\n".join(report)
    else:
        return True, f"所有文件匹配成功！图片数量: {len(img_files)}, 标签数量: {len(label_files)}"

# 你的文件夹路径
yolo_file = r'/home/autodl-tmp/rawdata/labels'   # 标签文件夹
img_file = r'/home/autodl-tmp/rawdata/images'    # 图片文件夹

# 执行检查
match, message = check_image_label_matching(img_file, yolo_file)
print("检查结果：")
print(message)
