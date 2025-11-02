import os
import shutil
import random
from pathlib import Path

# 配置参数
SOURCE_IMAGE_DIR = "./rawdata/images"  # 原始图片目录
SOURCE_LABEL_DIR = "./rawdata/labels"  # 原始标签目录
DEST_DIR = "./dataset"                 # 目标目录（与源目录相同，将在其中创建train/val/test）
SPLIT_RATIOS = (0.7, 0.2, 0.1)         # 划分比例：train:val:test
RANDOM_SEED = 42                       # 随机种子，确保结果可复现

# 支持的图片文件扩展名
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

def main():
    # 创建目标目录结构
    subsets = ["train", "val", "test"]
    for subset in subsets:
        Path(f"{DEST_DIR}/{subset}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{DEST_DIR}/{subset}/labels").mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [
        f for f in os.listdir(SOURCE_IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        print(f"错误：在 {SOURCE_IMAGE_DIR} 中未找到任何图片文件")
        return
    
    # 打乱文件顺序
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    
    # 计算每个子集的文件数量
    total = len(image_files)
    train_count = int(total * SPLIT_RATIOS[0])
    val_count = int(total * SPLIT_RATIOS[1])
    # 剩下的作为测试集
    
    # 分配文件到各个子集
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count+val_count]
    test_files = image_files[train_count+val_count:]
    
    print(f"数据集划分结果：")
    print(f"总文件数：{total}")
    print(f"训练集：{len(train_files)} 个文件")
    print(f"验证集：{len(val_files)} 个文件")
    print(f"测试集：{len(test_files)} 个文件")
    
    # 复制文件的函数
    def copy_files(file_list, subset):
        for img_file in file_list:
            # 复制图片文件
            src_img = os.path.join(SOURCE_IMAGE_DIR, img_file)
            dst_img = os.path.join(DEST_DIR, subset, "images", img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制对应的标签文件（假设标签文件与图片同名，扩展名为.txt）
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(SOURCE_LABEL_DIR, label_file)
            
            if os.path.exists(src_label):
                dst_label = os.path.join(DEST_DIR, subset, "labels", label_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告：未找到 {img_file} 对应的标签文件 {label_file}")
    
    # 复制文件到各个子集
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    print("数据集划分完成！")
    print(f"新的目录结构已创建在 {DEST_DIR} 下")

if __name__ == "__main__":
    main()
