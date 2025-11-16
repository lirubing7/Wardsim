import os
import xml.dom.minidom
import glob
from PIL import Image
import shutil

# 路径配置
yolo_file = r'/root/autodl-tmp/rawdata/labels'  # YOLO格式标签文件夹
turn_xml_file = r'/root/autodl-tmp/frcnn/faster-rcnn-pytorch-master/VOCdevkit/VOC2007/Annotations'  # 输出XML文件夹
img_file = r'/root/autodl-tmp/rawdata/images'  # 图片文件夹

# 类别列表
labels = ['monitor', 'bed', 'IV_stand', 'vent', 'pump', 'person']

# 创建输出文件夹（若不存在）
os.makedirs(turn_xml_file, exist_ok=True)

# 获取所有图片路径
img_Lists = glob.glob(os.path.join(img_file, '*.jpg'))
img_basenames = [os.path.basename(item) for item in img_Lists]
img_names = [os.path.splitext(name)[0] for name in img_basenames]

total_num = len(img_names)
count = 0

for img in img_names:
    count += 1
    # 打印进度（每100张或最后一张）
    if count % 100 == 0 or count == total_num:
        progress = (count / total_num) * 100
        print(f"转换进度：{progress:.1f}% ({count}/{total_num})")

    # 尝试打开图片获取尺寸
    try:
        img_path = os.path.join(img_file, f"{img}.jpg")
        im = Image.open(img_path)
        width, height = im.size
    except FileNotFoundError:
        print(f"警告：图片 {img}.jpg 不存在，跳过")
        continue
    except Exception as e:
        print(f"处理图片 {img} 出错：{e}，跳过")
        continue

    # 读取YOLO标签文件
    txt_path = os.path.join(yolo_file, f"{img}.txt")
    try:
        with open(txt_path, 'r') as f:
            gt = f.read().splitlines()
    except FileNotFoundError:
        print(f"警告：标签文件 {img}.txt 不存在，生成空XML")
        gt = []
    except Exception as e:
        print(f"读取标签 {img}.txt 出错：{e}，跳过")
        continue

    # 生成XML文件
    xml_path = os.path.join(turn_xml_file, f"{img}.xml")
    with open(xml_path, 'w') as xml_file:
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write(f'    <filename>{img}.jpg</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write(f'        <width>{width}</width>\n')
        xml_file.write(f'        <height>{height}</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # 写入目标边界框
        for label_line in gt:
            spt = label_line.strip().split(' ')
            if len(spt) != 5:
                print(f"标签格式错误：{img}.txt 中该行无效：{label_line}")
                continue

            try:
                class_id = int(spt[0])
                if class_id < 0 or class_id >= len(labels):
                    print(f"无效类别ID：{class_id} 在 {img}.txt 中")
                    continue
                # 转换YOLO坐标到VOC坐标
                center_x = float(spt[1]) * width
                center_y = float(spt[2]) * height
                bbox_width = float(spt[3]) * width
                bbox_height = float(spt[4]) * height

                xmin = int(round(center_x - bbox_width / 2))
                ymin = int(round(center_y - bbox_height / 2))
                xmax = int(round(center_x + bbox_width / 2))
                ymax = int(round(center_y + bbox_height / 2))

                # 确保坐标在有效范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)

                # 写入XML
                xml_file.write('    <object>\n')
                xml_file.write(f'        <name>{labels[class_id]}</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write(f'            <xmin>{xmin}</xmin>\n')
                xml_file.write(f'            <ymin>{ymin}</ymin>\n')
                xml_file.write(f'            <xmax>{xmax}</xmax>\n')
                xml_file.write(f'            <ymax>{ymax}</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
            except Exception as e:
                print(f"处理标签行出错：{label_line}，错误：{e}")
                continue

        xml_file.write('</annotation>')

print("转换完成！")

