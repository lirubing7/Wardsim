import os
import xml.dom.minidom
import glob
from PIL import Image
import shutil

# Path configuration
yolo_file = r'/root/autodl-tmp/rawdata/labels'  # YOLO-format label folder
turn_xml_file = r'/root/autodl-tmp/frcnn/faster-rcnn-pytorch-master/VOCdevkit/VOC2007/Annotations'  # Output XML folder
img_file = r'/root/autodl-tmp/rawdata/images'  # Image folder

# Class list
labels = ['monitor', 'bed', 'IV_stand', 'vent', 'pump', 'person']

# Create output folder if it does not exist
os.makedirs(turn_xml_file, exist_ok=True)

# Get all image paths
img_Lists = glob.glob(os.path.join(img_file, '*.jpg'))
img_basenames = [os.path.basename(item) for item in img_Lists]
img_names = [os.path.splitext(name)[0] for name in img_basenames]

total_num = len(img_names)
count = 0

for img in img_names:
    count += 1
    # Print progress (every 100 images or last image)
    if count % 100 == 0 or count == total_num:
        progress = (count / total_num) * 100
        print(f"Conversion progress: {progress:.1f}% ({count}/{total_num})")

    # Try to open image to get size
    try:
        img_path = os.path.join(img_file, f"{img}.jpg")
        im = Image.open(img_path)
        width, height = im.size
    except FileNotFoundError:
        print(f"Warning: image {img}.jpg does not exist, skipping.")
        continue
    except Exception as e:
        print(f"Error processing image {img}: {e}, skipping.")
        continue

    # Read YOLO label file
    txt_path = os.path.join(yolo_file, f"{img}.txt")
    try:
        with open(txt_path, 'r') as f:
            gt = f.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: label file {img}.txt does not exist, generating empty XML.")
        gt = []
    except Exception as e:
        print(f"Error reading label {img}.txt: {e}, skipping.")
        continue

    # Generate XML file
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

        # Write object bounding boxes
        for label_line in gt:
            spt = label_line.strip().split(' ')
            if len(spt) != 5:
                print(f"Label format error: invalid line in {img}.txt → {label_line}")
                continue

            try:
                class_id = int(spt[0])
                if class_id < 0 or class_id >= len(labels):
                    print(f"Invalid class ID: {class_id} in {img}.txt")
                    continue

                # Convert YOLO coords to VOC coords
                center_x = float(spt[1]) * width
                center_y = float(spt[2]) * height
                bbox_width = float(spt[3]) * width
                bbox_height = float(spt[4]) * height

                xmin = int(round(center_x - bbox_width / 2))
                ymin = int(round(center_y - bbox_height / 2))
                xmax = int(round(center_x + bbox_width / 2))
                ymax = int(round(center_y + bbox_height / 2))

                # Ensure coordinates stay within valid range
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)

                # Write object to XML
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
                print(f"Error processing label line {label_line}: {e}")
                continue

        xml_file.write('</annotation>')

print("Conversion complete!")
