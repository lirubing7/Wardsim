import os

def check_image_label_matching(img_dir, label_dir):
    """
    Check whether images and label files exist and whether their filenames match (ignoring extensions)
    :param img_dir: path to the image folder
    :param label_dir: path to the label folder
    :return: check result (True/False) and mismatch information
    """
    # Check whether folders exist
    if not os.path.isdir(img_dir):
        return False, f"Image folder does not exist: {img_dir}"
    if not os.path.isdir(label_dir):
        return False, f"Label folder does not exist: {label_dir}"
    
    # Get all image files (filtering common image formats)
    img_extensions = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    if not img_files:
        return False, f"No image files found in folder: {img_dir}"
    
    # Get all label files (assuming YOLO-format txt files)
    label_extension = '.txt'
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(label_extension)]
    if not label_files:
        return False, f"No label files found in folder: {label_dir}"
    
    # Extract basenames (without extensions) for matching
    img_basenames = {os.path.splitext(f)[0] for f in img_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}
    
    # Check mismatched files
    missing_labels = img_basenames - label_basenames  # Images without labels
    missing_images = label_basenames - img_basenames  # Labels without images
    
    # Generate report
    report = []
    if missing_labels:
        report.append(f"Images missing corresponding labels: {', '.join(missing_labels)}")
    if missing_images:
        report.append(f"Labels missing corresponding images: {', '.join(missing_images)}")
    
    if report:
        return False, "\n".join(report)
    else:
        return True, f"All files match! Total images: {len(img_files)}, total labels: {len(label_files)}"

# Your folder paths
yolo_file = r'/home/autodl-tmp/rawdata/labels'   # Label folder
img_file = r'/home/autodl-tmp/rawdata/images'    # Image folder

# Perform check
match, message = check_image_label_matching(img_file, yolo_file)
print("Check result:")
print(message)
