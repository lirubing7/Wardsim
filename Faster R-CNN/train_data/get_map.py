import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from frcnn import FRCNN

if __name__ == "__main__":
    '''
    Recall and Precision are not area-based metrics like AP, so their values change with different
    confidence thresholds. By default, this script computes the Recall and Precision when the
    confidence threshold (Confidence) is 0.5.

    Due to the principle of mAP computation, the network needs to keep almost all prediction boxes
    in order to compute Recall and Precision under different thresholds. Therefore, the number of
    predicted boxes written into map_out/detection-results/ is usually larger than that from a
    direct predict script, in order to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode specifies what this script will do:
    #   map_mode = 0: run the full mAP pipeline (get predictions, get ground truth, compute VOC mAP).
    #   map_mode = 1: only get predictions.
    #   map_mode = 2: only get ground truth.
    #   map_mode = 3: only compute VOC mAP.
    #   map_mode = 4: use COCO API to compute mAP@0.50:0.95. Requires predictions, ground truth,
    #                 and pycocotools installed.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   classes_path: classes to be evaluated for VOC mAP.
    #   Usually the same as the classes_path used in training and inference.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP sets which mAP@0.x you want to compute.
    #   For example, to compute mAP@0.75, set MINOVERLAP = 0.75.
    #
    #   When IoU between a predicted box and a ground-truth box is higher than MINOVERLAP,
    #   that prediction is considered a true positive, otherwise a false positive.
    #   The higher MINOVERLAP is, the stricter the IoU requirement and the lower the resulting mAP.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   Due to the mAP calculation principle, the network needs to keep nearly all prediction boxes
    #   when computing mAP. Therefore, the confidence should be set as low as possible to keep
    #   all potential predictions.
    #
    #   This value is generally not changed. For mAP computation, confidence should stay low.
    #   To analyze Recall and Precision under different thresholds, modify score_threhold instead.
    #--------------------------------------------------------------------------------------#
    confidence      = 0.02
    #--------------------------------------------------------------------------------------#
    #   Non-maximum suppression IoU used during prediction.
    #   Larger values mean NMS is less strict.
    #
    #   This value is generally not changed.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall and Precision are not area-based metrics like AP, so they depend on the threshold.
    #
    #   By default, this script reports Recall and Precision at threshold 0.5
    #   (defined below as score_threhold).
    #   Since mAP requires nearly all predictions, the confidence above cannot be changed freely.
    #   We use a separate score_threhold to indicate the threshold used when looking up
    #   Recall and Precision corresponding to that point.
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis: whether to enable visualization for VOC mAP
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   Path to the VOC dataset folder.
    #   By default points to the VOC dataset under the project root.
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   Output folder, default is map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
