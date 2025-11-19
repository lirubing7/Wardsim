import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)
from utils.utils_bbox import DecodeBox


#--------------------------------------------#
#   To use your own trained model for inference,
#   you must modify 2 parameters:
#   model_path and classes_path.
#   If a shape mismatch occurs,
#   carefully check NUM_CLASSES used during training,
#   as well as model_path and classes_path.
#--------------------------------------------#
class FRCNN(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   When using your own trained model, be sure to modify model_path and classes_path!
        #   model_path should point to the weight file under logs,
        #   classes_path should point to the txt file under model_data.
        #
        #   After training, there will be multiple weight files under logs.
        #   Choose the one with lower validation loss.
        #   Lower validation loss does not necessarily mean higher mAP;
        #   it only indicates better generalization on the validation set.
        #   If there is a shape mismatch, also check model_path and classes_path
        #   are consistent with the training configuration.
        #--------------------------------------------------------------------------#
        #"model_path"    : 'model_data/voc_weights_resnet.pth',
        "model_path"    : 'model_data/best_epoch_weights_01.pth',
        "classes_path"  : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   Backbone network for feature extraction: resnet50 or vgg
        #---------------------------------------------------------------------#
        "backbone"      : "resnet50",
        #---------------------------------------------------------------------#
        #   Only prediction boxes with score greater than confidence are kept
        #---------------------------------------------------------------------#
        "confidence"    : 0.5,
        #---------------------------------------------------------------------#
        #   IoU threshold for non-maximum suppression (NMS)
        #---------------------------------------------------------------------#
        "nms_iou"       : 0.3,
        #---------------------------------------------------------------------#
        #   Anchor scales
        #---------------------------------------------------------------------#
        'anchors_size'  : [8, 16, 32],
        #-------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU
        #-------------------------------#
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize Faster R-CNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        #---------------------------------------------------#
        #   Get class names and number of classes
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std    = self.std.cuda()
        self.bbox_util  = DecodeBox(self.std, self.num_classes)

        #---------------------------------------------------#
        #   Assign different colors for each class
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   Load network and weights
        #-------------------------------#
        self.net    = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   Inference on an image
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False, return_bboxes=False):
        #---------------------------------------------------#
        #   Get original image height and width
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------#
        #   Compute resized image size (shorter side = 600)
        #---------------------------------------------------#
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   Convert image to RGB to avoid errors with grayscale images.
        #   The code only supports RGB; other formats are converted to RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Resize original image so that its shorter side is 600
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   Add batch dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #-------------------------------------------------------------#
            #   roi_cls_locs: bbox regression outputs for proposals
            #   roi_scores   : classification scores for proposals
            #   rois         : proposal box coordinates
            #-------------------------------------------------------------#
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   Decode proposals using classifier predictions to obtain final boxes
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #---------------------------------------------------------#
            #   If no object is detected, return original image
            #---------------------------------------------------------#           
            if len(results[0]) <= 0:
                return image
                
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        #---------------------------------------------------------#
        #   Set font and bounding box thickness
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        #---------------------------------------------------------#
        #   Object counting
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   Whether to crop detected objects
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   Draw detection results
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        bboxes = []
        for i in range(len(top_label)):
            box = top_boxes[i]
            cls = top_label[i]
            score = top_conf[i]
            # Only keep bbox coordinates, class, and confidence (6 elements)
            bboxes.append((box[0], box[1], box[2], box[3], cls, score))

        if return_bboxes:
            return image, bboxes  # Note: in the original code this was mistakenly written as r_image; here it is corrected to image
        else:
            return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   Get original image height and width
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   Convert image to RGB to avoid errors with grayscale images.
        #   The code only supports RGB; other formats are converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   Resize original image so that its shorter side is 600
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   Add batch dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   Decode proposals using classifier predictions to obtain final boxes
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                #-------------------------------------------------------------#
                #   Decode proposals using classifier predictions to obtain final boxes
                #-------------------------------------------------------------#
                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   Inference for mAP calculation
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        #---------------------------------------------------#
        #   Get original image height and width
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   Convert image to RGB to avoid errors with grayscale images.
        #   The code only supports RGB; other formats are converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   Resize original image so that its shorter side is 600
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   Add batch dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   Decode proposals using classifier predictions to obtain final boxes
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   If no objects detected, return
            #--------------------------------------#
            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return
