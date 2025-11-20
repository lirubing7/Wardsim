#----------------------------------------------------#
#   This file integrates single-image prediction,
#   video/camera detection, and FPS testing.
#   Mode can be switched by setting the "mode" variable.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
import os
import sys
import torch
from pathlib import Path

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import optimized depth estimation and BEV modules
from depth_map_frcnn import DepthEstimator  # Depth estimator based on Depth Anything v2
from bev_frcnn import BirdEyeView, BBox3DEstimator

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    
    # Initialize depth estimator (Depth Anything v2)
    depth_estimator = DepthEstimator(
        model_size='base',  # choices: 'small'/'base'/'large', balancing speed & accuracy
        device='cuda' if frcnn.cuda else 'cpu'  # use same device as detection model
    )
    
    # Initialize 3D bounding box estimator & BEV generator
    bbox3d_estimator = BBox3DEstimator()
    bev_generator = BirdEyeView(
        size=(300, 300),  # BEV image size
        scale=60,         # pixel/meter
        max_distance=20   # max display distance (meters)
    )
#----------------------------------------------------------------------------------------------------------#
    #   mode specifies the testing mode:
    #   'predict'           for single-image prediction (supports saving result, cropping objects, etc.)
    #   'video'             for detection on video or camera stream
    #   'fps'               to test FPS using img/street.jpg
    #   'dir_predict'       run detection on all images in a directory and save results
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                whether to crop detected objects (only in predict mode)
    #   count               whether to count detected objects (only in predict mode)
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          path to video (0 for webcam)
    #                       e.g., video_path = "xxx.mp4"
    #
    #   video_save_path     path to save output video ("" = do not save)
    #                       e.g., video_save_path = "yyy.mp4"
    #
    #   video_fps           FPS of saved video
    #
    #   These parameters only work in mode='video'.
    #   Note: to properly save video, press Ctrl+C or let it run to the end.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = "../test/input_video.mp4"
    video_save_path = "../test/output_input_video.mp4"
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       number of runs when measuring FPS; larger -> more accurate
    #   fps_image_path      image used for FPS measurement
    #
    #   Only valid when mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    # test_interval   = 100
    # fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     directory containing input images
    #   dir_save_path       directory to save output images
    #
    #   Only valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    # dir_origin_path = "img/"
    # dir_save_path   = "img_out/"
    
    if mode == "video":
        capture = cv2.VideoCapture(video_path)

         # Get original video dimensions
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (width, height))
        
            
        fps = 0.0
        depth_height = height // 4
        depth_width = min(depth_height * width // height, width)   # compute depth window width preserving aspect ratio
        # BEV window remains square (same as depth height)
        bev_size = depth_height
        
        while(True):
            t1 = time.time()

            ret, frame = capture.read()
            if not ret:
                break  # end of video

            # detection: BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert to PIL Image
            pil_frame = Image.fromarray(np.uint8(frame_rgb))
            
            # perform detection
            det_result = frcnn.detect_image(pil_frame, return_bboxes=True)
            det_frame_pil, bboxes = det_result
            det_frame = cv2.cvtColor(np.array(det_frame_pil), cv2.COLOR_RGB2BGR)
            # convert to OpenCV format and save as result_frame
            result_frame = cv2.cvtColor(np.array(det_frame_pil), cv2.COLOR_RGB2BGR)  # define result_frame

            # 2. depth estimation
            depth_map_norm = depth_estimator.estimate_depth(frame)
            colored_depth = depth_estimator.colorize_depth(depth_map_norm)

            """
            # Depth validation code
            depth_mean = depth_map_norm.mean()
            depth_non_zero_ratio = (depth_map_norm > 0.01).sum() / (depth_map_norm.shape[0] * depth_map_norm.shape[1])
            print(f"Depth validation - mean: {depth_mean:.4f}, valid pixel ratio: {depth_non_zero_ratio:.4f}")
            if depth_mean < 0.01 or depth_non_zero_ratio < 0.1:
                print("⚠️ Depth map abnormal (possibly black), check depth module")
            else:
                print("✅ Depth map normal")

            cv2.imwrite("debug_depth.png", colored_depth)
            """

            """
            # 3. Generate BEV
            bev_generator.reset()
            bev_generator.draw_depth_map(depth_map_norm)
            active_ids = []
            for bbox in bboxes:
                x1, y1, x2, y2, cls_id, score = bbox
                # Convert class ID to class name (frcnn must have class_names list)
                class_name = frcnn.class_names[cls_id]
                obj_depth = depth_estimator.get_depth_in_bbox(
                    depth_map_norm, [x1, y1, x2, y2], method='median'
                )
                box_3d = bbox3d_estimator.estimate_3d_box(
                    [x1, y1, x2, y2], obj_depth, class_name, score
                )
                bev_generator.draw_3d_box(box_3d)
                
            bev_image = bev_generator.get_image()
            """

            # -------------------------- Overlay logic: add windows --------------------------
            # Ensure depth map is 3-channel
            if len(colored_depth.shape) == 2:
                colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_GRAY2BGR)

            # Resize depth map window
            depth_resized = cv2.resize(colored_depth, (depth_width, depth_height))

            # Overlay depth map on top-right corner
            if depth_height > 0 and depth_width > 0:
                depth_roi = result_frame[0:depth_height, width - depth_width:width]
                result_frame[0:depth_height, width - depth_width:width] = depth_resized
                # Draw bounding box and title
                cv2.rectangle(result_frame,
                             (width - depth_width, 0),
                             (width, depth_height),
                             (0, 255, 0), 2)  # green border
                cv2.putText(result_frame, "Depth Map",
                           (width - depth_width + 10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            """
            # Overlay BEV on bottom-left
            if bev_size > 0:
                bev_roi = result_frame[height - bev_size:height, 0:bev_size]
                result_frame[height - bev_size:height, 0:bev_size] = bev_resized
                cv2.rectangle(result_frame,
                             (0, height - bev_size),
                             (bev_size, height),
                             (255, 0, 0), 2)  # blue border
                cv2.putText(result_frame, "BEV View",
                           (10, height - bev_size + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            """

            # compute FPS
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            """
            # Display
            #cv2.imshow("Detection (Video)", det_frame)
            #cv2.imshow("Depth Map", colored_depth)
            #cv2.imshow("BEV", bev_image)
            """

            # save video
            if video_save_path != "":
                out.write(result_frame)
         
            # ESC to exit
            #if cv2.waitKey(1) & 0xff == 27:
            #    break

        # Release resources
        capture.release()
        if video_save_path != "":
            out.release()
            
        cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
