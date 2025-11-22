#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

def run_yolo_on_video(
    source,
    output_path,
    yolo_model_size="small",
    depth_model_size="small",
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    conf_threshold=0.3,
    iou_threshold=0.8,
    classes=None,
    enable_tracking=True,
    enable_bev=None,
    enable_pseudo_3d=None,
    enable_depth_thumbnail=True,
    camera_params_file=None,
    max_frames=1000,
):
    """Run YOLO + DepthAnything + 3D/BEV pipeline on a single video file."""
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    # Initialize 3D bounding box estimator with default parameters
    bbox3d_estimator = BBox3DEstimator()
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        bev = BirdEyeView(scale=60, size=(300, 300))
    
    # Open video source
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
    except ValueError:
        pass
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("Starting processing...")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    class_name = detector.get_class_names()[class_id]
                    
                    # Depth sampling
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    use_kalman = True
                    if use_kalman:
                        box_3d = bbox3d_estimator.estimate_3d_box(
                            bbox_2d=bbox,
                            depth_value=depth_value,
                            class_name=class_name,
                            object_id=obj_id,
                            score=score
                        )
                    else:
                        box_3d = {
                            'bbox_2d': bbox,
                            'depth_value': depth_value,
                            'depth_method': depth_method,
                            'class_name': class_name,
                            'object_id': obj_id,
                            'score': score
                        }
                    
                    boxes_3d.append(box_3d)
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            bbox3d_estimator.cleanup_trackers(active_ids)
            
            # Step 4: Visualization
            for box_3d in boxes_3d:
                try:
                    class_name = box_3d['class_name'].lower()
                    if 'vent' in class_name:
                        color = (0, 0, 255)
                    elif 'person' in class_name:
                        color = (0, 255, 0)
                    elif 'monitor' in class_name:
                        color = (255, 0, 0)
                    elif 'bed' in class_name:
                        color = (255, 165, 0)
                    elif 'iv_stand' in class_name or 'iv-stand' in class_name:
                        color = (0, 255, 255)
                    elif 'pump' in class_name:
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 255)
                    
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            # Bird's Eye View
            if enable_bev:
                try:
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    
                    bev_height = height // 4
                    bev_width = bev_height
                    
                    if bev_height > 0 and bev_width > 0:
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        cv2.rectangle(result_frame, 
                                     (0, height - bev_height), 
                                     (bev_width, height), 
                                     (255, 255, 255), 1)
                        cv2.putText(result_frame, "Bird's Eye View", 
                                   (10, height - bev_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            
            # Depth thumbnail
            if enable_depth_thumbnail:
                try:
                    depth_height = height // 4
                    depth_width = depth_height * width // height
                    depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                    result_frame[0:depth_height, 0:depth_width] = depth_resized
                except Exception as e:
                    print(f"Error adding depth map to result: {e}")
            
            out.write(result_frame)
            
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames, saved to {output_path}")
            if max_frames is not None and frame_count >= max_frames:
                print("Reached max frame count, exiting...")
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    print("Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")
    return output_path


def main():
    """Keep old behaviour when running yolo_run.py directly."""
    source = "ICU_doctor.mp4"
    output_path = "output_ICU_doctor_43.mp4"
    
    run_yolo_on_video(
        source=source,
        output_path=output_path,
        yolo_model_size="small",
        depth_model_size="small",
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        conf_threshold=0.3,
        iou_threshold=0.8,
        classes=None,
        enable_tracking=True,
        enable_bev=None,
        enable_pseudo_3d=None,
        enable_depth_thumbnail=True,
        camera_params_file=None,
        max_frames=1000,
    )



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 
