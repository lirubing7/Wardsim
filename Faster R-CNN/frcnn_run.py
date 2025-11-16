#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
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

# 导入优化后的深度估计和鸟瞰图模块
from depth_map_frcnn import DepthEstimator  # 基于Depth Anything v2的深度估计器
from bev_frcnn import BirdEyeView, BBox3DEstimator

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    
    # 初始化深度估计器（Depth Anything v2）
    depth_estimator = DepthEstimator(
        model_size='base',  # 可选'small'/'base'/'large'，平衡速度与精度
        device='cuda' if frcnn.cuda else 'cpu'  # 与检测模型共享设备
    )
    
    # 初始化3D边界框估计器和鸟瞰图生成器
    bbox3d_estimator = BBox3DEstimator()
    bev_generator = BirdEyeView(
        size=(300, 300),  # BEV图像尺寸
        scale=60,         # 像素/米
        max_distance=20   # 最大显示距离（米）
    )
#----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = "test/input_video.mp4"
    video_save_path = "test/output_input_video.mp4"
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    
    if mode == "video":
        capture = cv2.VideoCapture(video_path)

         # 获取原视频尺寸
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (width, height))
        
            
        fps = 0.0
        depth_height = height // 4
        depth_width = min(depth_height * width // height, width)   # 按原图宽高比计算宽度
        # BEV小窗口保持正方形（与深度图高度一致）
        bev_size = depth_height
        
        while(True):
            t1 = time.time()

            ret, frame = capture.read()
            if not ret:
                break  # 视频结束

            # detection 格式转变，BGRtoRGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            pil_frame = Image.fromarray(np.uint8(frame_rgb))
            
            # 进行检测
            det_result = frcnn.detect_image(pil_frame, return_bboxes=True)
            det_frame_pil, bboxes = det_result
            det_frame = cv2.cvtColor(np.array(det_frame_pil), cv2.COLOR_RGB2BGR)
            # 转换为OpenCV格式并赋值给 result_frame
            result_frame = cv2.cvtColor(np.array(det_frame_pil), cv2.COLOR_RGB2BGR)  # 定义 result_frame

            # 2. 深度估计
            depth_map_norm = depth_estimator.estimate_depth(frame)
            colored_depth = depth_estimator.colorize_depth(depth_map_norm)
            """
            depth_mean = depth_map_norm.mean()
            depth_non_zero_ratio = (depth_map_norm > 0.01).sum() / (depth_map_norm.shape[0] * depth_map_norm.shape[1])
            print(f"深度图数据验证 - 均值: {depth_mean:.4f}, 有效像素占比: {depth_non_zero_ratio:.4f}")
            if depth_mean < 0.01 or depth_non_zero_ratio < 0.1:
                print("⚠️ 深度图数据异常（可能全黑），检查深度估计模块")
            else:
                print("✅ 深度图数据正常")

            # 2. 保存单帧深度图到本地，直观确认
            colored_depth = depth_estimator.colorize_depth(depth_map_norm)
            cv2.imwrite("debug_depth.png", colored_depth)  # 保存到当前目录
            #print("✅ 深度图已保存为 debug_depth.png，打开文件确认是否正常")
            """
            """
            # 3. 生成BEV
            bev_generator.reset()
            bev_generator.draw_depth_map(depth_map_norm)
            active_ids = []
            for bbox in bboxes:
                x1, y1, x2, y2, cls_id, score = bbox
                # 将类别ID转换为字符串名称（假设frcnn有class_names列表）
                class_name = frcnn.class_names[cls_id]  # 关键：根据实际类别映射修改
                obj_depth = depth_estimator.get_depth_in_bbox(
                    depth_map_norm, [x1, y1, x2, y2], method='median'
                )
                box_3d = bbox3d_estimator.estimate_3d_box(
                    [x1, y1, x2, y2], obj_depth, class_name, score
                )
                bev_generator.draw_3d_box(box_3d)
                
            bev_image = bev_generator.get_image()

            """
            # -------------------------- 合并逻辑：叠加小窗口 --------------------------
            # 确保深度图和BEV图为3通道（防止单通道报错）
            if len(colored_depth.shape) == 2:
                colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_GRAY2BGR)
            #if len(bev_image.shape) == 2:
            #    bev_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)

            # 调整小窗口尺寸
            depth_resized = cv2.resize(colored_depth, (depth_width, depth_height))
            #bev_resized = cv2.resize(bev_image, (bev_size, bev_size))

            # 叠加深度图到主帧右上角
            if depth_height > 0 and depth_width > 0:
                depth_roi = result_frame[0:depth_height, width - depth_width:width]
                result_frame[0:depth_height, width - depth_width:width] = depth_resized
                # 绘制深度图边框和标题
                cv2.rectangle(result_frame,
                             (width - depth_width, 0),
                             (width, depth_height),
                             (0, 255, 0), 2)  # 绿色边框
                cv2.putText(result_frame, "Depth Map",
                           (width - depth_width + 10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            """
            # 叠加BEV图到主帧左下角
            if bev_size > 0:
                bev_roi = result_frame[height - bev_size:height, 0:bev_size]
                result_frame[height - bev_size:height, 0:bev_size] = bev_resized
                # 绘制BEV边框和标题
                cv2.rectangle(result_frame,
                             (0, height - bev_size),
                             (bev_size, height),
                             (255, 0, 0), 2)  # 蓝色边框
                cv2.putText(result_frame, "BEV View",
                           (10, height - bev_size + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            # 计算FPS
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            """
            # 显示
            #cv2.imshow("目标检测（视频）", det_frame)
            #cv2.imshow("深度图", colored_depth)
            #cv2.imshow("鸟瞰图", bev_image)

            # 保存视频
            if video_save_path != "":
                out.write(result_frame)
         
            # 按ESC退出
            #if cv2.waitKey(1) & 0xff == 27:
            #    break

        # 释放资源
        capture.release()
        if video_save_path != "":
            out.release()
            
        cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")