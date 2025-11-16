import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math
from collections import defaultdict
from filterpy.kalman import KalmanFilter

LOW_SCORE = 0.25

# Default camera intrinsic matrix (can be overridden)
DEFAULT_K = np.array([
    [718.856, 0.0, 607.1928],
    [0.0, 718.856, 185.2157],
    [0.0, 0.0, 1.0]
])

# Default camera projection matrix (can be overridden)
DEFAULT_P = np.array([
    [718.856, 0.0, 607.1928, 45.38225],
    [0.0, 718.856, 185.2157, -0.1130887],
    [0.0, 0.0, 1.0, 0.003779761]
])

# Average dimensions for common objects (height, width, length) in meters
DEFAULT_DIMS = {
    'car': np.array([1.52, 1.64, 3.85]),
    'truck': np.array([3.07, 2.63, 11.17]),
    'bus': np.array([3.07, 2.63, 11.17]),
    'motorcycle': np.array([1.50, 0.90, 2.20]),
    'bicycle': np.array([1.40, 0.70, 1.80]),
    'person': np.array([1.75, 0.60, 0.60]),  # Adjusted width/length for person
    'dog': np.array([0.80, 0.50, 1.10]),
    'cat': np.array([0.40, 0.30, 0.70]),
    # Add indoor objects
    'potted plant': np.array([0.80, 0.40, 0.40]),  # Reduced size for indoor plants
    'plant': np.array([0.80, 0.40, 0.40]),  # Alias for potted plant
    'chair': np.array([0.80, 0.60, 0.60]),
    'sofa': np.array([0.80, 0.85, 2.00]),
    'table': np.array([0.75, 1.20, 1.20]),
    'bed': np.array([0.60, 1.50, 2.00]),
    'tv': np.array([0.80, 0.15, 1.20]),
    'laptop': np.array([0.02, 0.25, 0.35]),
    'keyboard': np.array([0.03, 0.15, 0.45]),
    'mouse': np.array([0.03, 0.06, 0.10]),
    'book': np.array([0.03, 0.20, 0.15]),
    'bottle': np.array([0.25, 0.10, 0.10]),
    'cup': np.array([0.10, 0.08, 0.08]),
    'vase': np.array([0.30, 0.15, 0.15]),
    'IV_stand': np.array([1.50, 0.20, 0.10]),
    'vent': np.array([1.40, 0.30, 0.30]),
    'monitor': np.array([0.40, 0.40, 0.20]),
    'pump': np.array([0.20, 0.20, 0.10]),
    'staff': np.array([1.75, 0.60, 0.60])
}

class BBox3DEstimator:
    """
    3D bounding box estimation from 2D detections and depth
    """
    def __init__(self, camera_matrix=None, projection_matrix=None, class_dims=None):
        """
        Initialize the 3D bounding box estimator
        
        Args:
            camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
            projection_matrix (numpy.ndarray): Camera projection matrix (3x4)
            class_dims (dict): Dictionary mapping class names to dimensions (height, width, length)
        """
        self.K = camera_matrix if camera_matrix is not None else DEFAULT_K
        self.P = projection_matrix if projection_matrix is not None else DEFAULT_P
        self.dims = class_dims if class_dims is not None else DEFAULT_DIMS
        
        # Initialize Kalman filters for tracking 3D boxes
        self.kf_trackers = {}
        
        # Store history of 3D boxes for filtering
        self.box_history = defaultdict(list)
        self.max_history = 5
    

    def estimate_3d_box(self, bbox_2d, depth_value, class_name, object_id=None, score=0.9):
        """
        Estimate 3D bounding box from 2D bounding box and depth
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            depth_value (float): Depth value at the center of the bounding box
            class_name (str): Class name of the object
            object_id (int): Object ID for tracking (None for no tracking)
            
        Returns:
            dict: 3D bounding box parameters
        """
        # Get 2D box center and dimensions
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Get dimensions for the class
        if class_name.lower() in self.dims:
            dimensions = self.dims[class_name.lower()].copy()  # Make a copy to avoid modifying the original
        else:
            # Use default car dimensions if class not found
            dimensions = self.dims['car'].copy()
        
        # Adjust dimensions based on 2D box aspect ratio and size
        aspect_ratio_2d = width_2d / height_2d if height_2d > 0 else 1.0
        
        # For plants, adjust dimensions based on 2D box
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Scale height based on 2D box height
            dimensions[0] = height_2d / 120  # Convert pixels to meters with a scaling factor
            # Make width and length proportional to height
            dimensions[1] = dimensions[0] * 0.6  # width
            dimensions[2] = dimensions[0] * 0.6  # length
        
        # For people, adjust dimensions based on 2D box
        elif 'person' in class_name.lower():
            # Scale height based on 2D box height
            dimensions[0] = height_2d / 100  # Convert pixels to meters with a scaling factor
            # Make width and length proportional to height
            dimensions[1] = dimensions[0] * 0.3  # width
            dimensions[2] = dimensions[0] * 0.3  # length
        
        # Convert depth to distance - use a larger range for better visualization
        # Map depth_value (0-1) to a range of 1-10 meters
        distance = 1.0 + depth_value * 4.0  # Increased from 4.0 to 9.0 for a larger range
        
        # Calculate 3D location
        location = self._backproject_point(center_x, center_y, distance)
        
        # For plants, adjust y-coordinate to place them on a surface
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Assume plants are on a surface (e.g., table, floor)
            # Adjust y-coordinate based on the bottom of the 2D bounding box
            bottom_y = y2  # Bottom of the 2D box
            location[1] = self._backproject_point(center_x, bottom_y, distance)[1]
        
        # Estimate orientation
        orientation = self._estimate_orientation(bbox_2d, location, class_name)
        
        # Create 3D box
        box_3d = {
            'dimensions': dimensions,
            'location': location,
            'orientation': orientation,
            'bbox_2d': bbox_2d,
            'object_id': object_id,
            'class_name': class_name,
            'score': score
        }
        
        # Apply Kalman filtering if tracking is enabled
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
            
            # Add to history for temporal filtering
            self.box_history[object_id].append(box_3d)
            if len(self.box_history[object_id]) > self.max_history:
                self.box_history[object_id].pop(0)
            
            # Apply temporal filtering
            box_3d = self._apply_temporal_filter(object_id)
        
        return box_3d
    
    def _backproject_point(self, x, y, depth):
        """
        Backproject a 2D point to 3D space
        
        Args:
            x (float): X coordinate in image space
            y (float): Y coordinate in image space
            depth (float): Depth value
            
        Returns:
            numpy.ndarray: 3D point (x, y, z) in camera coordinates
        """
        # Create homogeneous coordinates
        point_2d = np.array([x, y, 1.0])
        
        # Backproject to 3D
        # The z-coordinate is the depth
        # The x and y coordinates are calculated using the inverse of the camera matrix
        point_3d = np.linalg.inv(self.K) @ point_2d * depth
        
        # For indoor scenes, adjust the y-coordinate to be more realistic
        # In camera coordinates, y is typically pointing down
        # Adjust y to place objects at a reasonable height
        # This is a simplification - in a real system, this would be more sophisticated
        point_3d[1] = point_3d[1] * 0.5  # Scale down y-coordinate
        
        return point_3d
    
    def _estimate_orientation(self, bbox_2d, location, class_name):
        """
        Estimate orientation of the object
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            location (numpy.ndarray): 3D location of the object
            class_name (str): Class name of the object
            
        Returns:
            float: Orientation angle in radians
        """
        # Calculate ray from camera to object center
        theta_ray = np.arctan2(location[0], location[2])
        
        # For plants and stationary objects, orientation doesn't matter much
        # Just use a fixed orientation aligned with the camera view
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Plants typically don't have a specific orientation
            # Just use the ray angle
            return theta_ray
        
        # For people, they might be facing the camera
        if 'person' in class_name.lower():
            # Assume person is facing the camera
            alpha = 0.0
        else:
            # For other objects, use the 2D box aspect ratio to estimate orientation
            x1, y1, x2, y2 = bbox_2d
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            
            # If the object is wide, it might be facing sideways
            if aspect_ratio > 1.5:
                # Object is wide, might be facing sideways
                # Use the position relative to the image center to guess orientation
                image_center_x = self.K[0, 2]  # Principal point x
                if (x1 + x2) / 2 < image_center_x:
                    # Object is on the left side of the image
                    alpha = np.pi / 2  # Facing right
                else:
                    # Object is on the right side of the image
                    alpha = -np.pi / 2  # Facing left
            else:
                # Object has normal proportions, assume it's facing the camera
                alpha = 0.0
        
        # Global orientation
        rot_y = alpha + theta_ray
        
        return rot_y
    
    def _init_kalman_filter(self, box_3d):
        """
        Initialize a Kalman filter for a new object
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            filterpy.kalman.KalmanFilter: Initialized Kalman filter
        """
        # State: [x, y, z, width, height, length, yaw, vx, vy, vz, vyaw]
        kf = KalmanFilter(dim_x=11, dim_z=7)
        
        # Initial state
        kf.x = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation'],
            0, 0, 0, 0  # Initial velocities
        ])
        
        # State transition matrix (motion model)
        dt = 1.0  # Time step
        kf.F = np.eye(11)
        kf.F[0, 7] = dt  # x += vx * dt
        kf.F[1, 8] = dt  # y += vy * dt
        kf.F[2, 9] = dt  # z += vz * dt
        kf.F[6, 10] = dt  # yaw += vyaw * dt
        
        # Measurement function
        kf.H = np.zeros((7, 11))
        kf.H[0, 0] = 1  # x
        kf.H[1, 1] = 1  # y
        kf.H[2, 2] = 1  # z
        kf.H[3, 3] = 1  # width
        kf.H[4, 4] = 1  # height
        kf.H[5, 5] = 1  # length
        kf.H[6, 6] = 1  # yaw
        
        # Measurement uncertainty
        kf.R = np.eye(7) * 0.1
        kf.R[0:3, 0:3] *= 1.0  # Location uncertainty
        kf.R[3:6, 3:6] *= 0.1  # Dimension uncertainty
        kf.R[6, 6] = 0.3  # Orientation uncertainty
        
        # Process uncertainty
        kf.Q = np.eye(11) * 0.1
        kf.Q[7:11, 7:11] *= 0.5  # Velocity uncertainty
        
        # Initial state uncertainty
        kf.P = np.eye(11) * 1.0
        kf.P[7:11, 7:11] *= 10.0  # Velocity uncertainty
        
        return kf
    
    def _apply_kalman_filter(self, box_3d, object_id):
        """
        Apply Kalman filtering to smooth 3D box parameters
        
        Args:
            box_3d (dict): 3D bounding box parameters
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Filtered 3D bounding box parameters
        """
        #print(f"[滤波启用] 对目标ID={object_id}应用卡尔曼滤波")  # 新增调试打印

        # Initialize Kalman filter if this is a new object
        if object_id not in self.kf_trackers:
            self.kf_trackers[object_id] = self._init_kalman_filter(box_3d)
        
        # Get the Kalman filter for this object
        kf = self.kf_trackers[object_id]
        
        # Predict
        kf.predict()
        
        # Update with measurement
        measurement = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation']
        ])
        
        kf.update(measurement)
        
        # Update box_3d with filtered values
        filtered_box = box_3d.copy()
        filtered_box['location'] = np.array([kf.x[0], kf.x[1], kf.x[2]])
        filtered_box['dimensions'] = np.array([kf.x[4], kf.x[3], kf.x[5]])  # height, width, length
        filtered_box['orientation'] = kf.x[6]
        
        return filtered_box
    
    def _apply_temporal_filter(self, object_id):
        """
        Apply temporal filtering to smooth 3D box parameters over time
        
        Args:
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Temporally filtered 3D bounding box parameters
        """
        #print(f"[滤波启用] 对目标ID={object_id}应用时序平滑")  # 新增调试打印

        history = self.box_history[object_id]
        
        if len(history) < 2:
            return history[-1]
        
        # Get the most recent box
        current_box = history[-1]
        
        # Apply exponential moving average to location and orientation
        alpha = 0.7  # Weight for current measurement (higher = less smoothing)
        
        # Initialize with current values
        filtered_box = current_box.copy()
        
        # Apply EMA to location and orientation
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            filtered_box['location'] = filtered_box['location'] * (1 - weight) + history[i]['location'] * weight
            
            # Handle orientation wrapping
            angle_diff = history[i]['orientation'] - filtered_box['orientation']
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            filtered_box['orientation'] += angle_diff * weight
        
        return filtered_box
    
    def project_box_3d_to_2d(self, box_3d):
        """
        Project 3D bounding box corners to 2D image space
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            numpy.ndarray: 2D points of the 3D box corners (8x2)
        """
        # Extract parameters
        h, w, l = box_3d['dimensions']
        x, y, z = box_3d['location']
        rot_y = box_3d['orientation']
        class_name = box_3d['class_name'].lower()
        
        # Get 2D box for reference
        x1, y1, x2, y2 = box_3d['bbox_2d']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Create rotation matrix
        R_mat = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        
        # 3D bounding box corners
        # For plants and stationary objects, make the box more centered
        if 'plant' in class_name or 'potted plant' in class_name:
            # For plants, center the box on the plant
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])  # Center vertically
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        else:
            # For other objects, use standard box configuration
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])  # Bottom at y=0
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        
        # Rotate and translate corners
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = R_mat @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        
        # Project to 2D
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        corners_2d_homo = self.P @ corners_3d_homo
        corners_2d = corners_2d_homo[:2, :] / corners_2d_homo[2, :]
        
        # Constrain the 3D box to be within a reasonable distance of the 2D box
        # This helps prevent wildly incorrect projections
        mean_x = np.mean(corners_2d[0, :])
        mean_y = np.mean(corners_2d[1, :])
        
        # If the projected box is too far from the 2D box center, adjust it
        if abs(mean_x - center_x) > width_2d or abs(mean_y - center_y) > height_2d:
            # Shift the projected points to center on the 2D box
            shift_x = center_x - mean_x
            shift_y = center_y - mean_y
            corners_2d[0, :] += shift_x
            corners_2d[1, :] += shift_y
        
        return corners_2d.T
    

    def draw_box_3d(self, image, box_3d, color=(0, 255, 0), thickness=2):
        """
        Draw enhanced 3D bounding box on image with better depth perception
        
        Args:
            image (numpy.ndarray): Image to draw on
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with 3D box drawn
        """
        # 新增：检查置信度，低于0.4则不绘制
        score = box_3d.get('score', 0.0)
        if score < LOW_SCORE:
            return image  # 直接返回原图，不绘制边框

        # -------------------------- 新增：超出屏幕判断（仅针对 monitor）--------------------------
        # 1. 获取图像尺寸（height, width, channels）
        #img_h, img_w = image.shape[:2]  # 关键：图像的高（y轴范围0~img_h-1）、宽（x轴范围0~img_w-1）

        # 2. 提取 monitor 的 2D 边界框（x1,y1=左上，x2,y2=右下）
        #class_name = box_3d['class_name']
        #if class_name == "monitor":  # 仅对 monitor 执行超出判断
        #    x1, y1, x2, y2 = box_3d['bbox_2d']

            # 3. 定义 2D 边界框的 4 个顶点（左上、右上、左下、右下）
        #   vertices_2d = [
        #       (x1, y1),  # 左上（top-left）
        #        (x2, y1),  # 右上（top-right）
        #        (x1, y2),  # 左下（bottom-left）
        #        (x2, y2)   # 右下（bottom-right）
        #    ]

            # 4. 判断是否有顶点超出屏幕（x<0 或 x>=img_w；y<0 或 y>=img_h）
            # 注：图像坐标范围是 [0, img_w-1] 和 [0, img_h-1]，>=img_w 或 >=img_h 即超出
        #   is_out_of_screen = any(
        #        (x < 0.0 or x > (img_w - 1.0)) or (y < 0.0 or y > (img_h - 1.0))
        #        for x, y in vertices_2d
        #    )

            # 5. 若超出屏幕，直接返回原图（不绘制 3D 边框）
        #   if is_out_of_screen:
        #       #print(f"Warning: monitor (ID: {box_3d.get('object_id', 'unknown')}) is out of screen, skip 3D box drawing.")
        #       #print(f"out of screen Score: {score}")
        #       return image
        # -------------------------- 超出判断逻辑结束 --------------------------

        # Get 2D box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box_3d['bbox_2d']]
        
        # Get depth value for scaling
        depth_value = box_3d.get('depth_value', 0.5)
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # 新增：防止边界框尺寸异常（如 width/height <=0，避免后续计算出错）
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid 2D box size (w={width}, h={height}) for {class_name}, skip 3D box drawing.")
            return image

        # Calculate the offset for the 3D effect (deeper objects have smaller offset)
        # Inverse relationship with depth - closer objects have larger offset
        offset_factor = 1.0 - depth_value
        offset_x = int(width * 0.3 * offset_factor)
        offset_y = int(height * 0.3 * offset_factor)
        
        # Ensure minimum offset for visibility
        offset_x = max(15, min(offset_x, 50))
        offset_y = max(15, min(offset_y, 50))
        
        # Create points for the 3D box
        # Front face (the 2D bounding box)
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        
        # Back face (offset by depth)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        
        front_vertices = [
            (x1, y1),  # 正面左上
            (x2, y1),  # 正面右上
            (x2, y2),  # 正面右下
            (x1, y2)   # 正面左下
        ]   
    
        # 3. 计算背面顶点
        back_vertices = [
            (x1 + offset_x, y1 - offset_y),  # back_tl
            (x2 + offset_x, y1 - offset_y),  # back_tr
            (x2 + offset_x, y2 - offset_y),  # back_br
            (x1 + offset_x, y2 - offset_y)   # back_bl
        ]   


        # 4. 关键：判断3D框是否出界
        # 合并正面和背面所有顶点（共8个顶点）
        #all_vertices = front_vertices + back_vertices

        # 检查每个顶点是否超出图像范围
        #for (x, y) in all_vertices:
            # 注意：x和y可能是浮点坐标，直接用原始值判断（不转int，避免精度损失）
        #    if x < 0 or y < 0 :
                #print(f"[3D框出界] ID:{box_3d.get('object_id', 'unknown')} 部分顶点超出图像范围，跳过绘制")
        #        return image  # 出界则不绘制，直接返回原图

        # Create a slightly transparent copy of the image for the 3D effect
        overlay = image.copy()
        
        # Draw the front face (2D bounding box)
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        
        # Draw the connecting lines between front and back faces
        cv2.line(image, front_tl, back_tl, color, thickness)
        cv2.line(image, front_tr, back_tr, color, thickness)
        cv2.line(image, front_br, back_br, color, thickness)
        cv2.line(image, front_bl, back_bl, color, thickness)
        
        # Draw the back face
        cv2.line(image, back_tl, back_tr, color, thickness)
        cv2.line(image, back_tr, back_br, color, thickness)
        cv2.line(image, back_br, back_bl, color, thickness)
        cv2.line(image, back_bl, back_tl, color, thickness)
        
        # Fill the top face with a semi-transparent color to enhance 3D effect
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        pts_top = pts_top.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        
        # Fill the right face with a semi-transparent color
        pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        pts_right = pts_right.reshape((-1, 1, 2))
        # Darken the right face color for better 3D effect
        right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.fillPoly(overlay, [pts_right], right_color)
        
        # Apply the overlay with transparency
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Get class name and object ID
        class_name = box_3d['class_name']
        obj_id = box_3d['object_id'] if 'object_id' in box_3d else None
        
        # Draw text information
        text_y = y1 - 10
        
        # 新增：确保文字不超出图像顶部（text_y < 0 时向下偏移）
        text_y = max(20, text_y)  # 文字顶部至少留 20 像素空间
        
        if obj_id is not None:
            cv2.putText(image, f"ID:{obj_id}", (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        
        cv2.putText(image, class_name, (x1, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        text_y -= 15
        
        # Get depth information if available
        if 'depth_value' in box_3d:
            depth_value = box_3d['depth_value']
            depth_method = box_3d.get('depth_method', 'unknown')
            depth_text = f"D:{depth_value:.2f} ({depth_method})"
            cv2.putText(image, depth_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        

        # Get score if available
        if 'score' in box_3d:
            score = box_3d['score']
            score_text = f"S:{score:.2f}"
            cv2.putText(image, score_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw a vertical line from the bottom of the box to the ground
        # This helps with depth perception
        #ground_y = y2 + int(height * 0.2)  # A bit below the bottom of the box
        #cv2.line(image, (int((x1 + x2) / 2), y2), (int((x1 + x2) / 2), ground_y), color, thickness)
        
        # Draw a small circle at the bottom to represent the ground contact point
        #cv2.circle(image, (int((x1 + x2) / 2), ground_y), thickness * 2, color, -1)
        
        return image
    
    
class BirdEyeView:
    """
    优化后的鸟瞰图可视化工具，支持3D目标的精确投影与动态跟踪
    """
    def __init__(self, 
                 size=(800, 600),  # 加宽尺寸以显示更多细节
                 scale=40,         # 像素/米（增大以提升近距离精度）
                 camera_height=1.6,  # 相机离地高度（米）
                 max_distance=30,   # 最大显示距离（米）
                 camera_pitch=0.087  # 相机俯仰角（弧度，默认~5°）
                ):
        """
        初始化鸟瞰图参数
        
        Args:
            size: BEV图像尺寸 (宽, 高)
            scale: 缩放因子（像素/米）
            camera_height: 相机离地高度（米）
            max_distance: 最大显示距离（米）
            camera_pitch: 相机俯仰角（弧度，向下为正）
        """
        self.width, self.height = size
        self.scale = scale
        self.camera_height = camera_height
        self.max_distance = max_distance
        self.camera_pitch = camera_pitch  # 用于修正深度投影
        
        # 原点设置在图像底部中心（相机位置）
        self.origin_x = self.width // 2
        self.origin_y = self.height - 50  # 底部留50像素空间
        
        # 颜色映射（按类别区分）
        self.class_colors = {
            'car': (0, 0, 255),        # 红色
            'truck': (0, 165, 255),    # 橙色
            'bus': (0, 140, 255),      # 暗橙
            'person': (0, 255, 0),     # 绿色
            'bicycle': (255, 0, 0),    # 蓝色
            'motorcycle': (255, 105, 180),  # 粉红
            'potted plant': (0, 255, 255),  # 青色
            'chair': (128, 0, 128),    # 紫色
            'table': (255, 165, 0),    # 橙黄
            'default': (255, 255, 255) # 白色
        }
        
        # 初始化BEV图像
        self.bev_image = self._create_background()
        
        
    def _create_background(self):
        """创建带网格和坐标系的背景"""
        bev = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bev[:, :] = (30, 30, 30)  # 深灰色背景
        
        # 绘制网格线（每1米）
        grid_step = self.scale  # 1米对应scale像素
        for y in range(0, self.height, grid_step):
            cv2.line(bev, (0, y), (self.width, y), (60, 60, 60), 1)
        for x in range(0, self.width, grid_step):
            cv2.line(bev, (x, 0), (x, self.height), (60, 60, 60), 1)
        
        # 绘制坐标轴（X:前向，Y:侧向）
        axis_len = min(100, self.height // 4)
        # X轴（前向，绿色）
        cv2.line(bev, 
                (self.origin_x, self.origin_y), 
                (self.origin_x, self.origin_y - axis_len), 
                (0, 200, 0), 2)
        # Y轴（侧向，红色）
        cv2.line(bev, 
                (self.origin_x, self.origin_y), 
                (self.origin_x + axis_len, self.origin_y), 
                (0, 0, 200), 2)
        # 轴标签
        cv2.putText(bev, "X (前向)", (self.origin_x - 30, self.origin_y - axis_len - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        cv2.putText(bev, "Y (侧向)", (self.origin_x + axis_len + 5, self.origin_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        
        # 绘制距离刻度（每5米）
        for dist in range(5, self.max_distance + 1, 5):
            y = self.origin_y - int(dist * self.scale)
            if y < 20:  # 超出图像范围则跳过
                continue
            # 刻度线
            cv2.line(bev, (self.origin_x - 5, y), (self.origin_x + 5, y), (150, 150, 150), 2)
            # 距离文本
            cv2.putText(bev, f"{dist}m", (self.origin_x - 20, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return bev

    def reset(self):
        """重置BEV图像（保留背景，清除目标）"""
        self.bev_image = self._create_background()

    def _3d_to_bev(self, location_3d):
        """
        将3D相机坐标转换为BEV像素坐标
        
        Args:
            location_3d: 3D坐标 (x, y, z)（相机坐标系：x右, y下, z前）
        
        Returns:
            (bev_x, bev_y): BEV像素坐标
        """
        x, y, z = location_3d
        
        # 1. 修正相机俯仰角影响（将z（前向）转换为水平距离）
        # 相机坐标系中，z是沿光轴方向，需转换为水平地面距离
        horizontal_dist = z * np.cos(self.camera_pitch) + y * np.sin(self.camera_pitch)
        
        # 2. 限制最大距离
        if horizontal_dist > self.max_distance:
            return None, None  # 超出范围不显示
        
        # 3. 转换为BEV坐标
        # BEV中Y轴（上下）对应前向距离，X轴（左右）对应侧向距离
        bev_y = self.origin_y - int(horizontal_dist * self.scale)
        bev_x = self.origin_x + int(x * self.scale)  # x为侧向偏移
        
        # 4. 确保坐标在图像范围内
        if not (0 <= bev_x < self.width and 0 <= bev_y < self.height):
            return None, None
        
        return bev_x, bev_y

    def draw_3d_box(self, box_3d):
        """
        绘制3D目标的鸟瞰图投影（包含朝向和尺寸）
        
        Args:
            box_3d: 3D边界框字典，包含：
                - location: 3D坐标 (x, y, z)
                - dimensions: 尺寸 (height, width, length)
                - orientation: 朝向角（弧度）
                - class_name: 类别名称
                - object_id: 目标ID（用于跟踪）
        """
        try:
            
            # 提取3D参数
            loc = box_3d['location']
            print(f"绘制3D框, 位置: {loc}")  # 新增
            dims = box_3d['dimensions']  # (h, w, l)
            yaw = box_3d['orientation']  # 朝向角（绕y轴旋转）
            cls = box_3d['class_name'].lower()
            
            # 转换3D坐标到BEV
            bev_x, bev_y = self._3d_to_bev(loc)
            if bev_x is None:
                print(f"3D框超出范围 ------")  # 新增
                return  # 超出范围不绘制
            
            # 获取颜色
            color = self.class_colors.get(cls, self.class_colors['default'])
            
            # 1. 绘制目标轮廓（基于尺寸和朝向）
            # 目标在BEV中的尺寸（宽=dimensions[1], 长=dimensions[2]）
            w_pix = int(dims[1] * self.scale)  # 宽度（侧向）
            l_pix = int(dims[2] * self.scale)  # 长度（前向）
            
            # 旋转矩阵（绕中心旋转yaw角）
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # 定义目标矩形的四个顶点（中心在原点）
            vertices = [
                (-w_pix/2, -l_pix/2),
                (w_pix/2, -l_pix/2),
                (w_pix/2, l_pix/2),
                (-w_pix/2, l_pix/2)
            ]
            
            # 旋转并平移顶点到BEV坐标
            rotated_vertices = []
            for (vx, vy) in vertices:
                # 旋转
                rx = vx * cos_yaw - vy * sin_yaw
                ry = vx * sin_yaw + vy * cos_yaw
                # 平移到目标位置
                rotated_vertices.append((bev_x + rx, bev_y + ry))
            
            # 绘制旋转后的矩形（填充半透明）
            pts = np.array(rotated_vertices, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(self.bev_image, [pts], (color[0], color[1], color[2], 128), cv2.LINE_AA)
            # 绘制边框（更清晰）
            cv2.polylines(self.bev_image, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            
            # 2. 绘制朝向箭头（指示目标前进方向）
            arrow_len = max(l_pix // 2, 10)
            arrow_end_x = bev_x + arrow_len * sin_yaw  # 朝向角的侧向分量
            arrow_end_y = bev_y - arrow_len * cos_yaw  # 朝向角的前向分量（y减小为前）
            cv2.arrowedLine(self.bev_image, 
                           (bev_x, bev_y), 
                           (int(arrow_end_x), int(arrow_end_y)), 
                           (255, 255, 255), 2, tipLength=0.3)
            
            # 3. 绘制目标ID和类别
            text = cls
            cv2.putText(self.bev_image, text, 
                       (bev_x - 20, bev_y + l_pix//2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"BEV绘制错误: {e}")

    def draw_depth_map(self, depth_map_norm):
        """
        将深度图投影到BEV作为背景（增强场景感知）
        
        Args:
            depth_map_norm: 归一化深度图（0-1）
        """
        print(f"开始绘制深度图到BEV - 深度图尺寸: {depth_map_norm.shape}")  # 新增
        h, w = depth_map_norm.shape
        # 每隔10像素采样（平衡精度和速度）
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                depth_val = depth_map_norm[y, x]
                if depth_val < 0.05:  # 忽略过近的点
                    continue
                # 转换图像坐标到3D相机坐标（简化版）
                # 假设相机内参：fx=700, cx=w/2, cy=h/2
                fx = 700
                cx, cy = w//2, h//2
                z = 1.0 + depth_val * (self.max_distance - 1.0)  # 1m到max_distance
                x_cam = (x - cx) * z / fx
                y_cam = (y - cy) * z / fx
                # 转换到BEV坐标
                bev_x, bev_y = self._3d_to_bev((x_cam, y_cam, z))
                if bev_x is not None:
                    # 用深度值控制亮度（越远越暗）
                    intensity = int(200 * (1 - depth_val))
                    cv2.circle(self.bev_image, (bev_x, bev_y), 1, (intensity, intensity, intensity), -1)


    def get_image(self):
        """返回当前BEV图像"""
        return self.bev_image.copy()