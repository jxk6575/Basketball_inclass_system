from ultralytics import YOLO
import cv2
import numpy as np
from utils import get_device
import math
from typing import Dict, List, Tuple, Optional

class PoseDetector:
    def __init__(self, model_resolution=640, stream=True, half=True, retina_masks=True, best_detection=True):
        # 加载指定分辨率的模型
        self.model = YOLO("weights/Pose.pt")
        self.stream = stream
        self.model_resolution = model_resolution
        self.device = get_device()
        self.half = half
        self.retina_masks = retina_masks
        self.best_detection = best_detection  # 是否只检测置信度最高的人
        
        # 关键点颜色
        self.keypoint_colors = {
            'pose': (0, 255, 0),    # 绿色
            'face': (0, 0, 255),    # 红色
            'left_arm': (255, 0, 0), # 蓝色
            'right_arm': (255, 165, 0), # 橙色
            'left_leg': (255, 255, 0), # 黄色
            'right_leg': (255, 0, 255),  # 紫色
        }

        # 关键点连接
        self.skeleton = [
            [5, 6], [5, 11], [6, 12], [11, 12], # 躯干
            [5, 7], [7, 9], # 左臂
            [6, 8], [8, 10], # 右臂
            [11, 13], [13, 15], # 左腿
            [12, 14], [14, 16], # 右腿
            [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [0, 6] # 面部
        ]

        # 当前帧关键点
        self.keypoints = []
        
        # 跟踪数据
        self.frame_count = 0
        self.pose_history = []  # 保存最近帧的姿态关键点
        
        # 脚底延伸点
        self.shoe_bottom_points = []  # 存储脚底延伸点位置

    def process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """处理单个帧图像，检测人体姿态"""
        try:
            if frame is None:
                print("Warning: Received None frame in PoseDetector")
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
            self.frame = frame.copy()
            self.frame_count = frame_count

            # 重置脚底延伸点列表
            self.shoe_bottom_points = []

            try:
                # 使用模型检测人体姿态，指定分辨率
                results = self.model(self.frame, stream=self.stream, device=self.device, 
                                   imgsz=self.model_resolution, half=self.half, retina_masks=self.retina_masks)

                for r in results:
                    if r.keypoints is not None:
                        # 清除当前帧关键点
                        self.keypoints = []
                        
                        try:
                            # 获取关键点数据和边界框数据
                            keypoints = r.keypoints.data
                            boxes = r.boxes
                            
                            # 如果启用最佳检测，先选择置信度最高的人
                            person_indices_to_process = []
                            if self.best_detection and boxes is not None and len(boxes) > 0 and len(keypoints) > 0:
                                # 获取所有人的置信度（确保边界框和关键点数量一致）
                                person_confidences = []
                                min_count = min(len(boxes), len(keypoints))
                                for box_idx in range(min_count):
                                    try:
                                        confidence = float(boxes[box_idx].conf[0])
                                        person_confidences.append((box_idx, confidence))
                                    except Exception:
                                        continue
                                
                                # 选择置信度最高的人
                                if person_confidences:
                                    best_person = max(person_confidences, key=lambda x: x[1])
                                    person_indices_to_process = [best_person[0]]
                                else:
                                    # 如果没有有效的置信度，处理第一个检测到的人
                                    person_indices_to_process = [0] if len(keypoints) > 0 else []
                            else:
                                # 处理所有检测到的人
                                person_indices_to_process = list(range(len(keypoints)))
                            
                            # 处理选定的人
                            for person_idx in person_indices_to_process:
                                if person_idx >= len(keypoints):
                                    continue
                                    
                                try:
                                    kpts = keypoints[person_idx]
                                    
                                    # 提取关键点坐标和置信度
                                    person_keypoints = []
                                    for kpt in kpts:
                                        try:
                                            x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
                                            if conf > 0.5:  # 只处理高置信度的关键点
                                                person_keypoints.append((int(x), int(y), conf))
                                            else:
                                                person_keypoints.append(None)  # 对低置信度关键点使用None
                                        except Exception as kpt_error:
                                            print(f"Error processing keypoint: {kpt_error}")
                                            person_keypoints.append(None)
                                    
                                    self.keypoints.append(person_keypoints)
                                    
                                    # 绘制骨架
                                    try:
                                        self.draw_skeleton(person_keypoints)
                                    except Exception as draw_error:
                                        print(f"Error drawing skeleton: {draw_error}")
                                        
                                except Exception as person_error:
                                    print(f"Error processing person {person_idx}: {person_error}")
                                    continue
                                    
                        except Exception as keypoints_error:
                            print(f"Error processing keypoints: {keypoints_error}")
                
                # 更新姿态历史
                if len(self.keypoints) > 0:
                    self.pose_history.append((self.keypoints, self.frame_count))
                    # 只保留最近30帧
                    if len(self.pose_history) > 30:
                        self.pose_history.pop(0)
                        
            except Exception as model_error:
                print(f"Error in pose detection model: {model_error}")
                return self.frame
            
            return self.frame
            
        except Exception as e:
            print(f"Critical error in PoseDetector process_frame: {e}")
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    
    def draw_point(self, point: Tuple[int, int], color, size=5):
        """在帧上绘制点的通用方法"""
        x, y = point
        cv2.circle(self.frame, (x, y), size, color, -1)
        return (x, y)
    
    def draw_line(self, point1: Tuple[int, int], point2: Tuple[int, int], color, thickness=2):
        """在帧上绘制线的通用方法"""
        cv2.line(self.frame, point1, point2, color, thickness)

    def draw_skeleton(self, keypoints: List):
        """绘制人体骨架"""
        if keypoints is None or len(keypoints) == 0:
            return
        
        shoe_height = 36  # 脚底延伸的像素数
        person_shoe_bottom_points = []  # 当前人的脚底延伸点
        
        # 绘制所有关键点并计算延伸点
        for i, kpt in enumerate(keypoints):
            if kpt is not None:
                x, y, conf = kpt
                x, y = int(x), int(y)  # 转换为整数
                # 根据身体部位确定关键点颜色
                if i <= 4:  # 面部关键点
                    color = self.keypoint_colors['face']
                elif i in [5, 6, 11, 12]:  # 躯干关键点
                    color = self.keypoint_colors['pose']
                elif i in [7, 9]:  # 左臂关键点
                    color = self.keypoint_colors['left_arm']
                elif i in [8, 10]:  # 右臂关键点
                    color = self.keypoint_colors['right_arm']
                elif i in [13, 15]:  # 左腿关键点
                    color = self.keypoint_colors['left_leg']
                elif i in [14, 16]:  # 右腿关键点
                    color = self.keypoint_colors['right_leg']
                else:
                    color = (0, 255, 0)  # 默认绿色
                
                self.draw_point((x, y), color)
                
                # 处理脚部关键点（15和16）以创建延伸点
                if i == 15 or i == 16:
                    self.process_foot_extensions(i, keypoints, (x, y), shoe_height, person_shoe_bottom_points)
        
        # 如果检测到脚底延伸点，将其添加到全局列表
        if person_shoe_bottom_points:
            self.shoe_bottom_points.append(person_shoe_bottom_points)
        
        # 绘制骨架线
        self.draw_skeleton_lines(keypoints)
    
    def process_foot_extensions(self, foot_idx: int, keypoints: List, foot_point: Tuple[int, int], 
                               shoe_height: int, person_shoe_bottom_points: List):
        """处理和绘制脚部延伸点"""
        x, y = foot_point
        foot_color = self.keypoint_colors['left_leg'] if foot_idx == 15 else self.keypoint_colors['right_leg']
        
        # 1. 垂直延伸点
        shoe_bottom_x = x
        shoe_bottom_y = y + shoe_height
        
        # 绘制点和连接线
        self.draw_point((shoe_bottom_x, shoe_bottom_y), foot_color)
        self.draw_line((x, y), (shoe_bottom_x, shoe_bottom_y), foot_color)
        
        # 保存脚底延伸点位置
        foot_index = 0 if foot_idx == 15 else 1  # 0表示左脚，1表示右脚
        person_shoe_bottom_points.append((foot_index, (shoe_bottom_x, shoe_bottom_y)))
        
        # 2. 膝盖到脚的延伸点
        knee_idx = 13 if foot_idx == 15 else 14  # 13表示左膝，14表示右膝
        
        if keypoints[knee_idx] is not None:
            knee_x, knee_y, _ = keypoints[knee_idx]
            knee_x, knee_y = int(knee_x), int(knee_y)
            # 计算沿膝盖到脚方向的延伸点
            extended_point = self.calculate_extended_point((knee_x, knee_y), (x, y), shoe_height)
            
            # 绘制点和连接线
            self.draw_point(extended_point, foot_color)
            self.draw_line((x, y), extended_point, foot_color)
            
            # 保存延伸点
            foot_index = 2 if foot_idx == 15 else 3  # 2表示左膝延伸，3表示右膝延伸
            person_shoe_bottom_points.append((foot_index, extended_point))
    
    def draw_skeleton_lines(self, keypoints: List):
        """绘制连接关键点的线以形成骨架"""
        for pair in self.skeleton:
            idx1, idx2 = pair
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                if keypoints[idx1] is not None and keypoints[idx2] is not None:
                    x1, y1, _ = keypoints[idx1]
                    x2, y2, _ = keypoints[idx2]
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    
                    # 根据骨架部位确定线条颜色
                    if idx1 <= 4 and idx2 <= 6:  # 面部连接
                        color = self.keypoint_colors['face']
                    elif (idx1 in [5, 6, 11, 12] and idx2 in [5, 6, 11, 12]):  # 躯干连接
                        color = self.keypoint_colors['pose']
                    elif (idx1 in [5, 7] and idx2 in [7, 9]) or (idx2 in [5, 7] and idx1 in [7, 9]):  # 左臂
                        color = self.keypoint_colors['left_arm']
                    elif (idx1 in [6, 8] and idx2 in [8, 10]) or (idx2 in [6, 8] and idx1 in [8, 10]):  # 右臂
                        color = self.keypoint_colors['right_arm']
                    elif (idx1 in [11, 13] and idx2 in [13, 15]) or (idx2 in [11, 13] and idx1 in [13, 15]):  # 左腿
                        color = self.keypoint_colors['left_leg']
                    elif (idx1 in [12, 14] and idx2 in [14, 16]) or (idx2 in [12, 14] and idx1 in [14, 16]):  # 右腿
                        color = self.keypoint_colors['right_leg']
                    else:
                        color = (0, 255, 0)  # 默认绿色
                    
                    self.draw_line(pt1, pt2, color)
    
    def calculate_extended_point(self, start_point: Tuple[int, int], end_point: Tuple[int, int], distance: int) -> Tuple[int, int]:
        """计算从end_point沿start_point到end_point方向延伸的点
        
        Args:
            start_point: 起始点 (x, y)
            end_point: 要延伸的终点 (x, y)
            distance: 延伸距离（像素）
            
        Returns:
            Tuple[int, int]: 延伸点的坐标 (x, y)
        """
        # 计算方向向量
        dir_x = end_point[0] - start_point[0]
        dir_y = end_point[1] - start_point[1]
        
        # 归一化和缩放
        length = math.sqrt(dir_x**2 + dir_y**2)
        if length > 0:  # 避免除零
            dir_x = (dir_x / length) * distance
            dir_y = (dir_y / length) * distance
            
            # 计算延伸点
            extended_x = int(end_point[0] + dir_x)
            extended_y = int(end_point[1] + dir_y)
            return (extended_x, extended_y)
        
        # 如果会发生除零，则返回原终点
        return end_point
    
    def get_head_center(self) -> Optional[Tuple[int, int]]:
        """
        获取头部中心点
        
        Returns:
            Optional[Tuple[int, int]]: 头部中心点坐标 (x, y)，如果无法获取则返回None
        """
        if len(self.keypoints) == 0:
            return None
        
        # 只处理第一个（最佳检测）人的关键点
        person_keypoints = self.keypoints[0]
        if len(person_keypoints) < 5:  # 至少需要5个面部关键点（索引0-4）
            return None
        
        # 提取面部关键点（索引0-4）
        face_keypoints = []
        for i in range(5):  # 关键点0-4是面部关键点
            if person_keypoints[i] is not None:
                x, y, _ = person_keypoints[i]
                face_keypoints.append((x, y))
        
        if len(face_keypoints) == 0:
            return None
        
        # 计算面部关键点的中心点
        avg_x = sum(pt[0] for pt in face_keypoints) // len(face_keypoints)
        avg_y = sum(pt[1] for pt in face_keypoints) // len(face_keypoints)
        
        return (avg_x, avg_y)
    
    def get_foot_points(self) -> List[Tuple]:
        """
        获取脚部点和脚底延伸点
        
        Returns:
            List[Tuple]: 脚部点和脚底延伸点（如果全部检测到共6个点）：
                        - 2个脚部关键点（索引15, 16）
                        - 2个垂直延伸点（脚下方15像素）
                        - 2个膝盖方向延伸点（沿膝盖到脚方向15像素）
        """
        foot_points = []
        
        # 遍历所有检测到的人
        for person_idx, person_keypoints in enumerate(self.keypoints):
            if len(person_keypoints) >= 17:  # 确保完整的关键点
                left_foot = person_keypoints[15]  # 左脚
                right_foot = person_keypoints[16]  # 右脚
                
                if left_foot is not None or right_foot is not None:
                    person_foot_points = []
                    
                    # 添加左脚点（如果存在）
                    if left_foot is not None:
                        person_foot_points.append(left_foot[:2])  # 只取坐标，不取置信度
                    
                    # 添加右脚点（如果存在）
                    if right_foot is not None:
                        person_foot_points.append(right_foot[:2])
                    
                    # 添加脚底延伸点（垂直和膝盖方向）
                    if person_idx < len(self.shoe_bottom_points):
                        for foot_index, bottom_point in self.shoe_bottom_points[person_idx]:
                            # 所有延伸点（索引0, 1为垂直，2, 3为膝盖方向）
                            person_foot_points.append(bottom_point)
                    
                    # 只有至少检测到一个脚部点才添加到结果中
                    if person_foot_points:
                        foot_points.append(person_foot_points)
        
        return foot_points
