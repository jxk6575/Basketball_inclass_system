from ultralytics import YOLO
import cv2
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.device_utils import get_device

class BallDetector:
    """
    模块化的篮球检测器类
    专注于基础的篮球检测功能，可检测篮球
    """
    
    def __init__(self, 
                 model_path: str = "weights/Basketball_v1.pt",
                 class_num: int = 0,
                 best_detection: bool = True,
                 confidence_threshold: float = 0.3,
                 imgsz: int = 800,
                 half: bool = False,
                 retina_masks: bool = False,
                 stream: bool = False,):
        """
        初始化篮球检测器
        
        Args:
            model_path: YOLO模型文件路径 (.pt文件)
            class_num: 篮球的类别ID
            best_detection: 是否只返回最佳检测结果，True=只返回最高置信度结果，False=返回所有结果，默认 True
            confidence_threshold: 检测置信度阈值，默认 0.3
            imgsz: 模型输入图像尺寸，默认 800
            half: 是否使用半精度推理，默认 True
            retina_masks: 是否使用视网膜掩码，默认 False
            stream: 是否使用流式推理，默认 True
        """
        self.model_path = model_path
        self.class_num = class_num
        self.best_detection = best_detection
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.half = half
        self.retina_masks = retina_masks
        self.stream = stream
        self.device = get_device()
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def process(self, frame: np.ndarray) -> List[Dict]:
        """
        处理单帧图像，检测篮球
        
        Args:
            frame: 输入图像帧 (BGR格式)
            
        Returns:
            检测结果列表，每个元素包含bbox、confidence等信息
        """
        if frame is None or frame.size == 0:
            return []
        
        # 进行推理
        results = self.model(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            retina_masks=self.retina_masks,
            stream=self.stream,
            verbose=False
        )
        
        # 解析检测结果
        detections = self._parse_detections(results)
        
        return detections
    
    def _parse_detections(self, results) -> List[Dict]:
        """解析YOLO检测结果，只检测指定类别"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # 提取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                center_x, center_y = int(x1 + w/2), int(y1 + h/2)
                
                # 提取置信度和类别
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # 只检测指定的类别
                if class_id != self.class_num:
                    continue
                
                # 过滤低置信度检测
                if confidence < self.confidence_threshold:
                    continue
                
                detection = {
                    'class_id': class_id,
                    'class_name': "basketball",
                    'confidence': confidence,
                    'bbox': (x1, y1, w, h),
                    'center': (center_x, center_y),
                    'area': w * h
                }
                
                detections.append(detection)
        
        # 如果只需要最佳检测结果，返回置信度最高的那个
        if self.best_detection and detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            return [best_detection]
        
        return detections

def main():
    print("篮球检测器测试")
    detector = BallDetector()
    cap = cv2.VideoCapture("inputs/test.mp4")
    
    print("开始检测，按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        detections = detector.process(frame)
        
        # 绘制检测结果
        for detection in detections:
            x1, y1, w, h = detection['bbox']
            x2, y2 = x1 + w, y1 + h
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绿色
            # 绘制置信度
            label = f"{confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow('Basketball Detection', frame)
        
        # 控制
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    