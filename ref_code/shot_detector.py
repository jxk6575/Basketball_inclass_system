from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import score, detect_down, detect_up, clean_hoop_pos, clean_ball_pos, get_device, ThreePointMarker
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Union

# 导入检测器 - 支持相对导入和绝对导入
try:
    # 当作为模块被导入时使用相对导入
    from .ball_detector import BallDetector
    from .basket_detector import BasketDetector
except ImportError:
    # 当作为主程序运行时使用绝对导入
    from ball_detector import BallDetector
    from basket_detector import BasketDetector

class ShotDetector:
    def __init__(self, model_resolution=640, stream=False, half=False, retina_masks=False):
        self.overlay_text = "Waiting..."
        
        # 使用检测器进行篮球和篮筐检测
        self.ball_detector = BallDetector(
            model_path="weights/Basketball_v1.pt",
            class_num=0,  # 篮球
            best_detection=True,
            confidence_threshold=0.3,
            imgsz=model_resolution,
            half=half,
            retina_masks=retina_masks,
            stream=stream
        )
        
        self.basket_detector = BasketDetector(
            model_path="weights/Basketball_v1.pt", 
            class_num=1,  # 篮筐
            best_detection=True,
            confidence_threshold=0.5,
            imgsz=model_resolution,
            half=half,
            retina_masks=retina_masks,
            stream=stream
        )
        
        # 保留原有的参数设置，用于兼容性
        self.stream = stream
        self.model_resolution = model_resolution
        self.device = get_device()
        self.half = half
        self.retina_masks = retina_masks

        # 跟踪数据结构
        self.ball_pos = []  # 元组数组 ((x_pos, y_pos), 帧计数, 宽度, 高度, 置信度)
        self.hoop_pos = []  # 元组数组 ((x_pos, y_pos), 帧计数, 宽度, 高度, 置信度)
        self.frame_count = 0
        self.frame = None
        self.pending_ball_pos = []  # 待加入轨迹的点队列，用于异常点检测（最多保存2个点）
        
        # 篮筐位置固定模式
        self.fixed_hoop_position = None  # 固定的篮筐位置
        self.hoop_calibration_frames = []  # 用于校准的篮筐检测帧
        self.hoop_calibration_complete = False  # 篮筐校准是否完成

        # 得分相关
        self.makes = 0
        self.attempts = 0

        # 投篮检测状态
        self.shot_in_progress = False
        self.shot_ball_positions = []  # 跟踪投篮尝试期间的球位置
        self.shot_frames = []  # 跟踪投篮尝试期间的原始帧数据（用于颜色检测）

        # 视觉效果
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        
        # 三分线标记器
        self.three_point_marker = ThreePointMarker()
        self.is_three_point_mode = False  # 是否处于三分线标记模式
        
        # 添加首次上篮和补篮的跟踪
        self.is_first_shot = True  # 是否是首次上篮
        self.retry_count = 0       # 当前补篮次数
        
    def start_three_point_marking(self, frame: np.ndarray):
        """开始三分线标记模式"""
        self.is_three_point_mode = True
        return self.three_point_marker.start_marking(frame)
    
    def stop_three_point_marking(self):
        """停止三分线标记模式"""
        self.is_three_point_mode = False
    
    def load_three_point_config(self, video_name: str) -> bool:
        """加载三分线配置"""
        return self.three_point_marker.load_config(video_name)
    
    def save_three_point_config(self, video_name: str) -> str:
        """保存三分线配置"""
        return self.three_point_marker.save_config(video_name)
    
    def three_point_mouse_callback(self, event, x, y, flags, param):
        """三分线标记鼠标回调函数"""
        if self.is_three_point_mode:
            self.three_point_marker.mouse_callback(event, x, y, flags, param)
            # 如果标记完成，退出标记模式
            if not self.three_point_marker.is_marking and self.is_three_point_mode:
                self.is_three_point_mode = False

    def process_frame(self, frame: np.ndarray, frame_count: int, pose_detector=None) -> np.ndarray:
        """处理单个图像帧
        
        Args:
            frame: 输入帧
            frame_count: 帧计数
            pose_detector: 姿态检测器实例（可选），用于获取头部中心点
        """
        self.frame = frame.copy()
        self.frame_count = frame_count
        
        # 如果投篮正在进行，在绘制任何内容之前保存原始帧（用于颜色检测）
        if self.shot_in_progress:
            self.shot_frames.append(frame.copy())  # 保存原始帧，不包含任何绘制
        
        # 如果处于三分线标记模式，让标记器处理
        if self.is_three_point_mode:
            return self.three_point_marker.process_frame(self.frame)

        # 使用检测器进行篮球检测
        ball_detections = self.ball_detector.process(self.frame)

        # 处理篮球检测结果（由于设置了best_detection=True，只返回一个最佳结果）
        if ball_detections:
            detection = ball_detections[0]  # 获取最佳检测结果
            center = detection['center']
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # 检查篮球中心点是否与人体头部中心点离的很近
            should_skip = False
            if pose_detector is not None:
                head_center = pose_detector.get_head_center()
                if head_center is not None:
                    # 计算篮球中心点和头部中心点的距离
                    ball_center_x, ball_center_y = center
                    head_center_x, head_center_y = head_center
                    distance = math.sqrt((ball_center_x - head_center_x)**2 + (ball_center_y - head_center_y)**2)
                    
                    # 如果距离小于篮球检测框的对角线长度，则认为离得很近
                    ball_diagonal = math.sqrt(bbox[2]**2 + bbox[3]**2)
                    distance_threshold = ball_diagonal * 0.8  # 1.5倍对角线长度作为阈值
                    
                    if distance < distance_threshold:
                        should_skip = True
            
            # 如果距离很近，跳过这个点，不加入轨迹
            if should_skip:
                # 绘制检测框（仍然绘制，但不加入轨迹）
                x1, y1, w, h = bbox
                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(0, 255, 0))
            else:
                # 保存为待加入的点
                new_point = (center, self.frame_count, bbox[2], bbox[3], conf)
                
                # 将新点加入pending队列
                self.pending_ball_pos.append(new_point)
            
            # 如果pending中有2个点，检查是否需要跳过中间点
            if len(self.pending_ball_pos) >= 2:
                point_b = self.pending_ball_pos[0]  # pending中的第一个点（b）
                point_c = self.pending_ball_pos[1]  # pending中的第二个点（c）
                
                # 检查：如果a（ball_pos最后一个）、c距离相近，但b距离两者都很远，则跳过b
                if len(self.ball_pos) > 0 and self._should_skip_middle_point(point_b, point_c):
                    # 跳过b，用c取代b的位置
                    self.pending_ball_pos.pop(0)  # 删除b
                    # c已经在pending[0]位置了（因为删除了b）
                else:
                    # b没问题，b正式入列
                    self.ball_pos.append(point_b)
                    # c留在pending[0]位置（因为删除了b）
                    self.pending_ball_pos.pop(0)
            
            # 绘制检测框
            x1, y1, w, h = bbox
            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(0, 255, 0))

        # 处理篮筐检测 - 使用固定位置或实时检测
        if self.fixed_hoop_position is not None:
            # 使用固定的篮筐位置
            center = self.fixed_hoop_position['center']
            bbox = self.fixed_hoop_position['bbox']
            conf = self.fixed_hoop_position['confidence']
            
            # 添加到跟踪列表（使用当前帧计数）
            self.hoop_pos.append((center, self.frame_count, bbox[2], bbox[3], conf))
            
            # 绘制检测框（使用固定颜色表示固定位置）
            x1, y1, w, h = bbox
            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 0, 255))  # 紫色表示固定位置
        else:
            # 实时检测篮筐（用于校准阶段）
            basket_detections = self.basket_detector.process(self.frame)
            if basket_detections:
                detection = basket_detections[0]  # 获取最佳检测结果
                center = detection['center']
                bbox = detection['bbox']
                conf = detection['confidence']
                
                # 添加到跟踪列表
                self.hoop_pos.append((center, self.frame_count, bbox[2], bbox[3], conf))
                
                # 绘制检测框
                x1, y1, w, h = bbox
                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 0, 0))

        self.clean_motion()
        
        # 应用三分区域标记
        if not self.is_three_point_mode and len(self.three_point_marker.arc_points) >= 3:
            # 让标记器直接在帧上绘制，但不显示点和线
            return self.three_point_marker.draw_markers(self.frame, show_points_and_lines=False)
            
        return self.frame

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def display_score(self):
        """显示得分信息 - 不再直接在帧上绘制文字"""
        # 所有文本消息都通过detector_manager的compose_display_frame在信息区显示
        pass

        # 淡入淡出效果
        if self.fade_counter > 0:
            self.fade_counter -= 1

    def reset_shot_status(self):
        """重置上篮状态"""
        self.is_first_shot = True
        self.retry_count = 0
        self.shot_in_progress = False
        self.shot_ball_positions = []
        self.shot_frames = []
    
    def start_hoop_calibration(self):
        """开始篮筐位置校准"""
        self.hoop_calibration_frames = []
        self.hoop_calibration_complete = False
        self.fixed_hoop_position = None
        print("Starting hoop position calibration...")
    
    def calibrate_hoop_position(self, frame: np.ndarray) -> bool:
        """校准篮筐位置，返回是否完成校准"""
        if self.hoop_calibration_complete:
            return True
            
        # 检测篮筐
        basket_detections = self.basket_detector.process(frame)
        
        if basket_detections:
            detection = basket_detections[0]  # 获取最佳检测结果
            center = detection['center']
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # 添加到校准帧列表
            self.hoop_calibration_frames.append({
                'center': center,
                'bbox': bbox,
                'confidence': conf
            })
            
            print(f"Hoop calibration frame {len(self.hoop_calibration_frames)}: center={center}, conf={conf:.3f}")
            
            # 如果收集了足够的帧，计算平均位置
            if len(self.hoop_calibration_frames) >= 3:
                # 计算平均中心位置
                avg_center_x = sum(d['center'][0] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)
                avg_center_y = sum(d['center'][1] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)
                
                # 计算平均边界框
                avg_bbox_w = sum(d['bbox'][2] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)
                avg_bbox_h = sum(d['bbox'][3] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)
                
                # 使用第一个检测的边界框位置作为基准
                base_bbox = self.hoop_calibration_frames[0]['bbox']
                avg_bbox_x = base_bbox[0]
                avg_bbox_y = base_bbox[1]
                
                # 设置固定的篮筐位置
                self.fixed_hoop_position = {
                    'center': (int(avg_center_x), int(avg_center_y)),
                    'bbox': (avg_bbox_x, avg_bbox_y, int(avg_bbox_w), int(avg_bbox_h)),
                    'confidence': 1.0  # 固定位置使用最高置信度
                }
                
                self.hoop_calibration_complete = True
                print(f"Hoop calibration complete: fixed position = {self.fixed_hoop_position['center']}")
                return True
        
        return False
    
    def get_fixed_hoop_position(self):
        """获取固定的篮筐位置"""
        return self.fixed_hoop_position

    def check_shot(self) -> Optional[Dict]:
        """检查是否发生投篮并返回结果"""
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return None

        # 检查球是否在篮筐下边界之上（投篮开始）
        if not self.shot_in_progress:
            if detect_up(self.ball_pos, self.hoop_pos):
                # 球的长宽均需小于篮筐的长宽，防止球离摄像头近的误判
                ball_width = self.ball_pos[-1][2]
                ball_height = self.ball_pos[-1][3]
                hoop_width = self.hoop_pos[-1][2]
                hoop_height = self.hoop_pos[-1][3]
                
                if ball_width < hoop_width and ball_height < hoop_height:
                    self.shot_in_progress = True
                    self.shot_ball_positions = []  # 开始跟踪球位置
                    self.shot_frames = []  # 开始跟踪原始帧数据
                    print("Shot attempt started")
        
        # 如果投篮正在进行，跟踪球位置
        if self.shot_in_progress:
            # 将当前球位置添加到投篮跟踪
            if len(self.ball_pos) > 0:
                self.shot_ball_positions.append(self.ball_pos[-1])
            
            # 检查球是否在篮筐下边界之下（投篮结束）
            if detect_down(self.ball_pos, self.hoop_pos):
                self.shot_in_progress = False
                self.attempts += 1
                
                # 确定是否得分（传入原始帧数据用于颜色检测）
                is_score = score(self.shot_ball_positions, self.hoop_pos, self.shot_frames)
                
                # 清除投篮跟踪数据
                self.shot_ball_positions = []
                self.shot_frames = []
                
                # 根据是否是首次上篮返回不同的结果
                if is_score:
                    self.makes += 1
                    result = {
                        "text": "Made Shot!" if self.is_first_shot else "Made Retry Shot!",
                        "color": (0, 255, 0),
                        "event": "first_shot_made" if self.is_first_shot else "retry_shot_made"
                    }
                    # 重置状态
                    self.reset_shot_status()
                    return result
                else:
                    if self.is_first_shot:
                        self.is_first_shot = False
                        self.retry_count = 0
                        return {
                            "text": "Missed First Shot - Retry Available",
                            "color": (255, 165, 0),
                            "event": "first_shot_missed"
                        }
                    else:
                        self.retry_count += 1
                        if self.retry_count >= 2:
                            # 两次补篮都未进，重置状态
                            self.reset_shot_status()
                            return {
                                "text": "Failed After Two Retries - Exit Three-Point Line",
                                "color": (255, 0, 0),
                                "event": "retry_shot_missed"
                            }
                        return {
                            "text": f"Missed Retry ({self.retry_count}/2) - Try Again",
                            "color": (255, 165, 0),
                            "event": "retry_shot_missed"
                        }

        return None

    def _should_skip_middle_point(self, point_b, point_c):
        """
        检查是否应该跳过中间点（异常跳跃点）
        
        逻辑：如果a（ball_pos最后一个）、c（pending第二个）距离相近，但b（pending第一个）距离两者都很远，则跳过b
        
        Args:
            point_b: pending中的第一个点（b），格式：((x, y), frame_count, width, height, confidence)
            point_c: pending中的第二个点（c），格式：((x, y), frame_count, width, height, confidence)
        
        Returns:
            bool: 如果应该跳过b，返回True
        """
        # 至少需要1个点（a）才能进行判断
        if len(self.ball_pos) < 1:
            return False
        
        # 获取a（ball_pos最后一个）、b（pending第一个）、c（pending第二个）
        point_a = self.ball_pos[-1][0]  # a的位置
        point_b_pos = point_b[0]         # b的位置
        point_c_pos = point_c[0]         # c的位置
        
        # 计算距离
        dist_ac = math.sqrt((point_c_pos[0] - point_a[0]) ** 2 + (point_c_pos[1] - point_a[1]) ** 2)
        dist_ab = math.sqrt((point_b_pos[0] - point_a[0]) ** 2 + (point_b_pos[1] - point_a[1]) ** 2)
        dist_bc = math.sqrt((point_c_pos[0] - point_b_pos[0]) ** 2 + (point_c_pos[1] - point_b_pos[1]) ** 2)
        
        # 如果a和c距离相近（小于阈值），但b距离两者都很远（大于阈值），则判定b为异常点
        # 阈值：使用球的平均尺寸作为参考
        if len(self.ball_pos) >= 1:
            avg_ball_size = (self.ball_pos[-1][2] + self.ball_pos[-1][3]) / 2
            threshold_close = avg_ball_size * 1  # 相近距离阈值：2倍球尺寸
            threshold_far = avg_ball_size * 3    # 远距离阈值：5倍球尺寸
        else:
            threshold_close = 30  # 默认阈值
            threshold_far = 100
        
        # 判断：a和c距离相近，且b距离a和c都很远
        if dist_ac < threshold_close and dist_ab > threshold_far and dist_bc > threshold_far:
            return True
        
        return False


def main():
    """
    主函数 - 测试投篮检测系统
    使用 inputs/standard.mp4 进行测试
    """
    # 确保工作目录正确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    
    print("🏀 投篮检测系统测试")
    print("使用视频: inputs/standard.mp4")
    print("按键控制:")
    print("  'q' - 退出")
    print("  'r' - 重置投篮统计")
    
    # 初始化投篮检测器
    detector = ShotDetector(
        model_resolution=640,
        stream=False,
        half=False,
        retina_masks=False
    )
    
    # 打开视频文件
    video_path = "inputs/standard.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    frame_count = 0
    print("开始检测...")
    
    # 创建显示窗口
    cv2.namedWindow('Shot Detection Test')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break
        
        frame_count += 1
        
        # 处理帧
        processed_frame = detector.process_frame(frame, frame_count)
        
        # 检查投篮
        shot_result = detector.check_shot()
        if shot_result:
            print(f"第{frame_count}帧: {shot_result['text']}")
        
        # 在图像上显示统计信息
        info_y = 30
        cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(processed_frame, f"Makes: {detector.makes}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(processed_frame, f"Attempts: {detector.attempts}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        info_y += 30
        success_rate = (detector.makes / detector.attempts * 100) if detector.attempts > 0 else 0
        cv2.putText(processed_frame, f"Success Rate: {success_rate:.1f}%", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 显示投篮状态
        info_y += 30
        shot_status = "First Shot" if detector.is_first_shot else f"Retry ({detector.retry_count}/2)"
        cv2.putText(processed_frame, f"Status: {shot_status}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示处理后的帧
        cv2.imshow('Shot Detection Test', processed_frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("退出检测")
            break
        elif key == ord('r'):
            detector.makes = 0
            detector.attempts = 0
            detector.reset_shot_status()
            print("投篮统计已重置")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 显示最终统计
    print("最终统计:")
    print(f"   进球数: {detector.makes}")
    print(f"   尝试数: {detector.attempts}")
    success_rate = (detector.makes / detector.attempts * 100) if detector.attempts > 0 else 0
    print(f"   成功率: {success_rate:.1f}%")


if __name__ == "__main__":
    main()