import math
import numpy as np
import cv2
import os
from datetime import datetime

DEBUG = False  # 调试模式开关，设为True时保存调试图片

def save_debug_rim_images(rim_region_base, occluded_frames_list, debug_dir, shot_frames=None):
    """
    保存篮筐上边缘区域的调试图片
    
    Args:
        rim_region_base: 基准篮筐上边缘区域（第一帧）
        occluded_frames_list: 所有检测到遮挡的帧列表，格式：[(frame_index, rim_region), ...]
        debug_dir: 调试目录路径（已创建）
        shot_frames: 投篮过程中的完整帧列表（可选，用于保存原始帧）
    
    Returns:
        str: 调试目录路径
    """
    try:
        # 保存基准区域（第一帧的正常篮筐区域）
        if rim_region_base is not None and rim_region_base.size > 0:
            # 放大区域以便更好地查看（放大10倍）
            base_region_resized = cv2.resize(rim_region_base, 
                                            (rim_region_base.shape[1] * 10, rim_region_base.shape[0] * 10),
                                            interpolation=cv2.INTER_NEAREST)
            base_path = os.path.join(debug_dir, "00_base_rim_region.png")
            cv2.imwrite(base_path, base_region_resized)
        
        # 保存所有检测到遮挡的帧
        for frame_index, rim_region_occluded in occluded_frames_list:
            if rim_region_occluded is not None and rim_region_occluded.size > 0:
                # 放大区域以便更好地查看（放大10倍）
                occluded_region_resized = cv2.resize(rim_region_occluded,
                                                    (rim_region_occluded.shape[1] * 10, rim_region_occluded.shape[0] * 10),
                                                    interpolation=cv2.INTER_NEAREST)
                occluded_path = os.path.join(debug_dir, f"{frame_index:03d}_occluded_rim_region.png")
                cv2.imwrite(occluded_path, occluded_region_resized)
                
                # 保存出现遮挡的帧的原始完整图片
                if shot_frames is not None and frame_index < len(shot_frames):
                    original_frame = shot_frames[frame_index]
                    original_frame_path = os.path.join(debug_dir, f"{frame_index:03d}_original_frame.png")
                    cv2.imwrite(original_frame_path, original_frame)
        
        return debug_dir
        
    except Exception as e:
        print(f"保存调试图片时出错: {e}")
        return None


def check_hoop_rim_occlusion(shot_frames, hoop_pos, ball_color, ball_pos=None):
    """
    通过检测像素颜色是否与篮球颜色相近来判断篮球是否遮挡了篮筐上边缘
    
    Args:
        shot_frames: 投篮过程中的帧列表（BGR格式）
        hoop_pos: 篮筐位置历史，格式：((x, y), frame_count, width, height, confidence)
        ball_color: 篮球颜色（BGR格式），格式：(B, G, R)或numpy数组
        ball_pos: 球位置历史，格式：((x, y), frame_count, width, height, confidence)（可选，用于验证遮挡）
    
    Returns:
        bool: 如果检测到遮挡返回True
    """
    
    if len(shot_frames) < 2 or len(hoop_pos) < 1:
        return False
    
    # 获取篮筐位置信息
    hoop_center = hoop_pos[-1][0]
    hoop_center_x, hoop_center_y = int(hoop_center[0]), int(hoop_center[1])
    hoop_width = int(hoop_pos[-1][2])
    hoop_height = int(hoop_pos[-1][3])
    
    # 计算篮筐上边缘的位置
    # hoop_pos存储格式：((center_x, center_y), frame_count, width, height, confidence)
    # 需要从center计算bbox的左上角
    hoop_x1_full = hoop_center_x - hoop_width // 2  # 篮筐完整左边缘X坐标
    hoop_y1 = hoop_center_y - hoop_height // 2  # 篮筐上边缘Y坐标
    rim_line_height = 5  # 上边缘线的高度（像素）
    
    # 裁剪操作：砍掉两侧各15%的宽度，裁去最上面和最下面各1行像素
    # 1. 先砍掉两侧各15%的宽度
    trim_ratio = 0.15
    rim_x1 = hoop_x1_full + int(hoop_width * trim_ratio)  # 检测区域的左边缘（砍掉左侧15%）
    rim_width = int(hoop_width * (1 - 2 * trim_ratio))  # 检测区域的宽度（中间70%）
    
    # 2. 再裁去最上面和最下面各1行像素，剩下中间3行
    rim_y1_trimmed = hoop_y1 + 1  # 裁去最上面1行
    rim_height_trimmed = rim_line_height - 2  # 裁去上下各1行，剩下中间3行
    
    # 确保坐标在图像范围内
    if rim_x1 < 0 or rim_y1_trimmed < 0:
        return False
    
    # 获取第一帧（投篮开始前）的篮筐上边缘颜色作为基准
    if len(shot_frames) == 0:
        return False
    
    first_frame = shot_frames[0]
    frame_h, frame_w = first_frame.shape[:2]
    
    # 确保坐标在图像范围内
    if rim_x1 + rim_width > frame_w or rim_y1_trimmed + rim_height_trimmed > frame_h:
        return False
    
    # 提取第一帧的篮筐上边缘区域（已应用两个裁剪：砍掉两侧和上下行）作为基准
    rim_region_base = first_frame[rim_y1_trimmed:rim_y1_trimmed + rim_height_trimmed, 
                                   rim_x1:rim_x1 + rim_width]
    if rim_region_base.size == 0:
        return False
    
    # 创建调试目录（仅在DEBUG模式下）
    debug_dir = None
    if DEBUG:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 精确到毫秒
        debug_dir = f"debug/shot/{timestamp}"
        os.makedirs(debug_dir, exist_ok=True)
    
    # 检查后续帧中篮筐上边缘的颜色变化
    occluded_frames_list = []  # 记录所有检测到遮挡的帧
    
    # 篮球颜色相似度阈值：像素颜色与篮球颜色的差异小于此阈值认为相近
    ball_color_similarity_threshold = 40  # BGR颜色空间，0-255范围
    # 遮挡判定阈值：与篮球颜色相近的像素点百分比超过此值，判定为遮挡
    occlusion_pixel_ratio_threshold = 0.1  # 10%的像素点与篮球颜色相近
    
    # 将篮球颜色转换为numpy数组格式
    if isinstance(ball_color, (list, tuple)):
        ball_color_array = np.array(ball_color, dtype=np.float32).reshape(1, 1, 3)
    else:
        ball_color_array = np.array(ball_color, dtype=np.float32).reshape(1, 1, 3)
    
    for i in range(1, len(shot_frames)):
        frame = shot_frames[i]
        frame_h, frame_w = frame.shape[:2]
        
        # 确保坐标在图像范围内
        if rim_x1 + rim_width > frame_w or rim_y1_trimmed + rim_height_trimmed > frame_h:
            continue
        
        # 提取当前帧的篮筐上边缘区域（已应用两个裁剪：砍掉两侧和上下行）
        rim_region_current = frame[rim_y1_trimmed:rim_y1_trimmed + rim_height_trimmed,
                                    rim_x1:rim_x1 + rim_width]
        
        if rim_region_current.size == 0:
            continue
        
        # 计算篮筐上边缘区域每个像素与篮球颜色的差异（欧氏距离）
        color_diff = np.sqrt(np.sum((rim_region_current.astype(np.float32) - ball_color_array) ** 2, axis=2))
        
        # 统计与篮球颜色相近（差异小于阈值）的像素点数量
        pixels_similar_to_ball = np.sum(color_diff < ball_color_similarity_threshold)
        total_pixels = color_diff.size
        similarity_ratio = pixels_similar_to_ball / total_pixels if total_pixels > 0 else 0
        
        # 如果与篮球颜色相近的像素点百分比超过阈值，则判定为遮挡
        if similarity_ratio > occlusion_pixel_ratio_threshold:
            # 如果这一帧能够检测到球，检查篮筐上界是否在球检测框上下界之间
            should_ignore_all = False
            if ball_pos is not None and i < len(ball_pos):
                ball_data = ball_pos[i]
                if ball_data is not None:
                    ball_center = ball_data[0]
                    ball_height = ball_data[3]
                    
                    # 篮筐上边缘Y坐标（使用rim_y1_trimmed作为裁剪后的上边缘）
                    hoop_rim_top = rim_y1_trimmed
                    
                    # 如果篮球下界在篮筐上界之上，则忽略之前所有的遮挡
                    if (ball_center[1] + ball_height < hoop_rim_top):
                        should_ignore_all = True
                        # 清空之前记录的所有遮挡帧
                        occluded_frames_list.clear()
            
            # 如果不忽略，记录所有检测到遮挡的帧
            if not should_ignore_all:
                occluded_frames_list.append((i, rim_region_current.copy()))
    
    # 如果检测到遮挡，判定为被遮挡（只要有一帧检测到遮挡）
    is_occluded = len(occluded_frames_list) > 0
    
    # 仅在DEBUG模式下保存调试图片：基准图片 + 所有遮挡帧图片 + 原始完整帧
    if DEBUG:
        save_debug_rim_images(rim_region_base, occluded_frames_list, debug_dir, shot_frames)
    
    return is_occluded


def score(ball_pos, hoop_pos, shot_frames=None):
    """
    检测球是否进入篮筐
    
    Args:
        ball_pos: 球位置历史，格式：((x, y), frame_count, width, height, confidence)
        hoop_pos: 篮筐位置历史，格式：((x, y), frame_count, width, height, confidence)
        shot_frames: 投篮过程中的帧列表（可选，用于颜色遮挡检测）
    
    Returns:
        bool: 如果球进入篮筐返回True
    """
    
    if len(ball_pos) < 2 or len(hoop_pos) < 1:
        return False
    
    # 颜色遮挡检测：如果球向下运动且遮挡了篮筐上边缘，则肯定没进
    if shot_frames is not None and len(shot_frames) > 0:
        # 检查球是否向下运动
        if len(ball_pos) >= 2:
            # 检查最后几帧的球位置
            ball_moving_down = False
            for i in range(max(0, len(ball_pos) - 3), len(ball_pos)):
                if i > 0:
                    prev_y = ball_pos[i-1][0][1]
                    curr_y = ball_pos[i][0][1]
                    if curr_y > prev_y:  # 向下运动
                        ball_moving_down = True
                        break
            
            # 提取篮球颜色（从投篮过程中的多个帧中提取，取平均值）
            ball_color = None
            if len(ball_pos) > 0 and len(shot_frames) > 0:
                ball_colors = []
                # 从投篮过程中的前几帧提取篮球颜色（避免球在篮筐附近时颜色可能被遮挡）
                for i in range(min(3, len(ball_pos), len(shot_frames))):
                    ball_data = ball_pos[i]
                    frame = shot_frames[i]
                    if ball_data is not None and frame is not None:
                        ball_center = ball_data[0]
                        ball_width = ball_data[2]
                        ball_height = ball_data[3]
                        
                        # 计算篮球区域的边界
                        ball_x1 = int(ball_center[0] - ball_width // 2)
                        ball_y1 = int(ball_center[1] - ball_height // 2)
                        ball_x2 = int(ball_center[0] + ball_width // 2)
                        ball_y2 = int(ball_center[1] + ball_height // 2)
                        
                        frame_h, frame_w = frame.shape[:2]
                        # 确保坐标在图像范围内
                        if ball_x1 >= 0 and ball_y1 >= 0 and ball_x2 < frame_w and ball_y2 < frame_h:
                            # 提取篮球区域
                            ball_region = frame[ball_y1:ball_y2, ball_x1:ball_x2]
                            if ball_region.size > 0:
                                # 裁剪：只保留中间部分（长宽各保留一半）
                                h, w = ball_region.shape[:2]
                                center_h, center_w = h // 2, w // 2
                                quarter_h, quarter_w = h // 4, w // 4
                                
                                # 提取中间区域（从1/4到3/4的位置）
                                ball_region_center = ball_region[quarter_h:center_h + quarter_h, 
                                                                 quarter_w:center_w + quarter_w]
                                
                                if ball_region_center.size > 0:
                                    # 计算中间区域的平均颜色
                                    avg_color = np.mean(ball_region_center.reshape(-1, 3), axis=0)
                                    ball_colors.append(avg_color)
                
                # 如果有提取到篮球颜色，计算平均值
                if len(ball_colors) > 0:
                    ball_color = np.mean(ball_colors, axis=0)
            
            # 如果球向下运动且遮挡了篮筐上边缘，判定为未进球
            if ball_moving_down and ball_color is not None:
                if check_hoop_rim_occlusion(shot_frames, hoop_pos, ball_color, ball_pos):
                    return False
    
    # 轨迹检测逻辑
    hoop_center = hoop_pos[-1][0]
    hoop_center_y = hoop_center[1]
    
    # 找到球在篮筐中心上方的最后一个位置（点a）
    point_a = None
    point_a_index = -1
    
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < hoop_center_y:
            point_a = ball_pos[i][0]
            point_a_index = i
            break
    
    # 如果没有找到点a，球没有经过篮筐中心上方
    if point_a is None:
        return False
    
    # 找到点a之后的球的下一个位置（点b）
    point_b = None
    if point_a_index + 1 < len(ball_pos):
        point_b = ball_pos[point_a_index + 1][0]
    
    # 如果没有找到点b，无法确定是否得分
    if point_b is None:
        return False
    
    # 计算线段ab到篮筐中心的距离
    distance = point_to_line_distance(point_a, point_b, hoop_center)
    
    # 如果距离小于20像素，球得分
    return distance < 20


def point_to_line_distance(point_a, point_b, target_point):
    """计算点到线段的距离"""
    line_vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    target_vector = (target_point[0] - point_a[0], target_point[1] - point_a[1])
    line_length_sq = line_vector[0]**2 + line_vector[1]**2
    if line_length_sq < 1e-6:
        return math.sqrt(target_vector[0]**2 + target_vector[1]**2)
    
    t = max(0, min(1, (target_vector[0] * line_vector[0] + target_vector[1] * line_vector[1]) / line_length_sq))
    closest_point = (
        point_a[0] + t * line_vector[0],
        point_a[1] + t * line_vector[1]
    )
    distance = math.sqrt((target_point[0] - closest_point[0])**2 + (target_point[1] - closest_point[1])**2)
    
    return distance


def detect_down(ball_pos, hoop_pos):
    """检测球是否在篮筐下边界下方 - 用于投篮尝试检测"""
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return False
    
    # 篮筐下边界
    hoop_lower_boundary = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    
    # 检查球心是否在篮筐下边界下方
    ball_center_y = ball_pos[-1][0][1]
    return ball_center_y > hoop_lower_boundary


def detect_up(ball_pos, hoop_pos):
    """检测球是否在篮筐下边界上方 - 用于投篮尝试检测"""
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return True
    
    # 篮筐下边界
    hoop_lower_boundary = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    
    # 检查球心是否在篮筐下边界上方
    ball_center_y = ball_pos[-1][0][1]
    return ball_center_y < hoop_lower_boundary


def in_hoop_region(center, hoop_pos):
    """检查中心点是否在篮筐附近"""
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def clean_ball_pos(ball_pos, frame_count):
    """清理不准确的球位置数据点"""
    # 移除不准确的球尺寸以防止跳转到错误的球
    if len(ball_pos) > 1:
        # 宽度和高度
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        # X和Y坐标
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # 帧计数
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # 球不应该在5帧内移动超过其直径的4倍
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        # 球应该是相对方形的
        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    # 移除超过30帧的旧点
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """清理不准确的篮筐位置数据点"""
    # 防止从一个篮筐跳转到另一个篮筐
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # 篮筐不应该在5帧内移动超过其直径的0.5倍
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # 篮筐应该是相对方形的
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    # 移除旧点
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos 