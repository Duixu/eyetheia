# -*- coding: utf-8 -*-
"""
视线跟踪器 - 专注于注意力分析、状态管理等分析功能
在多线程架构中，视线估计由 GazeEstimator 处理，这里只做分析
"""
import time
import numpy as np
import cv2
from scipy.spatial import distance as dist
from loguru import logger
from typing import Optional, Dict, Any, List, Tuple
from .probabilistic_model import rnn_method, thres_method

class GazeTracker:
    """
    视线跟踪器 - 专注于分析功能
    
    在多线程架构中，这个类不再负责：
    - 人脸检测（由 FaceDetectionThread 处理）
    - 关键点检测（由 FaceDetectionThread 处理）
    - 视线估计（由 GazeEstimator 处理）
    
    只负责：
    - 注意力分析
    - 状态管理
    - 数据统计
    """
    
    def __init__(self, 
                 fix_threshold=0.5,
                 saccade_threshold=10.0,
                 fluctuation_threshold=10.0,  # 波动容差阈值（度/秒）
                 fluctuation_duration=0.5,   # 波动持续时间阈值（秒）
                 ):
        """
        初始化视线跟踪器
        
        Args:
            fix_threshold: 注视阈值（秒）
            saccade_threshold: 扫视阈值（秒）
            fluctuation_threshold: 波动容差阈值（度/秒）
            fluctuation_duration: 波动持续时间阈值（秒）
        """
        # 注意力分析组件
        self.attention_analyzer = thres_method(buffer_size=10)
        
        # 配置参数
        self.fix_threshold = fix_threshold
        self.saccade_threshold = saccade_threshold
        self.fluctuation_threshold = fluctuation_threshold
        self.fluctuation_duration = fluctuation_duration
        
        # 状态变量
        self.fixation_start = 0
        self.saccade_start = 0
        self.attention_states = []
        self.focusing = 0
        self.last_gaze = None  # 存储上一次的注意力向量
        self.last_position = None  # 存储上一次的位置
        self.last_face_center = None
        
        logger.info(f"视线分析器初始化完成 - 波动阈值: {fluctuation_threshold}°/s, 波动持续时间: {fluctuation_duration}s")
    
    def process_gaze_data(self, 
                          pitch: float, 
                          yaw: float, 
                          gaze_vector: np.ndarray,
                          face_center: np.ndarray) -> Optional[Tuple]:
        """
        处理视线估计结果，进行注意力分析
        
        Args:
            pitch: 视线俯仰角（弧度）
            yaw: 视线偏航角（弧度）
            gaze_vector: 视线向量
            face_center: 人脸中心点
            confidence: 视线估计置信度
            landmarks: 人脸关键点
            
        Returns:
            分析结果元组，如果失败返回None
            # 返回具体的变量值（pitch, yaw, focusing） type=tuple
        """
        # 1. 注意力分析 (视线稳定性优化，以及状态计算)
        attention_analysis_result = self._analyze_attention(pitch, yaw, gaze_vector, face_center)
        # 2. 更新状态
        state = attention_analysis_result.get('flag', -1)
        current_time = time.time()
        # 记录状态变化
        if state == 1:  # 未注视（扫视）
            self.focusing = 0
            if self.saccade_start == 0:
                self.saccade_start = current_time
            if self.fixation_start != 0:
                # 记录注视持续时间
                fixation_duration = current_time - self.fixation_start
                self.attention_states.append(['focusing', fixation_duration])
                self.fixation_start = 0
            
            # 检查是否达到扫视阈值
            if self.saccade_start != 0 and current_time - self.saccade_start > self.saccade_threshold:
                saccade_duration = current_time - self.saccade_start
                self.attention_states.append(['not_focusing', saccade_duration]) 
                
        elif state == 0:  # 注视
            if self.fixation_start == 0:
                self.fixation_start = current_time
            if self.saccade_start != 0:
                # 记录扫视持续时间
                saccade_duration = current_time - self.saccade_start
                self.attention_states.append(['not_focusing', saccade_duration])
                self.saccade_start = 0

            # 检查是否达到注视阈值
            if self.fixation_start != 0 and current_time - self.fixation_start > self.fix_threshold:
                fixation_duration = current_time - self.fixation_start
                self.attention_states.append(['focusing', fixation_duration])
                self.focusing = 1

        # 保持最近的状态记录
        if len(self.attention_states) > 100:
            self.attention_states = self.attention_states[-100:]
        
        # 返回具体的变量值（pitch, yaw, focusing, attention_vector） type=tuple
        return (
            attention_analysis_result['position'][0],             # attention_pitch(平均pitch)
            attention_analysis_result['position'][1],             # attention_yaw(平均yaw)
            self.focusing,
            attention_analysis_result['attention_vector'],      # attention_vector
        )

            
    def _analyze_attention(self, pitch: float, yaw: float, gaze_vector: np.ndarray, face_center: np.ndarray) -> Dict[str, Any]:
        """
        分析注意力状态
        
        Args:
            pitch: 视线俯仰角（弧度）
            yaw: 视线偏航角（弧度）
            gaze_vector: 视线向量
            
        Returns:
            dict: 注意力分析结果
        """
        # 计算视线位置（弧度转度）
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        position = [pitch_deg, yaw_deg]
        # 计算视线速度 确保face_center是numpy数组
        if isinstance(face_center, (list, tuple)):
            face_center = np.array(face_center)
        face_data = [face_center[0], face_center[1], 0] if face_center.size >= 2 else [0, 0, 0]
        self.attention_analyzer.store(position, face_data)
        # 使用注意力分析器进行分析
        attention_result = self.attention_analyzer.analysis()
        # 解包结果：attention_result 返回 (state, position, velocity)
        if isinstance(attention_result, tuple) and len(attention_result) >= 3 and attention_result[0] != -1:
            state, avg_position, avg_velocity = attention_result
        else:
            # 如果分析失败，使用默认值
            state = -1
            avg_position = position
            avg_velocity = [-1, -1]
        # 应用波动容差处理
        state = self._apply_fluctuation_tolerance(state, avg_velocity, position)
        # 计算注意力向量
        attention_vec = self._calculate_attention_vector(state, avg_position, gaze_vector)
        # 平滑处理
        if self.last_gaze is not None:
            attention_vec = 0.85 * attention_vec + 0.15 * self.last_gaze
        self.last_gaze = attention_vec
        return {

            'flag': state,
            'position': avg_position,
            'velocity': avg_velocity,
            'attention_vector': attention_vec
        }
    
    def _apply_fluctuation_tolerance(self, state: int, velocity: List[float], angles: List[float]) -> int:
        """
        应用波动容差，让小幅度的视线波动仍然被判断为注视状态
        
        Args:
            state: 当前注意力状态标志 (0=注视, 1=扫视, -1=未知)
            velocity: 视线速度
            angles: 当前视线角度
        
        Returns:
            int: 处理后的注意力状态
        """
        try:
            # 初始化速度大小变量
            velocity_magnitude = 0.0
            
            # 如果当前被判断为扫视（flag=1），检查是否为小幅波动
            if state == 1:
                # 计算速度大小
                if isinstance(velocity, (list, np.ndarray)) and len(velocity) >= 2:
                    velocity_magnitude = np.linalg.norm(velocity)
                else:
                    velocity_magnitude = 0.0
                
                # 使用配置的波动容差阈值
                fluctuation_threshold = self.fluctuation_threshold  # 度/秒，低于此值认为是小幅波动
                
                # 如果速度小于波动阈值，且当前状态持续时间较短，则保持为注视状态
                if velocity_magnitude < fluctuation_threshold:
                    # 检查当前扫视状态的持续时间
                    current_time = time.time()
                    saccade_duration = 0  # 初始化变量
                    
                    if self.saccade_start > 0:
                        saccade_duration = current_time - self.saccade_start
                        # 如果扫视时间很短，认为是波动而不是真正的扫视
                        if saccade_duration < self.fluctuation_duration:
                            return 0  # 改为注视状态
                    
                    # 如果没有历史记录，直接判断为注视
                    return 0
            
            return state
    
        except Exception as e:
            logger.error(f"波动容差处理失败: {e}")
            return state
    
    def _calculate_attention_vector(self, state: int, position: List[float], gaze_vector: np.ndarray) -> np.ndarray:
        """
        计算注意力向量
        
        Args:
            state: 注意力状态标志 (0=注视, 1=扫视, -1=未知)
            position: 注意力位置 [pitch, yaw]
            gaze_vector: 原始视线向量
            
        Returns:
            np.ndarray: 注意力向量
        """
        try:
            if state == 0:  # 正在注视，使用注意力分析结果
                at_pitch = position[0] * np.pi / 180  # 度转弧度
                at_yaw = position[1] * np.pi / 180    # 度转弧度
                attention_vec = -np.array([
                    np.cos(at_pitch) * np.sin(at_yaw),
                    np.sin(at_pitch),
                    np.cos(at_pitch) * np.cos(at_yaw)
                ])
                attention_vec = attention_vec / np.linalg.norm(attention_vec)
            else:  # 未注视(flag=1)或未知(flag=-1)，使用原始视线向量
                attention_vec = gaze_vector
                
            return attention_vec
            
        except Exception as e:
            logger.error(f"注意力向量计算失败: {e}")
            return gaze_vector
    
    def reset_state(self):
        """重置所有状态"""
        try:
            self.fixation_start = 0
            self.saccade_start = 0
            self.attention_states.clear()
            self.focusing = 0
            self.last_gaze = None
            self.last_position = None
            self.last_face_center = None
            logger.info("视线跟踪器状态已重置")
            
        except Exception as e:
            logger.error(f"状态重置失败: {e}")
    
 