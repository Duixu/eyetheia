#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视线估计和注意力分析线程 - 包含人脸检测功能
"""

import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from base_thread import BaseThread
from logger_config import get_logger

# 导入模型
from .gaze_estimator import GazeEstimator
from .gaze_tracker import GazeTracker

# 获取日志器
logger = get_logger(__name__)

class GazeAndAttentionThread(BaseThread):
    """视线估计和注意力分析线程 - 包含人脸检测功能"""
    
    def __init__(self, data_manager, device: str = 'cpu'):
        """
        初始化视线和注意力分析线程
        
        Args:
            data_manager: 数据管理器实例
            device: 计算设备
        """
        super().__init__(name="GazeAndAttentionThread", data_manager=data_manager, device=device)
        
        # 视线估计器
        self.gaze_estimator = None
        
        # 视线跟踪器
        self.gaze_tracker = None
        
        # 检测结果缓存
        self._last_attention_result = None
        
    def initialize_models(self) -> bool:
        """初始化视线估计和跟踪模型"""
        try:
            # 初始化视线估计器
            self.gaze_estimator = GazeEstimator(device=self.device)
            
            # 初始化视线跟踪器
            self.gaze_tracker = GazeTracker()
            return True
        except Exception as e:
            print(e)
    
    def run(self):
        """视线和注意力分析线程主循环"""
        if not self.initialize_models():
            logger.error("视线估计和跟踪模型初始化失败，线程退出")
            return
        
        logger.info("视线和注意力分析线程开始运行")
        
        while self.running:
            try:
                if not self.wait_if_paused():
                    continue
                # 获取图像
                image_data = self.data_manager.get_image_for_attention(timeout=1.0)

                if image_data is None:
                    continue
                
                frame, timestamp, sequence = image_data
                
                # 事件 等待人脸检测完成
                if not self.data_manager.wait_for_face_detection_completed(timeout=0.1):
                    continue
                
                # 获取人脸检测结果
                face_result = self.data_manager.get_face_result()
                if face_result is None or not face_result.get('face_detected', False):
                    continue
                
                # 获取人脸关键点
                landmarks = face_result.get('landmarks', [])
                if landmarks.shape[0] == 0:
                    logger.error("人脸关键点为空")
                    return None

                # 模型初始化
                if self.gaze_estimator is None or self.gaze_tracker is None:
                    logger.error("获取视线估计和跟踪器实例失败")
                    return None
                
                # 视线估计
                gaze_start_time = time.time()
                gaze_result = self.gaze_estimator.
                ard(frame, landmarks)
                gaze_inference_time = (time.time() - gaze_start_time) * 1000  # 转换为毫秒    

                # 视线跟踪和注意力分析
                attention_start_time = time.time()    
                
                # 提取视线数据
                pitch = gaze_result.get('pitch', 0.0)
                yaw = gaze_result.get('yaw', 0.0)
                gaze_vector = gaze_result.get('gaze_vector', np.array([0.0, 0.0, 0.0]))    
                
                # 计算人脸中心点
                face_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2, 0])
                # 使用视线跟踪器分析注意力
                attention_tuple = self.gaze_tracker.process_gaze_data(
                    pitch=pitch,
                    yaw=yaw,
                    gaze_vector=gaze_vector,
                    face_center=face_center,
                )
                # 计算注意力分析耗时
                attention_inference_time = (time.time() - attention_start_time) * 1000  # 转换为毫秒
                
                # 将元组转换为字典，合并视线追踪和注意力信息
                attention_result = {
                    'attention_pitch': attention_tuple[0],
                    'attention_yaw': attention_tuple[1],
                    'focusing': attention_tuple[2],
                    'attention_vector': attention_tuple[3],
                    'timestamp': timestamp,
                }
                # 缓存结果
                self._last_attention_result = attention_result
                 # 更新检测统计
                logger.custom(f"attention --->完成 序列号: {sequence}, "
                              f"注意力: {attention_result.get('focusing', 0)}, "
                              f"方向: ({attention_result.get('attention_pitch', 0):.2f}, {attention_result.get('attention_yaw', 0):.2f}), "
                              f"方向向量: {attention_result.get('attention_vector', np.array([0.0, 0.0, 0.0]))}, "
                              f"时间戳: {attention_result.get('timestamp', 0)}, "
                              f"视线推理时间: {gaze_inference_time:.2f}ms, "
                              f"注意力分析时间: {attention_inference_time:.2f}ms")
                if attention_result :
                    self.data_manager.update_result(attention_result, 'attention')
            except Exception as e:
                logger.error(f"视线和注意力分析线程运行出错: {e}")
                time.sleep(0.1)
        logger.info("视线和注意力分析线程已停止")
    
    def get_last_attention_result(self) -> Optional[Dict[str, Any]]:
        """获取最后一次注意力分析结果"""
        return self._last_attention_result
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.gaze_estimator is not None:
                del self.gaze_estimator
                self.gaze_estimator = None
            
            if self.gaze_tracker is not None:
                self.gaze_tracker.reset_state()
                del self.gaze_tracker
                self.gaze_tracker = None
            logger.info("视线和注意力分析线程资源已清理")
        except Exception as e:
            logger.error(f"清理视线和注意力分析线程资源失败: {e}")
    
