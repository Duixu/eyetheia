#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation
import tkinter as tk
import platform

class CoordinateTransformer:
    """坐标系转换工具 - 相机Z轴指向人脸"""
    
    def __init__(self, camera_position=None, screen_size=None, screen_resolution=None):
        """
        初始化坐标系转换器
        
        Args:
            camera_position: 相机相对于屏幕的位置 (x, y, z) 单位：米
            screen_size: 屏幕物理尺寸 (width, height) 单位：米
            screen_resolution: 屏幕分辨率 (width, height) 单位：像素
        """
        # 默认相机位置：屏幕正前方1米处
        self.camera_position = camera_position if camera_position is not None else np.array([0.0, 0.0, 1.0])
        
        # 自动获取屏幕尺寸和分辨率
        self.screen_resolution = screen_resolution if screen_resolution is not None else self._get_screen_resolution()
        self.screen_size = screen_size if screen_size is not None else self._get_screen_physical_size()
        
        # 屏幕中心到相机中心的距离
        self.distance_to_screen = self.camera_position[2]
        
        # 计算屏幕坐标系到相机坐标系的转换矩阵
        self._compute_transformation_matrix()
    
    def _get_screen_resolution(self):
        """自动获取屏幕分辨率"""
        try:
            root = tk.Tk()
            root.withdraw()  # 隐藏窗口
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return np.array([screen_width, screen_height])
        except:
            # 如果无法获取，使用默认值
            return np.array([1920, 1080])
    
    def _get_screen_physical_size(self):
        """自动获取屏幕物理尺寸"""
        try:
            root = tk.Tk()
            root.withdraw()  # 隐藏窗口
            
            # 获取屏幕DPI
            dpi_x = root.winfo_fpixels('1i')  # 水平DPI
            dpi_y = root.winfo_fpixels('1i')  # 垂直DPI
            
            # 获取屏幕分辨率
            screen_width_px = root.winfo_screenwidth()
            screen_height_px = root.winfo_screenheight()
            
            root.destroy()
            
            # 计算物理尺寸（英寸转米）
            width_inches = screen_width_px / dpi_x
            height_inches = screen_height_px / dpi_y
            
            width_meters = width_inches * 0.0254  # 英寸转米
            height_meters = height_inches * 0.0254
            
            return np.array([width_meters, height_meters])
        except:
            # 如果无法获取，使用默认值（24英寸显示器）
            return np.array([0.53, 0.30])
    
    def _compute_transformation_matrix(self):
        """计算屏幕坐标系到相机坐标系的转换矩阵"""
        # 屏幕坐标系：原点在屏幕中心，X向右，Y向上，Z指向屏幕外（朝向用户）
        # 相机坐标系：原点在相机中心，X向右，Y向下，Z指向人脸（远离屏幕）
        
        # 相机相对于屏幕中心的偏移
        camera_offset_x = self.camera_position[0]
        camera_offset_y = self.camera_position[1]
        
        # 计算旋转矩阵
        # 1. 相机Y轴向下，屏幕Y轴向上，需要绕X轴旋转180度
        # 2. 相机Z轴指向人脸（远离屏幕），屏幕Z轴指向用户，需要绕Y轴旋转180度
        rotation_x = Rotation.from_euler('x', 180, degrees=True)
        rotation_y = Rotation.from_euler('y', 180, degrees=True)
        rotation_matrix = (rotation_y * rotation_x).as_matrix()
        
        # 计算平移向量
        # 相机在屏幕坐标系中的位置
        translation_vector = np.array([-camera_offset_x, camera_offset_y, -self.distance_to_screen])
        
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
    
    def camera_to_screen_angles(self, camera_pitch, camera_yaw):
        """
        将相机坐标系下的角度转换为屏幕坐标系下的角度
        
        Args:
            camera_pitch: 相机坐标系下的pitch角度（弧度）
            camera_yaw: 相机坐标系下的yaw角度（弧度）
            
        Returns:
            screen_pitch, screen_yaw: 屏幕坐标系下的角度（弧度）
        """
        # 1. 将角度转换为相机坐标系下的视线向量
        # 相机坐标系：Z轴指向人脸，X轴向右，Y轴向下
        camera_gaze_vector = np.array([
            np.cos(camera_pitch) * np.sin(camera_yaw),
            -np.sin(camera_pitch),  # 注意Y轴方向
            np.cos(camera_pitch) * np.cos(camera_yaw)
        ])
        
        # 2. 转换到屏幕坐标系
        screen_gaze_vector = self.rotation_matrix @ camera_gaze_vector
        
        # 3. 计算屏幕坐标系下的角度
        # 屏幕坐标系：Z轴指向用户，X轴向右，Y轴向上
        screen_pitch = np.arcsin(screen_gaze_vector[1])
        screen_yaw = np.arctan2(screen_gaze_vector[0], screen_gaze_vector[2])
        
        return screen_pitch, screen_yaw
    
    def screen_to_camera_angles(self, screen_pitch, screen_yaw):
        """
        将屏幕坐标系下的角度转换为相机坐标系下的角度
        
        Args:
            screen_pitch: 屏幕坐标系下的pitch角度（弧度）
            screen_yaw: 屏幕坐标系下的yaw角度（弧度）
            
        Returns:
            camera_pitch, camera_yaw: 相机坐标系下的角度（弧度）
        """
        # 1. 将角度转换为屏幕坐标系下的视线向量
        # 屏幕坐标系：Z轴指向用户，X轴向右，Y轴向上
        screen_gaze_vector = np.array([
            np.cos(screen_pitch) * np.sin(screen_yaw),
            np.sin(screen_pitch),
            np.cos(screen_pitch) * np.cos(screen_yaw)
        ])
        
        # 2. 转换到相机坐标系
        camera_gaze_vector = self.rotation_matrix.T @ screen_gaze_vector
        
        # 3. 计算相机坐标系下的角度
        # 相机坐标系：Z轴指向人脸，X轴向右，Y轴向下
        camera_pitch = -np.arcsin(camera_gaze_vector[1])  # 注意Y轴方向
        camera_yaw = np.arctan2(camera_gaze_vector[0], camera_gaze_vector[2])
        
        return camera_pitch, camera_yaw
    
    def get_screen_point_from_angles(self, pitch_deg, yaw_deg):
        """
        根据角度计算屏幕上的点坐标
        
        Args:
            pitch_deg: pitch角度（度）- 正值向上看，负值向下看
            yaw_deg: yaw角度（度）- 正值向右看，负值向左看
            
        Returns:
            screen_x, screen_y: 屏幕上的点坐标（像素）
        """
        # 转换为弧度
        pitch_rad = np.radians(pitch_deg)
        yaw_rad = np.radians(yaw_deg)
        
        # 相机位置
        camera_x = self.camera_position[0]  # 相机在屏幕坐标系中的X位置（米）
        camera_y = self.camera_position[1]  # 相机在屏幕坐标系中的Y位置（米）
        camera_z = self.camera_position[2]  # 相机到屏幕的距离（米）
        
        # 添加缩放因子来调整敏感度
        sensitivity_scale = 1  # 可以调整这个值来改变敏感度
        
        # 使用相机到屏幕的距离计算偏移量
        # 计算角度变化对应的屏幕偏移（米）
        offset_x_meters = camera_z * np.tan(yaw_rad) * sensitivity_scale
        offset_y_meters = camera_z * np.tan(pitch_rad) * sensitivity_scale
        
        # 转换为像素坐标
        # 屏幕坐标系：原点在屏幕中心，X向右，Y向上
        pixel_x = (offset_x_meters / self.screen_size[0] + 0.5) * self.screen_resolution[0]
        # 屏幕Y轴向下为正，所以需要翻转Y坐标
        pixel_y = (-offset_y_meters / self.screen_size[1] + 0.5) * self.screen_resolution[1]
        
        return pixel_x, pixel_y
    
    def get_angles_from_screen_point(self, pixel_x, pixel_y):
        """
        根据屏幕上的点坐标计算角度
        
        Args:
            pixel_x: 屏幕X坐标（像素）
            pixel_y: 屏幕Y坐标（像素）
            
        Returns:
            pitch_deg, yaw_deg: 角度（度）
        """
        # 相机位置
        camera_x = self.camera_position[0]  # 相机在屏幕坐标系中的X位置（米）
        camera_y = self.camera_position[1]  # 相机在屏幕坐标系中的Y位置（米）
        camera_z = self.camera_position[2]  # 相机到屏幕的距离（米）
        
        # 添加缩放因子，与get_screen_point_from_angles保持一致
        sensitivity_scale = 1
        
        # 转换为屏幕坐标（米）
        screen_x = (pixel_x / self.screen_resolution[0] - 0.5) * self.screen_size[0]
        # 屏幕Y轴向下为正，所以需要翻转Y坐标
        screen_y = (-(pixel_y / self.screen_resolution[1] - 0.5)) * self.screen_size[1]
        
        # 使用相机到屏幕的距离计算角度
        # 计算从相机到屏幕点的偏移量
        offset_x = screen_x - camera_x
        offset_y = screen_y - camera_y
        
        # 计算角度
        yaw_rad = np.arctan2(offset_x, camera_z * sensitivity_scale)
        pitch_rad = np.arctan2(offset_y, camera_z * sensitivity_scale)
        
        return np.degrees(pitch_rad), np.degrees(yaw_rad)
    
    def get_screen_info(self):
        """获取屏幕信息"""
        return {
            "resolution": self.screen_resolution,
            "physical_size": self.screen_size,
            "camera_position": self.camera_position,
            "distance_to_screen": self.distance_to_screen
        }

# 预定义的相机位置配置
CAMERA_POSITIONS = {
    'center': np.array([0.0, 0.0, 0.6]),  # 屏幕正前方0.5米
    'left': np.array([-0.3, 0.0, 0.5]),   # 屏幕左侧
    'right': np.array([0.3, 0.0, 0.5]),   # 屏幕右侧
    'top': np.array([0.0, 0.2, 0.5]),     # 屏幕上方
    'bottom': np.array([0.0, -0.2, 0.5]), # 屏幕下方
}

def create_coordinate_transformer(camera_position_name='center', custom_position=None, 
                                screen_size=None, screen_resolution=None, calibration_data=None):
    """
    创建坐标系转换器
    
    Args:
        camera_position_name: 预定义的相机位置名称
        custom_position: 自定义相机位置 (x, y, z)
        screen_size: 屏幕物理尺寸 (width, height) 单位：米
        screen_resolution: 屏幕分辨率 (width, height) 单位：像素
        calibration_data: 校准数据字典
        
    Returns:
        CoordinateTransformer: 坐标系转换器实例
    """
    if calibration_data is not None and calibration_data.get('distance_to_screen') is not None:
        # 使用校准数据
        position = np.array([
            calibration_data.get('camera_offset_x', 0.0),
            calibration_data.get('camera_offset_y', 0.0),
            calibration_data.get('distance_to_screen', 1.0)
        ])
        screen_size = calibration_data.get('screen_physical_size', screen_size)
        screen_resolution = calibration_data.get('screen_resolution', screen_resolution)
    elif custom_position is not None:
        position = np.array(custom_position)
    elif camera_position_name in CAMERA_POSITIONS:
        position = CAMERA_POSITIONS[camera_position_name]
    else:
        raise ValueError(f"未知的相机位置: {camera_position_name}")
    
    return CoordinateTransformer(
        camera_position=position,
        screen_size=screen_size,
        screen_resolution=screen_resolution
    ) 