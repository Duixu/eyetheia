#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC标定方法
参考项目的标定实现，使用最小二乘法求解旋转矩阵和平移向量
"""

import time
import numpy as np
import cv2
import sys
import os
from scipy.optimize import leastsq
import yaml

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def find_plane_equation(xo1, yo1, zo1, xo2, yo2, zo2, xo3, yo3, zo3):
    """计算平面方程参数"""
    a = (yo2 - yo1) * (zo3 - zo1) - (zo2 - zo1) * (yo3 - yo1)
    b = (xo3 - xo1) * (zo2 - zo1) - (xo2 - xo1) * (zo3 - zo1)
    c = (xo2 - xo1) * (yo3 - yo1) - (xo3 - xo1) * (yo2 - yo1)
    d = -(a * xo1 + b * yo1 + c * zo1)
    equation_parameters = np.array([a, b, c, d])
    return equation_parameters

def find_intersection(x1, y1, z1, m, n, p, a, b, c, d):
    """计算直线与平面的交点"""
    t = (-a * x1 - b * y1 - c * z1 - d) / (a * m + b * n + c * p)
    x = m * t + x1
    y = n * t + y1
    z = p * t + z1
    intersection = np.array([x, y, z])
    return intersection

def angle_diff(vector1, vector2):
    """计算两个向量的角度差"""
    angle = np.arccos(vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    angle_diff = np.degrees(angle)
    return angle_diff

class ARCCalibrator:
    def __init__(self, cali_nums=9):
        """初始化ARC标定器"""
        self.cali_nums = cali_nums
        
        # 自动获取屏幕参数
        self._get_screen_parameters()
        
        # 标定点位置（屏幕坐标）- 紧靠屏幕边缘
        margin_x = 10  # 水平边距，紧靠屏幕边缘
        margin_y = 10  # 垂直边距，紧靠屏幕边缘
        self.offset = 0
        if cali_nums == 9:
            self.cali_point = [
                (self.width // 2, self.height // 2),  # 中心
                (margin_x, margin_y),  # 左上
                (self.width - margin_x, margin_y),  # 右上
                (margin_x, self.height - margin_y),  # 左下
                (self.width - margin_x, self.height - margin_y),  # 右下
                (self.width // 2, margin_y),  # 上中
                (self.width // 2, self.height - margin_y),  # 下中
                (margin_x, self.height // 2),  # 左中
                (self.width - margin_x, self.height // 2),  # 右中
            ]
        elif cali_nums == 5:
            self.cali_point = [
                (self.width // 2, self.height // 2),  # 中心
                (margin_x, margin_y),  # 左上
                (self.width - margin_x, margin_y),  # 右上
                (margin_x, self.height - margin_y),  # 左下
                (self.width - margin_x, self.height - margin_y),  # 右下
            ]
        elif cali_nums == 3:
            self.cali_point = [
                (self.width // 2, self.height // 2),  # 中心
                (margin_x - self.offset, margin_y - self.offset),  # 左上
                (self.width - margin_x + self.offset, self.height - margin_y + self.offset),  # 右下
            ]
        
        # 样本收集配置
        self.current_samples = []
        self.current_face_samples = []
        self.collecting_samples = False
        
        # 时间控制配置
        self.required_duration = 3.0  # 每个标定点需要注视的时间（秒）
        self.collection_start_time = 0  # 开始收集样本的时间
        
        # 标定数据
        self.attention_vec_average = []
        self.face_center_average = []
        self.pitch = []
        self.yaw = []
        
        # 标定状态
        self.cali_ready = False
        self.cali_start = 0
        self.notfocusingstart = 0
        self.auto_start_enabled = True  # 是否启用自动开始机制
        self.auto_start_delay = 3.0  # 自动开始延迟（秒）
        
        # 标定结果
        self.rotation_matrix = np.eye(3)
        self.inverse_matrix = np.linalg.inv(self.rotation_matrix)
        self.tvec = [0, 0, 0]
        self.realworld_point_pos = []
        self.attention_vec_hat = []  # 添加attention_vec_hat
        self.angle_error = []
        self.inverse_pos = []
        self.iteration_vec = []  # 添加iteration_vec
        self.final_pos = []  # 添加final_pos
        self.coeff_pitch = [1.0, 0.0]
        self.coeff_yaw = [1.0, 0.0]
        self.origaze_pos = []  # 添加origaze_pos
        self.iterationnum = 0
        self.last_attention_vec = None
    
    def get_calibration_point(self, index):
        """获取指定索引的标定点坐标，只用于屏幕显示坐标，不用于标定计算
        
        参数:
            index (int): 标定点索引 (0 到 cali_nums-1)
            
        返回:
            tuple: (x, y) 标定点坐标，已消除offset影响
        """
        if index < 0 or index >= self.cali_nums:
            raise ValueError(f"标定点索引超出范围: {index}, 有效范围: 0-{self.cali_nums-1}")
        
        # 获取原始标定点坐标
        x, y = self.cali_point[index]
        
        # 对于3点标定，消除offset的影响
        if self.cali_nums == 3:  
            if index == 0:  # 中心点，不需要修改
                return (x, y)
            elif index == 1:  # 左上点，消除offset影响
                return ( x + self.offset, y + self.offset)
            elif index == 2:  # 右下点，消除offset影响
                return (x - self.offset, y - self.offset)
        # 对于其他标定点数量，直接返回原始坐标
        return (x, y)
    
    def get_all_calibration_points(self):
        """获取所有标定点坐标，已消除offset影响
        
        返回:
            list: 所有标定点的坐标列表 [(x1, y1), (x2, y2), ...]
        """
        points = []
        for i in range(self.cali_nums):
            points.append(self.get_calibration_point(i))
        return points
    
    def _get_screen_parameters(self):
        """自动获取屏幕分辨率和物理尺寸"""
        try:
            import tkinter as tk
            root = tk.Tk()
            
            # 获取屏幕分辨率
            self.width = root.winfo_screenwidth()
            self.height = root.winfo_screenheight()
            
            # 获取屏幕物理尺寸（通过DPI计算）
            dpi = root.winfo_fpixels('1i')  # 获取DPI
            if dpi > 0:
                # 通过DPI计算物理尺寸（英寸转米）
                self.a_w = self.width / dpi * 0.0254  # 英寸转米
                self.a_h = self.height / dpi * 0.0254
            else:
                # 如果无法获取DPI，使用默认值
                self.a_w = 0.344  # 默认屏幕物理宽度（米）
                self.a_h = 0.194  # 默认屏幕物理高度（米）
            
            root.destroy()
            
            print(f"自动获取屏幕参数:")
            print(f"  分辨率: {self.width} x {self.height}")
            print(f"  物理尺寸: {self.a_w:.3f}m x {self.a_h:.3f}m")
            
        except Exception as e:
            print(f"自动获取屏幕参数失败，使用默认值: {e}")
            # 使用默认值
            self.width = 1920
            self.height = 1080
            self.a_w = 0.344  # 屏幕物理宽度（米）
            self.a_h = 0.194  # 默认屏幕物理高度（米）
    
    def set_required_duration(self, duration_seconds=3.0):
        """设置每个标定点需要的注视时间"""
        self.required_duration = duration_seconds
        print(f"注视时间设置为: {duration_seconds}秒")
    
    def get_elapsed_time(self):
        """获取当前注视时间"""
        if self.collecting_samples and self.collection_start_time > 0:
            return time.time() - self.collection_start_time
        return 0.0
    
    def enable_auto_start(self, enabled=True, delay=1.0):
        """启用或禁用自动开始机制"""
        self.auto_start_enabled = enabled
        self.auto_start_delay = delay
        print(f"自动开始机制: {'启用' if enabled else '禁用'}, 延迟: {delay}秒")
    
    def start_collecting_samples(self):
        """手动开始收集样本"""
        self.collecting_samples = True
        self.current_samples = []
        self.current_face_samples = []
        self.collection_start_time = time.time()  # 设置开始收集时间
        print("手动开始收集样本...")
    
    def stop_collecting_samples(self):
        """停止收集样本"""
        self.collecting_samples = False
        self.current_samples = []
        self.current_face_samples = []
        self.collection_start_time = 0
    
    def calibration(self, num, frame):
        """执行标定,收集多个样本并计算平均值"""
        if self.notfocusingstart == 0:
            self.notfocusingstart = time.time()
        if self.cali_start == 0:
            self.cali_start = time.time()
        
        # 这里应该从核心线程管理器获取数据，而不是直接处理帧
        # 为了兼容性，我们暂时使用模拟数据
        # 在实际使用中，应该从CalibrationAPI传入处理后的数据
        
        # 模拟注意力向量和人脸中心（实际应该从核心系统获取）
        attention_vec = np.array([0, 0, 1.0])  # 默认值
        face_center = np.array([0, 0, 0.5])    # 默认
        
        # 计算角度
        pitch = np.arcsin(-attention_vec[1])
        yaw = np.arcsin(-attention_vec[0] / np.cos(pitch))
        
        self.pitch.append(pitch)
        self.yaw.append(yaw)
        
        # 自动开始机制（如果启用）
        if self.auto_start_enabled and not self.collecting_samples and (time.time() - self.cali_start) > self.auto_start_delay:
            self.collecting_samples = True
            self.current_samples = []
            self.current_face_samples = []
            self.collection_start_time = time.time()  # 设置开始收集时间
            print(f"自动开始收集标定点 {num + 1} 的样本...")
        
        # 收集样本
        if self.collecting_samples:
            current_time = time.time()
            self.current_samples.append(attention_vec)
            self.current_face_samples.append(face_center)
            
            # 检查是否达到要求的注视时间
            elapsed_time = current_time - self.collection_start_time
            if elapsed_time >= self.required_duration:
                # 计算平均值
                avg_attention_vec = self.current_samples[-1]
                avg_face_center = self.current_face_samples[-1] 
                
                # 添加到平均列表
                self.attention_vec_average.append(avg_attention_vec)
                self.face_center_average.append(avg_face_center)
                
                print(f"标定点 {num + 1} 注视完成（{elapsed_time:.1f}秒），平均值: {avg_attention_vec}")
                
                # 重置状态（但不重置cali_start，避免影响下一个点）
                self.collecting_samples = False
                self.current_samples = []
                self.current_face_samples = []
                self.collection_start_time = 0
                    
                return True, True  # 当前点完成
        
        return True, False  # 继续收集
    
    def equations(self, r):
        """标定方程组"""
        a, b, c, d = 0, 0, 1, 0
        
        if self.cali_nums == 9:
            f = np.zeros(19)
        elif self.cali_nums == 5:
            f = np.zeros(11)
        elif self.cali_nums == 3:
            f = np.zeros(7)
        
        x, y, z, w = r[0], r[1], r[2], r[3]
        t1, t2, t3 = r[4], r[5], r[6]
        tvec = [t1, t2, t3]
        
        # 旋转矩阵（四元数）
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])
        
        for i in range(self.cali_nums):
            face_center = self.face_center_average[i]
            face_center_rotated = np.dot(R, face_center) + tvec
            attention_vec = self.attention_vec_average[i]
            attention_vec_rotated = np.dot(R, attention_vec)
            
            dot = find_intersection(
                face_center_rotated[0], face_center_rotated[1], face_center_rotated[2],
                attention_vec_rotated[0], attention_vec_rotated[1], attention_vec_rotated[2],
                a, b, c, d
            )
            
            dot_x, dot_y = dot[0], dot[1]
            intersection_x = self.realworld_point_pos[i][0]
            intersection_y = self.realworld_point_pos[i][1]
            
            f[2*i] = dot_x - intersection_x
            f[2*i+1] = dot_y - intersection_y
        
        # 四元数约束
        f[self.cali_nums*2] = x**2 + y**2 + z**2 + w**2 - 1
        
        return f
    
    def rotation(self):
        """计算旋转矩阵和平移向量"""
        a, b, c, d = 0, 0, 1, 0
        
        # 计算真实世界坐标
        for i in range(len(self.cali_point)):
            self.realworld_point_pos.append([
                self.cali_point[i][0] / self.width * self.a_w,
                -self.cali_point[i][1] / self.height * self.a_h,
                0
            ])
        
        # 使用最小二乘法求解
        r = leastsq(self.equations, [0, 0, 1, 0, self.a_w/2, -self.a_h/2, 0.5])
        
        x, y, z, w = r[0][0], r[0][1], r[0][2], r[0][3]
        t1, t2, t3 = r[0][4], r[0][5], r[0][6]
        
        # 计算旋转矩阵
        self.rotation_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])
        
        self.tvec = [t1, t2, t3]
        self.inverse_matrix = np.linalg.inv(self.rotation_matrix)
        
        # 计算attention_vec_hat（预测的注意力向量）
        self.attention_vec_hat = []
        for i in range(len(self.cali_point)):
            face_center = self.face_center_average[i]
            face_center_rotated = np.dot(self.rotation_matrix, face_center) + self.tvec
            attention_vec = self.attention_vec_average[i]
            attention_vec_rotated = np.dot(self.rotation_matrix, attention_vec)
            
            # 计算交点
            dot = find_intersection(
                face_center_rotated[0], face_center_rotated[1], face_center_rotated[2],
                attention_vec_rotated[0], attention_vec_rotated[1], attention_vec_rotated[2],
                a, b, c, d
            )
            
            # 从交点反向计算注意力向量
            direction = dot - face_center_rotated
            direction = direction / np.linalg.norm(direction)
            
            # 转换回原始坐标系
            attention_vec_hat = np.dot(self.inverse_matrix, direction)
            self.attention_vec_hat.append(attention_vec_hat)
        
        # 执行迭代优化
        print("开始迭代优化...")
        self.iteration()
        
        # 计算回归系数
        print("计算回归系数...")
        self.regression()
        
        # 打印标定质量分析
        self._print_calibration_quality()
        
        return self.rotation_matrix, self.tvec, self.a_w, self.a_h 
    
    def iteration(self):
        """迭代优化注意力向量"""
        if_iteration = True
        num = 0
        
        while if_iteration and num <= 500:
            if_iteration = False
            self.iteration_vec = self.attention_vec_hat.copy()
            
            # 检查角度差异
            for i in range(self.cali_nums):
                vec1 = self.attention_vec_average[i]
                vec2 = self.iteration_vec[i]
                L1 = np.sqrt(vec1.dot(vec1))
                L2 = np.sqrt(vec2.dot(vec2))
                cos_angle = vec1.dot(vec2) / (L1 * L2)
                angle = np.arccos(cos_angle)
                angle2 = angle * 180 / np.pi
                if angle2 >= 0.5:
                    if_iteration = True
                    break
            
            # 更新迭代向量
            iteration_vec = []
            for i in range(len(self.cali_point)):
                upgraded_vec = 0.6 * self.attention_vec_average[i] + 0.4 * self.attention_vec_hat[i]
                length = np.linalg.norm(upgraded_vec)
                upgraded_vec = np.divide(upgraded_vec, length)
                iteration_vec.append(upgraded_vec)
            
            self.iteration_vec = iteration_vec
            print(f"{num} iteration:")
            
            # 重新计算旋转矩阵
            self.rotation_iteration()
            self.rotation_inverse()
            num += 1
        
        # 计算最终位置
        self.final_pos = []
        for i in range(len(self.cali_point)):
            face_center = self.face_center_average[i]
            face_center_rotated = np.dot(self.rotation_matrix, face_center) + self.tvec
            iteration_vec = self.iteration_vec[i]
            iteration_vec_rotated = np.dot(self.rotation_matrix, iteration_vec)
            dot = find_intersection(
                face_center_rotated[0], face_center_rotated[1], face_center_rotated[2],
                iteration_vec_rotated[0], iteration_vec_rotated[1], iteration_vec_rotated[2],
                0, 0, 1, 0
            )
            pos_x = dot[0] / self.a_w * self.width
            pos_y = -dot[1] / self.a_h * self.height
            pos = [pos_x, pos_y]
            #迭代后的标定点注意力向量在屏幕上的位置
            self.final_pos.append(pos)
        
        # 计算原始位置
        self.origaze_pos = []
        for i in range(len(self.cali_point)):
            face_center = self.face_center_average[i]
            face_center_rotated = np.dot(self.rotation_matrix, face_center) + self.tvec
            attention_vec = self.attention_vec_average[i]
            attention_vec_rotated = np.dot(self.rotation_matrix, attention_vec)
            dot = find_intersection(
                face_center_rotated[0], face_center_rotated[1], face_center_rotated[2],
                attention_vec_rotated[0], attention_vec_rotated[1], attention_vec_rotated[2],
                0, 0, 1, 0
            )
            pos_x = dot[0] / self.a_w * self.width
            pos_y = -dot[1] / self.a_h * self.height
            pos = [pos_x, pos_y]
            #原始标定点注意力向量在屏幕上的位置
            self.origaze_pos.append(pos)
    
    def rotation_iteration(self):
        """迭代版本的旋转矩阵计算"""
        a, b, c, d = 0, 0, 1, 0
        
        # 计算真实世界坐标
        self.realworld_point_pos = []
        for i in range(len(self.cali_point)):
            self.realworld_point_pos.append([
                self.cali_point[i][0] / self.width * self.a_w,
                -self.cali_point[i][1] / self.height * self.a_h,
                0
            ])
        
        # 使用最小二乘法求解
        r = leastsq(self.equations_iteration, [0, 0, 1, 0, self.a_w/2, -self.a_h/2, 0.5])
        
        x, y, z, w = r[0][0], r[0][1], r[0][2], r[0][3]
        t1, t2, t3 = r[0][4], r[0][5], r[0][6]
        
        # 计算旋转矩阵
        self.rotation_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])
        
        self.tvec = [t1, t2, t3]
    
    def equations_iteration(self, r):
        """迭代版本的标定方程组"""
        a, b, c, d = 0, 0, 1, 0
        
        if self.cali_nums == 9:
            f = np.zeros(19)
        elif self.cali_nums == 5:
            f = np.zeros(11)
        elif self.cali_nums == 3:
            f = np.zeros(7)
        
        x, y, z, w = r[0], r[1], r[2], r[3]
        t1, t2, t3 = r[4], r[5], r[6]
        tvec = [t1, t2, t3]
        
        # 旋转矩阵（四元数）
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])
        
        for i in range(self.cali_nums):
            face_center = self.face_center_average[i]
            face_center_rotated = np.dot(R, face_center) + tvec
            attention_vec = self.iteration_vec[i]  # 使用迭代向量
            attention_vec_rotated = np.dot(R, attention_vec)
            
            dot = find_intersection(
                face_center_rotated[0], face_center_rotated[1], face_center_rotated[2],
                attention_vec_rotated[0], attention_vec_rotated[1], attention_vec_rotated[2],
                a, b, c, d
            )
            
            dot_x, dot_y = dot[0], dot[1]
            intersection_x = self.realworld_point_pos[i][0]
            intersection_y = self.realworld_point_pos[i][1]
            
            f[2*i] = dot_x - intersection_x
            f[2*i+1] = dot_y - intersection_y
        
        # 四元数约束
        f[self.cali_nums*2] = x**2 + y**2 + z**2 + w**2 - 1
        
        return f
    
    def rotation_inverse(self):
        """计算旋转矩阵的逆矩阵和相关数据"""
        self.inverse_matrix = np.linalg.inv(self.rotation_matrix)
        
        # 计算attention_vec_hat（如果还没有计算的话）
        if not hasattr(self, 'attention_vec_hat') or len(self.attention_vec_hat) == 0:
            self.attention_vec_hat = []
            for i in range(len(self.cali_point)):
                face_center_rotated = np.dot(self.rotation_matrix, self.face_center_average[i]) + self.tvec
                
                # 从真实世界坐标到人脸中心的向量
                attention_vec_rotated = np.array([
                    self.realworld_point_pos[i][0] - face_center_rotated[0],
                    self.realworld_point_pos[i][1] - face_center_rotated[1],
                    self.realworld_point_pos[i][2] - face_center_rotated[2]
                ])
                
                # 转换回原始坐标系
                attention_vec_hat = np.dot(self.inverse_matrix, attention_vec_rotated)
                attention_vec_hat = attention_vec_hat / np.linalg.norm(attention_vec_hat)
                self.attention_vec_hat.append(attention_vec_hat)
        
        # 计算角度误差
        self.angle_error = []
        for i in range(self.cali_nums):
            vec1 = self.attention_vec_average[i]
            vec2 = self.attention_vec_hat[i]
            L1 = np.sqrt(vec1.dot(vec1))
            L2 = np.sqrt(vec2.dot(vec2))
            cos_angle = vec1.dot(vec2) / (L1 * L2)
            angle = np.arccos(cos_angle)
            angle2 = angle * 180 / np.pi
            self.angle_error.append(angle2)
        
        # 计算投影位置
        self.inverse_pos = []
        for i in range(self.cali_nums):
            face_center = self.face_center_average[i]
            face_center_rotated = np.dot(self.rotation_matrix, face_center) + self.tvec
            attention_vec = self.attention_vec_average[i]
            attention_vec_rotated = np.dot(self.rotation_matrix, attention_vec)
            dot = find_intersection(
                face_center_rotated[0], face_center_rotated[1], face_center_rotated[2],
                attention_vec_rotated[0], attention_vec_rotated[1], attention_vec_rotated[2],
                0, 0, 1, 0
            )
            pos_x = dot[0] / self.a_w * self.width
            pos_y = -dot[1] / self.a_h * self.height
            pos = [pos_x, pos_y]
            self.inverse_pos.append(pos)
        
        # 调试信息
        print(f"角度误差: {[f'{err:.2f}°' for err in self.angle_error]}")
        print(f"平均角度误差: {np.mean(self.angle_error):.2f}°")
        print(f"最大角度误差: {np.max(self.angle_error):.2f}°")
    
    def regression(self):
        """计算回归系数"""
        pitch = []
        yaw = []
        pitch_iteration = []
        yaw_iteration = []
        
        for i in range(len(self.cali_point)):
            # 计算原始角度
            pitch_i = np.arcsin(-self.attention_vec_average[i][1])
            yaw_i = np.arcsin(-self.attention_vec_average[i][0] / np.cos(pitch_i))
            pitch.append(pitch_i)
            yaw.append(yaw_i)
            
            # 计算迭代后角度
            pitch_iteration_i = np.arcsin(-self.iteration_vec[i][1])
            yaw_iteration_i = np.arcsin(-self.iteration_vec[i][0] / np.cos(pitch_i))
            pitch_iteration.append(pitch_iteration_i)
            yaw_iteration.append(yaw_iteration_i)
        
        # 线性回归
        self.coeff_pitch = np.polyfit(pitch, pitch_iteration, 1)
        self.coeff_yaw = np.polyfit(yaw, yaw_iteration, 1)
        
        print(f"回归系数 - Pitch: {self.coeff_pitch}, Yaw: {self.coeff_yaw}")
    
    def save_calibration(self, filename='arc_calibration.yaml'):
        """保存标定结果"""
        calibration_data = {
            'rotation_matrix': self.rotation_matrix.tolist(),
            'tvec': self.tvec,
            'screen_width': self.width,
            'screen_height': self.height,
            'screen_physical_width': self.a_w,
            'screen_physical_height': self.a_h,
            'calibration_points': self.cali_point,
            'coeff_pitch': self.coeff_pitch.tolist() if hasattr(self.coeff_pitch, 'tolist') else self.coeff_pitch,
            'coeff_yaw': self.coeff_yaw.tolist() if hasattr(self.coeff_yaw, 'tolist') else self.coeff_yaw,
            'attention_vec_average': [vec.tolist() for vec in self.attention_vec_average],
            'face_center_average': [vec.tolist() for vec in self.face_center_average],
            'attention_vec_hat': [vec.tolist() for vec in self.attention_vec_hat],
            'iteration_vec': [vec.tolist() for vec in self.iteration_vec],
            'final_pos': [[float(pos[0]), float(pos[1])] for pos in self.final_pos] if self.final_pos else [],
            'origaze_pos': [[float(pos[0]), float(pos[1])] for pos in self.origaze_pos] if self.origaze_pos else []
        }
        
        with open(filename, 'w') as file:
            yaml.dump(calibration_data, file, default_flow_style=False)
        
        print(f"标定结果已保存到 {filename}")
    
    def load_calibration(self, filename='arc_calibration.yaml'):
        """加载标定结果"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as file:
                calibration_data = yaml.safe_load(file)
            
            self.rotation_matrix = np.array(calibration_data['rotation_matrix'])
            self.tvec = calibration_data['tvec']
            self.width = calibration_data['screen_width']
            self.height = calibration_data['screen_height']
            self.a_w = calibration_data['screen_physical_width']
            self.a_h = calibration_data['screen_physical_height']
            self.cali_point = calibration_data['calibration_points']
            self.coeff_pitch = np.array(calibration_data['coeff_pitch']) if isinstance(calibration_data['coeff_pitch'], list) else calibration_data['coeff_pitch']
            self.coeff_yaw = np.array(calibration_data['coeff_yaw']) if isinstance(calibration_data['coeff_yaw'], list) else calibration_data['coeff_yaw']
            
            # 加载额外的标定数据
            if 'attention_vec_average' in calibration_data:
                self.attention_vec_average = [np.array(vec) for vec in calibration_data['attention_vec_average']]
            if 'face_center_average' in calibration_data:
                self.face_center_average = [np.array(vec) for vec in calibration_data['face_center_average']]
            if 'attention_vec_hat' in calibration_data:
                self.attention_vec_hat = [np.array(vec) for vec in calibration_data['attention_vec_hat']]
            if 'iteration_vec' in calibration_data:
                self.iteration_vec = [np.array(vec) for vec in calibration_data['iteration_vec']]
            if 'final_pos' in calibration_data:
                self.final_pos = [[float(pos[0]), float(pos[1])] for pos in calibration_data['final_pos']] if calibration_data['final_pos'] else []
            if 'origaze_pos' in calibration_data:
                self.origaze_pos = [[float(pos[0]), float(pos[1])] for pos in calibration_data['origaze_pos']] if calibration_data['origaze_pos'] else []
            
            self.inverse_matrix = np.linalg.inv(self.rotation_matrix)
            print(f"已加载标定结果: {filename}")
            print(f"回归系数 - Pitch: {self.coeff_pitch}, Yaw: {self.coeff_yaw}")
            return True
            
        except Exception as e:
            print(f"加载标定结果失败: {e}")
            return False
    
    def _print_calibration_quality(self):
        """打印标定质量分析"""
        if not self.final_pos or not self.origaze_pos:
            print("标定质量分析：数据不足")
            return
        
        print("\n=== 标定质量分析 ===")
        print(f"标定点数量: {len(self.cali_point)}")
        print(f"屏幕分辨率: {self.width} x {self.height}")
        
        # 分析每个标定点的误差
        errors = []
        for i in range(len(self.cali_point)):
            target_x, target_y = self.cali_point[i]
            final_x, final_y = self.final_pos[i]
            orig_x, orig_y = self.origaze_pos[i]
            
            # 计算误差
            final_error = np.sqrt((target_x - final_x)**2 + (target_y - final_y)**2)
            orig_error = np.sqrt((target_x - orig_x)**2 + (target_y - orig_y)**2)
            errors.append(final_error)
            
            print(f"标定点 {i+1} ({target_x}, {target_y}):")
            print(f"  原始位置: ({orig_x:.1f}, {orig_y:.1f}), 误差: {orig_error:.1f}px")
            print(f"  优化位置: ({final_x:.1f}, {final_y:.1f}), 误差: {final_error:.1f}px")
        
        # 统计信息
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        print(f"\n误差统计:")
        print(f"  平均误差: {mean_error:.1f}px")
        print(f"  最大误差: {max_error:.1f}px")
        print(f"  最小误差: {min_error:.1f}px")
        
        # 分析左右对称性
        if len(self.cali_point) >= 5:
            left_points = [i for i, (x, y) in enumerate(self.cali_point) if x < self.width // 2]
            right_points = [i for i, (x, y) in enumerate(self.cali_point) if x > self.width // 2]
            
            left_errors = [errors[i] for i in left_points]
            right_errors = [errors[i] for i in right_points]
            
            if left_errors and right_errors:
                left_avg = np.mean(left_errors)
                right_avg = np.mean(right_errors)
                print(f"\n左右对称性分析:")
                print(f"  左侧平均误差: {left_avg:.1f}px")
                print(f"  右侧平均误差: {right_avg:.1f}px")
                print(f"  左右误差差异: {abs(left_avg - right_avg):.1f}px")
        
        print("=" * 30) 