#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机内参标定脚本
使用棋盘格标定板来确定摄像头的内参矩阵和畸变系数
"""

import cv2
import numpy as np
import os
import json
import argparse
from datetime import datetime

class CameraIntrinsicCalibrator:
    def __init__(self, camera_index=0, chessboard_size=(8, 5), square_size=1):
        """
        初始化相机内参标定器
        
        Args:
            camera_index: 摄像头索引
            chessboard_size: 棋盘格内角点数量 (width, height)
            square_size: 棋盘格方格大小（米）
        """
        self.camera_index = camera_index
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 标定数据
        self.object_points = []  # 3D点
        self.image_points = []   # 2D点
        self.calibration_images = []
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
        # 标定质量指标
        self.reprojection_error = None
        
        # 初始化摄像头
        self.cap = self._init_camera()
        
        # 创建显示窗口
        cv2.namedWindow('Camera Calibration', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Camera Calibration', 1280, 720)
        
        print(f"相机内参标定器初始化完成")
        print(f"棋盘格大小: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
        print(f"方格大小: {square_size}m")
        print(f"摄像头索引: {camera_index}")
    
    def _init_camera(self):
        """初始化摄像头"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

        # 获取实际分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {width}x{height}")
        
        return cap
    
    def _create_object_points(self):
        """创建棋盘格的3D点坐标"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def capture_calibration_images(self, target_count=15):
        """
        采集标定图像
        
        Args:
            target_count: 目标采集图像数量
        """
        print(f"\n=== 开始采集标定图像 ===")
        print(f"目标采集数量: {target_count} (建议10-20张)")
        print("请将棋盘格标定板放在摄像头前，按以下操作：")
        print("- 按 'c' 键：捕获当前帧（如果检测到棋盘格）")
        print("- 按 's' 键：跳过当前帧")
        print("- 按 'q' 键：退出采集")
        print("- 按 'r' 键：重新开始采集")
        print("提示：请在不同角度和距离下采集图像以获得更好的标定效果")
        print()
        
        captured_count = 0
        frame_count = 0
        #cap = cv2.VideoCapture(0)
        while captured_count < target_count:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                continue

            frame_count += 1
            
            # 检测棋盘格
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_chess, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # 绘制检测结果
            display_frame = frame.copy()
            
            if ret_chess:
                # 绘制棋盘格角点
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret_chess)
                
                # 显示提示信息
                cv2.putText(display_frame, f"棋盘格已检测到 - 按 'c' 捕获", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"已采集: {captured_count}/{target_count}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # 显示提示信息
                cv2.putText(display_frame, "未检测到棋盘格 - 请调整位置", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, f"已采集: {captured_count}/{target_count}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 显示操作提示
            cv2.putText(display_frame, "c: 捕获 | s: 跳过 | q: 退出 | r: 重新开始", 
                       (50, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and ret_chess:
                # 捕获图像
                self.calibration_images.append(frame.copy())
                self.image_points.append(corners)
                captured_count += 1
                print(f"已捕获图像 {captured_count}/{target_count}")
            elif key == ord('s'):
                # 跳过当前帧
                print(f"跳过当前帧")
            elif key == ord('r'):
                # 重新开始
                self.calibration_images.clear()
                self.image_points.clear()
                captured_count = 0
                print("重新开始采集")
        
        print(f"\n采集完成，共捕获 {len(self.calibration_images)} 张图像")
        
        if len(self.calibration_images) > 0:
            return True
        else:
            print("未捕获任何图像")
            return False
    
    def calibrate_camera(self):
        """执行相机标定"""
        if len(self.calibration_images) < 5:
            print("标定图像数量不足，至少需要5张图像")
            return False
        
        print(f"\n=== 开始相机标定 ===")
        print(f"使用 {len(self.calibration_images)} 张标定图像")
        
        # 准备3D点
        objp = self._create_object_points()
        self.object_points = [objp] * len(self.calibration_images)
        
        # 获取图像尺寸
        height, width = self.calibration_images[0].shape[:2]
        
        # 执行标定
        print("计算相机内参...")
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, (width, height), None, None
        )
        
        if ret:
            print("相机标定成功！")
            
            # 计算重投影误差
            self._calculate_reprojection_error()
            
            # 显示标定结果
            self._display_calibration_results()
            
            return True
        else:
            print("相机标定失败")
            return False
    
    def _calculate_reprojection_error(self):
        """计算重投影误差"""
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i], self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        self.reprojection_error = total_error / len(self.object_points)
        print(f"平均重投影误差: {self.reprojection_error:.4f} 像素")
    
    def _display_calibration_results(self):
        """显示标定结果"""
        print("\n=== 标定结果 ===")
        print("相机内参矩阵:")
        print(self.camera_matrix)
        print("\n畸变系数:")
        print(self.dist_coeffs)
        print(f"\n重投影误差: {self.reprojection_error:.4f} 像素")
        
        # 计算焦距（像素）
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        print(f"\n焦距: fx={fx:.2f}, fy={fy:.2f} 像素")
        print(f"主点: cx={cx:.2f}, cy={cy:.2f} 像素")
    
    def save_calibration(self, filename=None):
        """保存标定结果"""
        if self.camera_matrix is None:
            print("没有标定结果可保存")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_calibration_{timestamp}.json"
            print(f"自动保存到: {filename}")
        
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'reprojection_error': self.reprojection_error,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size,
            'image_count': len(self.calibration_images),
            'calibration_date': datetime.now().isoformat(),
            'camera_index': self.camera_index
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"标定结果已保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存标定结果失败: {e}")
            return False
    
    def save_camera_config_py(self, filename=None):
        """按照camera_config.py格式保存相机参数"""
        if self.camera_matrix is None:
            print("没有标定结果可保存")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_config_{timestamp}.py"
            print(f"自动保存到: {filename}")
        
        # 计算归一化相机矩阵（假设图像尺寸为224x224）
        image_height, image_width = self.calibration_images[0].shape[:2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 归一化到224x224
        scale_x = 224.0 / image_width
        scale_y = 224.0 / image_height
        normalized_fx = fx * scale_x
        normalized_fy = fy * scale_y
        normalized_cx = cx * scale_x
        normalized_cy = cy * scale_y
        
        config_content = f'''# -*- coding: utf-8 -*-
"""
相机参数配置
标定时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
标定图像数量: {len(self.calibration_images)}
重投影误差: {self.reprojection_error:.6f}
棋盘格大小: {self.chessboard_size[0]}x{self.chessboard_size[1]}
方格大小: {self.square_size}m
"""
import numpy as np

# 相机内参矩阵
CAMERA_MATRIX = np.array([
    [{self.camera_matrix[0, 0]:.10f}, {self.camera_matrix[0, 1]:.10f}, {self.camera_matrix[0, 2]:.10f}],
    [{self.camera_matrix[1, 0]:.10f}, {self.camera_matrix[1, 1]:.10f}, {self.camera_matrix[1, 2]:.10f}],
    [{self.camera_matrix[2, 0]:.10f}, {self.camera_matrix[2, 1]:.10f}, {self.camera_matrix[2, 2]:.10f}]
], dtype=np.float64)

# 畸变系数
DIST_COEFFS = np.array(({self.dist_coeffs[0, 0]:.10f}, {self.dist_coeffs[0, 1]:.10f}, {self.dist_coeffs[0, 2]:.10f}, {self.dist_coeffs[0, 3]:.10f}, {self.dist_coeffs[0, 4]:.10f}), dtype=np.float64)

# 归一化相机矩阵 (224x224)
NORMALIZATION_CAMERA_MATRIX = np.array([
    [{normalized_fx:.10f}, 0., {normalized_cx:.10f}],
    [0., {normalized_fy:.10f}, {normalized_cy:.10f}],
    [0., 0., 1.]
], dtype=np.float64)

# 原始图像尺寸
IMAGE_WIDTH = {image_width}
IMAGE_HEIGHT = {image_height}

# 归一化图像尺寸
NORMALIZED_WIDTH = 224
NORMALIZED_HEIGHT = 224
'''
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print(f"相机配置文件已保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存相机配置文件失败: {e}")
            return False
    
    def load_calibration(self, filename):
        """加载标定结果"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.camera_matrix = np.array(calibration_data['camera_matrix'])
            self.dist_coeffs = np.array(calibration_data['dist_coeffs'])
            self.reprojection_error = calibration_data['reprojection_error']
            
            print(f"已加载标定结果: {filename}")
            self._display_calibration_results()
            return True
        except Exception as e:
            print(f"加载标定结果失败: {e}")
            return False
    
    def test_undistortion(self):
        """测试去畸变效果"""
        if self.camera_matrix is None:
            print("没有标定结果，无法测试去畸变")
            return
        
        print("测试去畸变效果...")
        print("左侧：原始图像，右侧：去畸变后图像")
        print("按任意键退出测试")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # 去畸变
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # 显示对比
            combined = np.hstack([frame, undistorted])
            cv2.putText(combined, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Undistorted", (frame.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Original vs Undistorted', combined)
            
            if cv2.waitKey(1) & 0xFF != 255:
                break
        
        cv2.destroyWindow('Original vs Undistorted')
    
    def run(self, save_config=None, overwrite_config=False):
        """运行标定程序"""
        if self.cap is None:
            print("摄像头初始化失败")
            return
        
        try:
            print("=== 相机内参标定程序 ===")
            print("程序将自动执行以下步骤：")
            print("1. 采集标定图像")
            print("2. 执行标定")
            print("3. 自动保存结果")
            print("4. 测试去畸变效果")
            print()
            
            # 自动执行标定流程
            print("开始自动标定流程...")
            
            # 步骤1: 采集标定图像
            print("\n=== 步骤1: 采集标定图像 ===")
            if self.capture_calibration_images():
                print("✓ 图像采集完成")
            else:
                print("✗ 图像采集失败，程序退出")
                return
            
            # 步骤2: 执行标定
            print("\n=== 步骤2: 执行标定 ===")
            if self.calibrate_camera():
                print("✓ 标定完成")
            else:
                print("✗ 标定失败，程序退出")
                return
            
            # 步骤3: 自动保存结果
            print("\n=== 步骤3: 自动保存结果 ===")
            # 保存JSON格式
            if self.save_calibration():
                print("✓ JSON格式标定结果已保存")
            else:
                print("✗ JSON格式保存失败")
            
            # 保存Python配置文件格式
            if self.save_camera_config_py(save_config):
                print("✓ Python配置文件格式已保存")
            else:
                print("✗ Python配置文件保存失败")
            
            # 如果指定了覆盖选项，同时保存到config目录
            if overwrite_config:
                config_path = os.path.join('config', 'camera_config.py')
                if self.save_camera_config_py(config_path):
                    print(f"✓ 已覆盖配置文件: {config_path}")
                else:
                    print(f"✗ 覆盖配置文件失败: {config_path}")
            
            # 步骤4: 测试去畸变
            print("\n=== 步骤4: 测试去畸变效果 ===")
            print("按任意键退出测试")
            self.test_undistortion()
            
            print("\n=== 标定流程完成 ===")
            print("标定结果已保存，可以用于后续的相机应用")
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='相机内参标定程序', 
                                   formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog="""
使用示例:
  # 基本标定
  python camera_intrinsic_calibration.py
  
  # 指定摄像头和棋盘格参数
  python camera_intrinsic_calibration.py --camera 1 --chessboard 9,6 --square 0.025
  
  # 保存到指定配置文件
  python camera_intrinsic_calibration.py --save-config my_camera_config.py
  
  # 覆盖现有配置文件
  python camera_intrinsic_calibration.py --overwrite-config
  
  # 加载已有标定结果进行测试
  python camera_intrinsic_calibration.py --load calibration_result.json
""")
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引 (默认: 0)')
    parser.add_argument('--chessboard', type=str, default='8,5', help='棋盘格内角点数量 (默认: 8,5)')
    parser.add_argument('--square', type=float, default=0.025, help='棋盘格方格大小，单位米 (默认: 0.025)')
    parser.add_argument('--load', type=str, help='加载已有标定结果文件')
    parser.add_argument('--save-config', type=str, help='保存相机配置文件的路径 (默认自动生成)')
    parser.add_argument('--overwrite-config', action='store_true', help='覆盖现有的camera_config.py文件')
    
    args = parser.parse_args()
    
    # 解析棋盘格大小
    chessboard_size = tuple(map(int, args.chessboard.split(',')))
    
    try:
        calibrator = CameraIntrinsicCalibrator(
            camera_index=args.camera,
            chessboard_size=chessboard_size,
            square_size=args.square
        )
        
        # 如果指定了加载文件
        if args.load:
            if calibrator.load_calibration(args.load):
                calibrator.test_undistortion()
            else:
                print("加载标定结果失败")
        else:
            calibrator.run(save_config=args.save_config, overwrite_config=args.overwrite_config)
            
    except Exception as e:
        print(f"程序启动失败: {e}")

if __name__ == "__main__":

    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     cv2.imshow('Camera Calibration', frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    main()