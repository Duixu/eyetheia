#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC标定点扩展器
通过修改标定点坐标来扩大视线覆盖范围，避免屏幕边缘空隙
"""

import numpy as np
import yaml
import os
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

class ARCCalibrationExpander:
    """ARC标定点扩展器"""
    
    def __init__(self, calibration_file='arc_calibration.yaml'):
        """初始化扩展器"""
        self.calibration_file = calibration_file
        self.calibration_data = None
        self.load_calibration()
        
    def load_calibration(self):
        """加载标定数据"""
        if not os.path.exists(self.calibration_file):
            print(f"标定文件不存在: {self.calibration_file}")
            return False
            
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = yaml.safe_load(f)
            print(f"已加载标定数据: {self.calibration_file}")
            return True
        except Exception as e:
            print(f"加载标定数据失败: {e}")
            return False
    
    def analyze_coverage(self):
        """分析当前视线覆盖范围"""
        if not self.calibration_data:
            print("没有标定数据")
            return
            
        calibration_points = np.array(self.calibration_data['calibration_points'])
        final_pos = np.array(self.calibration_data.get('final_pos', []))
        
        screen_width = self.calibration_data['screen_width']
        screen_height = self.calibration_data['screen_height']
        
        print("=== 视线覆盖范围分析 ===")
        print(f"屏幕分辨率: {screen_width} x {screen_height}")
        print(f"标定点数量: {len(calibration_points)}")
        
        # 分析理论标定点覆盖
        print("\n理论标定点覆盖:")
        x_coords = calibration_points[:, 0]
        y_coords = calibration_points[:, 1]
        
        print(f"X轴覆盖: {np.min(x_coords):.0f} - {np.max(x_coords):.0f} (范围: {np.max(x_coords) - np.min(x_coords):.0f}px)")
        print(f"Y轴覆盖: {np.min(y_coords):.0f} - {np.max(y_coords):.0f} (范围: {np.max(y_coords) - np.min(y_coords):.0f}px)")
        
        # 分析实际视线覆盖
        if len(final_pos) > 0:
            print("\n实际视线覆盖:")
            final_x = final_pos[:, 0]
            final_y = final_pos[:, 1]
            
            print(f"X轴覆盖: {np.min(final_x):.0f} - {np.max(final_x):.0f} (范围: {np.max(final_x) - np.min(final_x):.0f}px)")
            print(f"Y轴覆盖: {np.min(final_y):.0f} - {np.max(final_y):.0f} (范围: {np.max(final_y) - np.min(final_y):.0f}px)")
            
            # 计算覆盖率
            x_coverage = (np.max(final_x) - np.min(final_x)) / screen_width * 100
            y_coverage = (np.max(final_y) - np.min(final_y)) / screen_height * 100
            
            print(f"X轴覆盖率: {x_coverage:.1f}%")
            print(f"Y轴覆盖率: {y_coverage:.1f}%")
            
            # 分析边缘空隙
            edge_margin = 50  # 边缘边距
            left_gap = np.min(final_x) - edge_margin
            right_gap = screen_width - np.max(final_x) - edge_margin
            top_gap = np.min(final_y) - edge_margin
            bottom_gap = screen_height - np.max(final_y) - edge_margin
            
            print(f"\n边缘空隙分析:")
            print(f"左侧空隙: {left_gap:.0f}px")
            print(f"右侧空隙: {right_gap:.0f}px")
            print(f"顶部空隙: {top_gap:.0f}px")
            print(f"底部空隙: {bottom_gap:.0f}px")
            
            return {
                'screen_width': screen_width,
                'screen_height': screen_height,
                'calibration_points': calibration_points,
                'final_pos': final_pos,
                'coverage': {'x': x_coverage, 'y': y_coverage},
                'gaps': {'left': left_gap, 'right': right_gap, 'top': top_gap, 'bottom': bottom_gap}
            }
        
        return None
    
    def expand_calibration_points(self, expansion_factor=1.2, edge_margin=30):
        """扩展标定点坐标以扩大覆盖范围"""
        if not self.calibration_data:
            print("没有标定数据")
            return None
            
        calibration_points = np.array(self.calibration_data['calibration_points'])
        screen_width = self.calibration_data['screen_width']
        screen_height = self.calibration_data['screen_height']
        
        print(f"\n=== 扩展标定点坐标 ===")
        print(f"扩展因子: {expansion_factor}")
        print(f"边缘边距: {edge_margin}px")
        
        # 计算当前标定点的中心
        center_x = np.mean(calibration_points[:, 0])
        center_y = np.mean(calibration_points[:, 1])
        
        print(f"当前标定点中心: ({center_x:.0f}, {center_y:.0f})")
        
        # 扩展标定点坐标
        expanded_points = []
        for point in calibration_points:
            # 计算相对于中心的偏移
            offset_x = point[0] - center_x
            offset_y = point[1] - center_y
            
            # 应用扩展因子
            expanded_x = center_x + offset_x * expansion_factor
            expanded_y = center_y + offset_y * expansion_factor
            
            # 确保不超出屏幕边界
            expanded_x = np.clip(expanded_x, edge_margin, screen_width - edge_margin)
            expanded_y = np.clip(expanded_y, edge_margin, screen_height - edge_margin)
            
            expanded_points.append([expanded_x, expanded_y])
        
        expanded_points = np.array(expanded_points)
        
        # 分析扩展效果
        print(f"\n扩展效果分析:")
        print(f"原始X轴范围: {np.min(calibration_points[:, 0]):.0f} - {np.max(calibration_points[:, 0]):.0f}")
        print(f"扩展X轴范围: {np.min(expanded_points[:, 0]):.0f} - {np.max(expanded_points[:, 0]):.0f}")
        print(f"原始Y轴范围: {np.min(calibration_points[:, 1]):.0f} - {np.max(calibration_points[:, 1]):.0f}")
        print(f"扩展Y轴范围: {np.min(expanded_points[:, 1]):.0f} - {np.max(expanded_points[:, 1]):.0f}")
        
        return expanded_points
    
    def optimize_expansion_parameters(self):
        """优化扩展参数以最大化覆盖范围"""
        if not self.calibration_data:
            print("没有标定数据")
            return None
            
        calibration_points = np.array(self.calibration_data['calibration_points'])
        final_pos = np.array(self.calibration_data.get('final_pos', []))
        screen_width = self.calibration_data['screen_width']
        screen_height = self.calibration_data['screen_height']
        
        if len(final_pos) == 0:
            print("没有final_pos数据，无法优化")
            return None
        
        print("\n=== 优化扩展参数 ===")
        
        def objective_function(params):
            """优化目标函数：最大化覆盖范围"""
            expansion_factor, edge_margin = params
            
            # 扩展标定点
            center_x = np.mean(calibration_points[:, 0])
            center_y = np.mean(calibration_points[:, 1])
            
            expanded_points = []
            for point in calibration_points:
                offset_x = point[0] - center_x
                offset_y = point[1] - center_y
                expanded_x = center_x + offset_x * expansion_factor
                expanded_y = center_y + offset_y * expansion_factor
                expanded_x = np.clip(expanded_x, edge_margin, screen_width - edge_margin)
                expanded_y = np.clip(expanded_y, edge_margin, screen_height - edge_margin)
                expanded_points.append([expanded_x, expanded_y])
            
            expanded_points = np.array(expanded_points)
            
            # 计算覆盖范围
            x_range = np.max(expanded_points[:, 0]) - np.min(expanded_points[:, 0])
            y_range = np.max(expanded_points[:, 1]) - np.min(expanded_points[:, 1])
            
            # 计算覆盖率
            x_coverage = x_range / screen_width
            y_coverage = y_range / screen_height
            
            # 目标：最大化覆盖率，同时保持合理的边距
            coverage_score = x_coverage + y_coverage
            margin_penalty = max(0, 100 - edge_margin) / 100  # 边距惩罚
            
            return -(coverage_score - margin_penalty * 0.1)
        
        # 优化参数
        initial_params = [1.2, 30]  # 初始扩展因子和边距
        bounds = [(1.0, 2.0), (10, 100)]  # 参数范围
        
        result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            optimal_expansion, optimal_margin = result.x
            print(f"优化成功!")
            print(f"最优扩展因子: {optimal_expansion:.3f}")
            print(f"最优边缘边距: {optimal_margin:.1f}px")
            
            return optimal_expansion, optimal_margin
        else:
            print("优化失败")
            return None
    
    def generate_virtual_calibration_points(self, num_points=13):
        """生成虚拟标定点以增加覆盖密度"""
        if not self.calibration_data:
            print("没有标定数据")
            return None
            
        screen_width = self.calibration_data['screen_width']
        screen_height = self.calibration_data['screen_height']
        
        print(f"\n=== 生成虚拟标定点 ===")
        print(f"目标点数: {num_points}")
        
        # 生成网格化的标定点
        if num_points == 13:
            # 13点模式：3x3网格 + 4个角落点
            points = []
            
            # 3x3网格
            for i in range(3):
                for j in range(3):
                    x = (i + 1) * screen_width / 4
                    y = (j + 1) * screen_height / 4
                    points.append([x, y])
            
            # 4个角落点
            margin = 50
            points.extend([
                [margin, margin],  # 左上
                [screen_width - margin, margin],  # 右上
                [margin, screen_height - margin],  # 左下
                [screen_width - margin, screen_height - margin]  # 右下
            ])
            
        elif num_points == 16:
            # 16点模式：4x4网格
            points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) * screen_width / 4
                    y = (j + 0.5) * screen_height / 4
                    points.append([x, y])
        
        elif num_points == 9:
            # 9点模式：3x3网格
            points = []
            for i in range(3):
                for j in range(3):
                    x = (i + 1) * screen_width / 4
                    y = (j + 1) * screen_height / 4
                    points.append([x, y])
        
        else:
            # 自定义点数：使用Halton序列生成均匀分布的点
            points = self._generate_halton_points(num_points, screen_width, screen_height)
        
        virtual_points = np.array(points)
        
        print(f"生成的虚拟标定点:")
        for i, point in enumerate(virtual_points):
            print(f"点 {i+1}: ({point[0]:.0f}, {point[1]:.0f})")
        
        return virtual_points
    
    def _generate_halton_points(self, num_points, width, height):
        """使用Halton序列生成均匀分布的点"""
        def halton_sequence(n, base):
            result = 0
            f = 1
            while n > 0:
                f = f / base
                result = result + f * (n % base)
                n = n // base
            return result
        
        points = []
        margin = 50
        
        for i in range(num_points):
            x = margin + (width - 2 * margin) * halton_sequence(i + 1, 2)
            y = margin + (height - 2 * margin) * halton_sequence(i + 1, 3)
            points.append([x, y])
        
        return points
    
    def create_expanded_calibration(self, expansion_factor=1.2, edge_margin=30, 
                                   use_virtual_points=False, num_virtual_points=13):
        """创建扩展后的标定配置"""
        if not self.calibration_data:
            print("没有标定数据")
            return None
        
        print(f"\n=== 创建扩展标定配置 ===")
        
        # 获取原始数据
        original_data = self.calibration_data.copy()
        
        if use_virtual_points:
            # 使用虚拟标定点
            new_calibration_points = self.generate_virtual_calibration_points(num_virtual_points)
            print(f"使用虚拟标定点: {len(new_calibration_points)} 个点")
        else:
            # 扩展原始标定点
            new_calibration_points = self.expand_calibration_points(expansion_factor, edge_margin)
            print(f"扩展原始标定点: {len(new_calibration_points)} 个点")
        
        if new_calibration_points is None:
            return None
        
        # 创建新的标定数据
        expanded_data = original_data.copy()
        expanded_data['calibration_points'] = new_calibration_points.tolist()
        expanded_data['original_calibration_points'] = original_data['calibration_points']
        expanded_data['expansion_info'] = {
            'expansion_factor': expansion_factor,
            'edge_margin': edge_margin,
            'use_virtual_points': use_virtual_points,
            'num_virtual_points': num_virtual_points if use_virtual_points else len(new_calibration_points)
        }
        
        return expanded_data
    
    def save_expanded_calibration(self, expanded_data, filename='arc_calibration_expanded.yaml'):
        """保存扩展后的标定配置"""
        try:
            with open(filename, 'w') as f:
                yaml.dump(expanded_data, f, default_flow_style=False)
            print(f"扩展标定配置已保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存扩展标定配置失败: {e}")
            return False
    
    def visualize_expansion(self, original_points, expanded_points, screen_width, screen_height):
        """可视化扩展效果"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制屏幕边界
            screen_rect = plt.Rectangle((0, 0), screen_width, screen_height, 
                                      fill=False, color='black', linewidth=2, label='屏幕边界')
            plt.gca().add_patch(screen_rect)
            
            # 绘制原始标定点
            plt.scatter(original_points[:, 0], original_points[:, 1], 
                       c='red', s=100, marker='o', label='原始标定点', alpha=0.7)
            
            # 绘制扩展标定点
            plt.scatter(expanded_points[:, 0], expanded_points[:, 1], 
                       c='blue', s=100, marker='s', label='扩展标定点', alpha=0.7)
            
            # 连接对应的点
            for i in range(len(original_points)):
                plt.plot([original_points[i, 0], expanded_points[i, 0]], 
                        [original_points[i, 1], expanded_points[i, 1]], 
                        'g--', alpha=0.5)
            
            plt.xlabel('X坐标 (px)')
            plt.ylabel('Y坐标 (px)')
            plt.title('标定点扩展效果')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.savefig('calibration_expansion_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("扩展效果可视化已保存到: calibration_expansion_visualization.png")
            
        except ImportError:
            print("matplotlib未安装，跳过可视化")

def main():
    """主函数"""
    print("=== ARC标定点扩展器 ===")
    
    # 创建扩展器
    expander = ARCCalibrationExpander()
    
    # 分析当前覆盖范围
    coverage_info = expander.analyze_coverage()
    if coverage_info is None:
        return
    
    # 优化扩展参数
    optimal_params = expander.optimize_expansion_parameters()
    
    if optimal_params:
        optimal_expansion, optimal_margin = optimal_params
        
        # 创建扩展标定配置
        expanded_data = expander.create_expanded_calibration(
            expansion_factor=optimal_expansion,
            edge_margin=optimal_margin,
            use_virtual_points=False
        )
        
        if expanded_data:
            # 保存扩展配置
            expander.save_expanded_calibration(expanded_data)
            
            # 可视化扩展效果
            original_points = np.array(expanded_data['original_calibration_points'])
            expanded_points = np.array(expanded_data['calibration_points'])
            screen_width = expanded_data['screen_width']
            screen_height = expanded_data['screen_height']
            
            expander.visualize_expansion(original_points, expanded_points, screen_width, screen_height)
    
    # 生成虚拟标定点选项
    print("\n=== 生成虚拟标定点 ===")
    virtual_data = expander.create_expanded_calibration(
        use_virtual_points=True,
        num_virtual_points=13
    )
    
    if virtual_data:
        expander.save_expanded_calibration(virtual_data, 'arc_calibration_virtual_13points.yaml')
        
        # 可视化虚拟标定点
        original_points = np.array(virtual_data['original_calibration_points'])
        virtual_points = np.array(virtual_data['calibration_points'])
        screen_width = virtual_data['screen_width']
        screen_height = virtual_data['screen_height']
        
        expander.visualize_expansion(original_points, virtual_points, screen_width, screen_height)
    
    print("\n=== 扩展完成 ===")
    print("生成的文件:")
    print("- arc_calibration_expanded.yaml: 扩展后的标定配置")
    print("- arc_calibration_virtual_13points.yaml: 13点虚拟标定配置")
    print("- calibration_expansion_visualization.png: 扩展效果可视化")

if __name__ == "__main__":
    main() 