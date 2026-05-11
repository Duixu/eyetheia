# -*- coding: utf-8 -*-
"""
基础模型抽象类
参考deepface的设计模式
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from loguru import logger

warnings.filterwarnings("ignore")


class BaseModel(ABC):
    """基础模型抽象类，参考deepface的设计模式"""
    
    def __init__(self, model_name: str, task: str):
        self.model_name = model_name
        self.task = task
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False
        
    @abstractmethod
    def build_model(self) -> Any:
        """构建模型"""
        pass
    
    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """前向推理"""
        pass
    
    def load_weights(self, weight_path: str) -> bool:
        """加载模型权重"""
        try:
            if self.model is None:
                self.model = self.build_model()
            
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.is_initialized = True
                logger.info(f"成功加载模型权重: {weight_path}")
                return True
            else:
                logger.warning(f"模型权重文件不存在: {weight_path}")
                return False
        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            return False
    
    def to_device(self, device: str = None):
        """将模型移动到指定设备"""
        if device:
            self.device = torch.device(device)
        
        if self.model is not None:
            self.model = self.model.to(self.device)
    
    def set_eval_mode(self):
        """设置为评估模式"""
        if self.model is not None:
            self.model.eval()
    
    def set_train_mode(self):
        """设置为训练模式"""
        if self.model is not None:
            self.model.train()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'task': self.task,
            'device': str(self.device),
            'is_initialized': self.is_initialized,
            'model_type': type(self.model).__name__ if self.model else None,
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"模型已保存到: {save_path}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', task='{self.task}')" 