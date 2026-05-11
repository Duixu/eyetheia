import cv2
import os
import numpy as np
import torch
from typing import Optional, Dict, Any
from .face_landmark_pose import FaceLandmarkPose
from .gaze_model import GazeEstimationModel
from .utils.head_pose_normalizer import HeadPoseNormalizer
from .utils.gaze_math import vector_to_angle
from .camera_config import CAMERA_MATRIX, DIST_COEFFS, NORMALIZATION_CAMERA_MATRIX

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class GazeEstimator:
    """
    视线估计器 - 必须提供 landmarks 输入
    
    这个类专门用于视线估计，不进行人脸检测。
    调用者必须提供已经检测到的人脸关键点。
    """
    
    def __init__(self, gaze_model_type='swin', device='cpu', weight_path: str | None = None):
        """
        初始化视线估计器
        
        Args:
            gaze_model_type: 视线模型类型 ('swin')
            device: 计算设备 ('cpu', 'cuda')
        """
        self.device = torch.device(device)
        self.gaze_model = GazeEstimationModel(
            gaze_model_type,
            device=self.device,
            weight_path=weight_path,
        )
        # 加载模型权重
        try:
            self.gaze_model.load_weights()
        except Exception as e:
            raise RuntimeError(f"视线模型权重加载失败: {e}")
        
        self.normalizer = HeadPoseNormalizer(CAMERA_MATRIX, NORMALIZATION_CAMERA_MATRIX)
        self.face_model_points_3d = FaceLandmarkPose.get_3d_landmarks('mediapipe')
        
        # 不再需要创建姿态检测器实例，使用静态方法
        # self.pose_detector = FaceLandmarkPose()
        # self.pose_detector.set_pose_params(self.face_model_points_3d, CAMERA_MATRIX, DIST_COEFFS)
        
        # 验证3D模型点
        if self.face_model_points_3d is None:
            raise RuntimeError("3D人脸模型点加载失败")
    
    def forward(self, image: np.ndarray, face_landmarks_2d: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        估计视线方向
        
        Args:
            image: 输入图像 (BGR格式)
            face_landmarks_2d: 人脸关键点数组，必须提供，不能为None
            
        Returns:
            视线估计结果字典，如果失败返回None
            
        Raises:
            ValueError: 当 face_landmarks 为 None 或格式不正确时
            RuntimeError: 当视线估计过程中出现错误时
        """
        # 严格验证输入参数
        if face_landmarks_2d is None:
            raise ValueError("face_landmarks 不能为 None，必须提供人脸关键点")
        
        if not isinstance(face_landmarks_2d, np.ndarray):
            raise ValueError("face_landmarks 必须是 numpy.ndarray 类型")
        
        if len(face_landmarks_2d.shape) != 2 or face_landmarks_2d.shape[1] != 2:
            raise ValueError("face_landmarks 必须是形状为 (N, 2) 的数组")
        
        if len(face_landmarks_2d) < len(self.face_model_points_3d):
            raise ValueError(f"face_landmarks 数量不足，需要至少 {len(self.face_model_points_3d)} 个点，当前只有 {len(face_landmarks_2d)} 个")
        
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image 不能为 None 且必须是 numpy.ndarray 类型")
        
        try:
            # 1. 姿态估计
            image_points_2d = face_landmarks_2d[:len(self.face_model_points_3d)]
            rot, tvec = self._estimate_head_pose(image_points_2d)
            
            if rot is None or tvec is None:
                raise RuntimeError("头部姿态估计失败")
            
            # 2. 3D点变换
            face_landmarks_3d = FaceLandmarkPose.compute_3d_pose_respose_camera(
                self.face_model_points_3d, rot, tvec.ravel()
            )
            
            # 3. 计算中心点
            center, _, _ = FaceLandmarkPose.compute_face_eye_centers_respose_camera(face_landmarks_3d)
            
            # 4. 归一化
            norm_result = self.normalizer.normalize(image, center, rot)
            normalized_image = norm_result['normalized_image']
            normalizing_rot = norm_result['normalizing_rotation']
            
            # 5. 视线模型推理
            gaze_prediction = self._infer_gaze_model(normalized_image)
            
            if gaze_prediction is None:
                raise RuntimeError("视线模型推理失败")
            
            pitch, yaw = gaze_prediction[0], gaze_prediction[1]
            
            # 6. 角度转向量（归一化空间下）
            gaze_vector_norm = self._angles_to_vector(pitch, yaw)
            
            # 7. 反归一化到真实相机坐标系
            gaze_vector_real = gaze_vector_norm @ normalizing_rot.as_matrix()
            
            # 8. 计算真实角度(弧度)
            real_pitch, real_yaw = vector_to_angle(gaze_vector_real)
            
            return {
                'pitch': real_pitch, 
                'yaw': real_yaw, 
                'gaze_vector': gaze_vector_real,
                'head_pose': {
                    'rotation': rot,
                    'translation': tvec
                },
                'face_center': center,
                'model3d': face_landmarks_3d,
                'landmarks': face_landmarks_2d,
                'confidence': self._calculate_confidence(real_pitch, real_yaw, gaze_vector_real)
            }
            
        except Exception as e:
            raise RuntimeError(f"视线估计失败: {e}")
    
    def _estimate_head_pose(self, image_points_2d: np.ndarray) -> tuple:
        """
        估计头部姿态
        
        Args:
            image_points_2d: 2D图像点
            
        Returns:
            (rotation, translation) 元组
        """
        try:
            # 使用静态方法，避免实例化开销
            rot, tvec = FaceLandmarkPose.estimate_head_pose_static(
                image_points_2d, 
                self.face_model_points_3d, 
                CAMERA_MATRIX, 
                DIST_COEFFS
            )
            return rot, tvec
        except Exception as e:
            raise RuntimeError(f"头部姿态估计失败: {e}")
        
    def _infer_gaze_model(self, normalized_image: np.ndarray) -> np.ndarray:
        """
        PyTorch CPU forward
        """
        import torch
        from torchvision import transforms

        preprocess = transforms.Compose([transforms.ToTensor()])
        input_tensor = preprocess(normalized_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.gaze_model.forward(input_tensor)  # PyTorch 模型
            prediction = prediction.cpu().numpy()[0]

        return prediction

    def _infer_gaze_model_trt(self, normalized_image: np.ndarray) -> Optional[tuple]:
        """
        使用 TensorRT 视线模型进行推理
        """
        print("开始forward推理")
        result = self.gaze_model.forward(normalized_image)
        print("forward完成")
        if result is not None:
            # 转换为 numpy 数组
            if hasattr(result, 'cpu'):
                result = result.cpu().numpy().reshape(2)
            elif hasattr(result, 'numpy'):
                result = result.numpy().reshape(2)
            # 转换为元组
            if isinstance(result, np.ndarray):
                return tuple(result.tolist())
            elif isinstance(result, (list, tuple)):
                return tuple(result)
            else:
                return (float(result),)
        return None

    def _infer_gaze_model(self, normalized_image: np.ndarray) -> Optional[np.ndarray]:
        """
        使用视线模型进行推理
        
        Args:
            normalized_image: 归一化后的图像
            
        Returns:
            预测结果数组
        """
        try:
            if normalized_image.ndim != 3 or normalized_image.shape[2] != 3:
                raise ValueError(f"expected normalized HWC/BGR image, got {normalized_image.shape}")

            image_rgb = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
            image_rgb = image_rgb.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.to(self.device, dtype=torch.float32)

            with torch.no_grad():
                prediction = self.gaze_model.forward(input_tensor)
                prediction = prediction.detach().cpu().numpy().reshape(-1)

            return prediction[:2]
            
        except Exception as e:
            raise RuntimeError(f"视线模型推理失败: {e}")
    
    def _angles_to_vector(self, pitch: float, yaw: float) -> np.ndarray:
        """
        将角度转换为视线向量
        
        Args:
            pitch: 俯仰角
            yaw: 偏航角
            
        Returns:
            视线向量
        """
        return -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])
    
    def _calculate_confidence(self, pitch: float, yaw: float, gaze_vector: np.ndarray) -> float:
        """
        计算视线估计的置信度
        
        Args:
            pitch: 俯仰角
            yaw: 偏航角
            gaze_vector: 视线向量
            
        Returns:
            置信度分数 (0.0 - 1.0)
        """
        try:
            # 基于向量长度的置信度
            vector_length = np.linalg.norm(gaze_vector)
            length_confidence = min(vector_length, 1.0)
            
            # 基于角度范围的置信度
            pitch_confidence = 1.0 - min(abs(pitch) / np.pi, 1.0)
            yaw_confidence = 1.0 - min(abs(yaw) / np.pi, 1.0)
            
            # 综合置信度
            overall_confidence = (length_confidence + pitch_confidence + yaw_confidence) / 3.0
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception:
            return 0.5  # 默认中等置信度
    
    def get_required_landmarks_count(self) -> int:
        """
        获取所需的关键点数量
        
        Returns:
            所需的关键点数量
        """
        return len(self.face_model_points_3d)
    
    def is_landmarks_valid(self, landmarks: np.ndarray) -> bool:
        """
        验证关键点是否有效
        
        Args:
            landmarks: 关键点数组
            
        Returns:
            是否有效
        """
        try:
            if landmarks is None:
                return False
            
            if not isinstance(landmarks, np.ndarray):
                return False
            
            if len(landmarks.shape) != 2 or landmarks.shape[1] != 2:
                return False
            
            if len(landmarks) < self.get_required_landmarks_count():
                return False
            
            return True
            
        except Exception:
            return False 
