import cv2
import numpy as np
from scipy.spatial.transform import Rotation

'''
TODO: head_pose_rot是矩阵还是向量?  要统一下 下面这个是旋转向量
'''

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

class HeadPoseNormalizer:
    def __init__(self, camera_matrix, normalized_camera_matrix, normalized_distance=0.6):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        self.normalized_camera_matrix = np.asarray(normalized_camera_matrix, dtype=np.float32)
        self.normalized_distance = normalized_distance

    def normalize(self, image, center, head_pose_rot):
        if not isinstance(head_pose_rot, Rotation):
            head_pose_rot = Rotation.from_matrix(head_pose_rot)
        
        normalizing_rot = self._compute_normalizing_rotation(center, head_pose_rot)
        normalized_image = self._normalize_image(image, center, normalizing_rot)
        normalized_head_rot2d = self._normalize_head_pose(head_pose_rot, normalizing_rot)
        
        return {
            "normalized_image": normalized_image,
            "normalized_head_rot2d": normalized_head_rot2d,
            "normalizing_rotation": normalizing_rot
        }

    def _normalize_image(self, image, center, normalizing_rot):
        camera_matrix_inv = np.linalg.inv(self.camera_matrix)
        distance = np.linalg.norm(center)
        a  = normalizing_rot.as_matrix()
        scale_matrix = self._get_scale_matrix(distance)
        conversion_matrix = scale_matrix @ normalizing_rot.as_matrix()
        projection_matrix = self.normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv
        
        normalized_w = 224
        normalized_h = 224
        
        normalized_image = cv2.warpPerspective(
            image, projection_matrix, (normalized_w, normalized_h))
        return normalized_image

    @staticmethod
    def _normalize_head_pose(head_pose_rot, normalizing_rot):
        normalized_head_rot = head_pose_rot * normalizing_rot
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
        return euler_angles2d * np.array([1, -1])

    @staticmethod
    def _compute_normalizing_rotation(center, head_rot):
        z_axis = _normalize_vector(center.ravel())
        head_x_axis = head_rot.as_matrix()[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, self.normalized_distance / distance],
        ], dtype=np.float32) 