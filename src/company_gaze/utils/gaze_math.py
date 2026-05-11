import numpy as np

def vector_to_angle(vector: np.ndarray) -> np.ndarray:
    """
    将3D gaze向量转换为pitch/yaw角度（弧度）。
    vector: (3,) gaze向量
    返回: (2,) [pitch, yaw]，单位为弧度
    """
    assert vector.shape == (3, )
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw]) 