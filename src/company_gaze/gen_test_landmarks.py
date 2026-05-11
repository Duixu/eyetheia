import cv2
import mediapipe as mp
import numpy as np

# 读取测试图片
image_path = "src/test.jpg"  # 确保图片放在 src/ 目录下
image = cv2.imread(image_path)
if image is None:
    raise RuntimeError(f"无法读取图片: {image_path}")

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    # 将 BGR 转为 RGB
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

landmarks = []
if results.multi_face_landmarks:
    # 取第一张人脸
    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * image.shape[1])  # 转成像素坐标
        y = int(lm.y * image.shape[0])
        landmarks.append([x, y])

landmarks = np.asarray(landmarks, dtype=np.float32)
print(f"生成 landmarks shape: {landmarks.shape}")

# 保存为 npy 文件
np.save("src/test_landmarks.npy", landmarks)
print("保存完成: src/test_landmarks.npy")