import os

import cv2
import mediapipe as mp
import numpy as np
import torch

from company_gaze.gaze_estimator import GazeEstimator


def generate_landmarks(image: np.ndarray) -> np.ndarray:
    """Return MediaPipe face landmarks as an (N, 2) pixel-coordinate array."""
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        raise RuntimeError("No face was detected in the test image")

    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        landmarks.append([x, y])

    return np.asarray(landmarks, dtype=np.float32)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "test.png")
    weight_path = os.path.join(base_dir, "models", "Iter_19_swin_peg.pt")

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load test image: {image_path}")
    if not os.path.exists(weight_path):
        raise RuntimeError(f"Company gaze weight file not found: {weight_path}")
    print("Loaded image:", image.shape)

    landmarks = generate_landmarks(image)
    print("MediaPipe landmarks:", landmarks.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = GazeEstimator(
        gaze_model_type="swin",
        device=device,
        weight_path=weight_path,
    )

    result = estimator.forward(image, landmarks)

    print("\n=== Company Gaze Result ===")
    print("device:", device)
    print("pitch (rad):", result["pitch"])
    print("yaw   (rad):", result["yaw"])
    print("pitch (deg):", np.rad2deg(result["pitch"]))
    print("yaw   (deg):", np.rad2deg(result["yaw"]))
    print("confidence:", result["confidence"])
    print("gaze_vector:", result["gaze_vector"])
    print("face_center:", result.get("face_center"))

    face_center = result.get("face_center")
    if face_center is None or len(face_center) < 2:
        center = (image.shape[1] // 2, image.shape[0] // 2)
    else:
        center = (int(face_center[0]), int(face_center[1]))

    vec = result["gaze_vector"] * 100
    end_point = (int(center[0] + vec[0]), int(center[1] + vec[1]))
    vis = image.copy()
    cv2.arrowedLine(vis, center, end_point, (0, 255, 0), 2)
    cv2.imshow("Gaze Vector", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
