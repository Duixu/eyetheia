#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Utilities for recording raw MediaPipe-derived calibration features to CSV.
"""

from __future__ import annotations

import csv
import datetime
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


LEFT_EYE = [33, 133, 159, 160, 158, 144]
RIGHT_EYE = [362, 263, 386, 387, 385, 373]
FACE_OVAL = list(range(10, 338))
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


RAW_FEATURE_COLUMNS = [
    "head_yaw",
    "head_pitch",
    "head_roll",
    "left_eye_vec_x",
    "left_eye_vec_y",
    "left_eye_vec_z",
    "right_eye_vec_x",
    "right_eye_vec_y",
    "right_eye_vec_z",
    "face_x1",
    "face_y1",
    "face_x2",
    "face_y2",
    "face_center_x",
    "face_center_y",
]


CALIBRATION_RAW_FEATURE_FIELDNAMES = [
    "session_id",
    "sample_index",
    "timestamp_iso",
    "target_x_px",
    "target_y_px",
    "screen_width",
    "screen_height",
    "frame_width",
    "frame_height",
    *RAW_FEATURE_COLUMNS,
    "feature_status",
    "feature_error",
]


def _nan() -> float:
    return float("nan")


def _empty_raw_features() -> dict[str, float]:
    return {column: _nan() for column in RAW_FEATURE_COLUMNS}


def _landmark_points(face_landmarks: Any) -> list[Any]:
    return list(getattr(face_landmarks, "landmark", []))


def _normalized_xyz(points: list[Any], indices: Iterable[int]) -> np.ndarray | None:
    coords = []
    for index in indices:
        if index >= len(points):
            return None
        point = points[index]
        coords.append([
            float(point.x),
            float(point.y),
            float(getattr(point, "z", 0.0)),
        ])
    return np.asarray(coords, dtype=np.float64)


def _unit_vector(start: np.ndarray, end: np.ndarray) -> tuple[float, float, float] | None:
    vector = end - start
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    unit = vector / norm
    return float(unit[0]), float(unit[1]), float(unit[2])


def _eye_vector(points: list[Any], eye_indices: list[int], iris_indices: list[int]) -> tuple[float, float, float]:
    eye = _normalized_xyz(points, eye_indices)
    iris = _normalized_xyz(points, iris_indices)
    if eye is None or iris is None:
        return _nan(), _nan(), _nan()

    vector = _unit_vector(eye.mean(axis=0), iris.mean(axis=0))
    if vector is None:
        return _nan(), _nan(), _nan()
    return vector


def _fallback_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    focal_length = float(max(frame_width, frame_height))
    return np.array(
        [
            [focal_length, 0.0, frame_width / 2.0],
            [0.0, focal_length, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _get_bounding_box(
    indices: list[int],
    landmarks: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    coords = [landmarks[i] for i in indices]
    x_min = max(0, min(pt[0] for pt in coords))
    y_min = max(0, min(pt[1] for pt in coords))
    x_max = min(width, max(pt[0] for pt in coords))
    y_max = min(height, max(pt[1] for pt in coords))
    return x_min, y_min, x_max, y_max


def _estimate_head_pose_degrees(points_2d: np.ndarray, frame_width: int, frame_height: int) -> tuple[float, float, float]:
    import cv2
    from company_gaze.face_landmark_pose import FaceLandmarkPose

    model_points_3d = FaceLandmarkPose.get_3d_landmarks("mediapipe")
    usable_count = min(len(model_points_3d), len(points_2d))
    if usable_count < 6:
        raise ValueError("not enough landmarks for head pose estimation")

    camera_matrix = _fallback_camera_matrix(frame_width, frame_height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        model_points_3d[:usable_count],
        points_2d[:usable_count],
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("solvePnP failed while estimating head pose")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1), dtype=np.float64)))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

    pitch = float(euler_angles[0][0])
    yaw = float(euler_angles[1][0])
    roll = float(euler_angles[2][0])
    return yaw, pitch, roll


def extract_calibration_raw_features(
    face_landmarks: Any,
    image_shape: tuple[int, int, int],
    screen_size: tuple[int, int],
) -> tuple[dict[str, float], str, str]:
    """
    Extract raw MediaPipe-derived calibration features.

    Returns (features, status, error). Missing iris landmarks produce NaN eye
    vectors and a partial status, while failures in optional head-pose
    estimation do not prevent bbox/eye features from being recorded.
    """
    del screen_size
    frame_height, frame_width = int(image_shape[0]), int(image_shape[1])
    points = _landmark_points(face_landmarks)
    if not points:
        raise ValueError("no face landmarks available")

    pixel_points = [
        (int(point.x * frame_width), int(point.y * frame_height))
        for point in points
    ]

    features = _empty_raw_features()
    errors: list[str] = []

    face_bbox = _get_bounding_box(FACE_OVAL, pixel_points, frame_width, frame_height)
    x1, y1, x2, y2 = face_bbox
    features.update(
        {
            "face_x1": float(x1),
            "face_y1": float(y1),
            "face_x2": float(x2),
            "face_y2": float(y2),
            "face_center_x": float((x1 + x2) / 2.0),
            "face_center_y": float((y1 + y2) / 2.0),
        }
    )

    try:
        points_2d = np.asarray(pixel_points, dtype=np.float64)
        yaw, pitch, roll = _estimate_head_pose_degrees(points_2d, frame_width, frame_height)
        features.update(
            {
                "head_yaw": yaw,
                "head_pitch": pitch,
                "head_roll": roll,
            }
        )
    except Exception as exc:
        errors.append(f"head_pose: {exc}")

    left_eye_vec = _eye_vector(points, LEFT_EYE, LEFT_IRIS)
    right_eye_vec = _eye_vector(points, RIGHT_EYE, RIGHT_IRIS)
    features.update(
        {
            "left_eye_vec_x": left_eye_vec[0],
            "left_eye_vec_y": left_eye_vec[1],
            "left_eye_vec_z": left_eye_vec[2],
            "right_eye_vec_x": right_eye_vec[0],
            "right_eye_vec_y": right_eye_vec[1],
            "right_eye_vec_z": right_eye_vec[2],
        }
    )

    if any(math.isnan(value) for value in (*left_eye_vec, *right_eye_vec)):
        errors.append("eye_vector: missing or invalid iris landmarks")

    status = "ok" if not errors else "partial"
    return features, status, "; ".join(errors)


@dataclass
class CalibrationRawFeatureLogger:
    output_dir: str | os.PathLike[str] | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "experiments",
                    "calibration_raw_features",
                )
            )

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        unique_suffix = uuid.uuid4().hex[:8]
        if self.session_id is None:
            self.session_id = f"calib_{timestamp}_{unique_suffix}"
        self.csv_path = os.path.join(
            str(self.output_dir),
            f"calibration_raw_features_{timestamp}_{unique_suffix}.csv",
        )
        self._write_header()

    def _write_header(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CALIBRATION_RAW_FEATURE_FIELDNAMES)
            writer.writeheader()

    def log_sample(
        self,
        sample_index: int,
        target_x_px: float,
        target_y_px: float,
        face_landmarks: Any,
        image_shape: tuple[int, int, int],
        screen_size: tuple[int, int],
    ) -> dict[str, Any]:
        frame_height, frame_width = int(image_shape[0]), int(image_shape[1])
        screen_width, screen_height = int(screen_size[0]), int(screen_size[1])
        row: dict[str, Any] = {
            "session_id": self.session_id,
            "sample_index": int(sample_index),
            "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_x_px": float(target_x_px),
            "target_y_px": float(target_y_px),
            "screen_width": screen_width,
            "screen_height": screen_height,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }

        try:
            features, status, error = extract_calibration_raw_features(
                face_landmarks,
                image_shape,
                (screen_width, screen_height),
            )
            row.update(features)
            row["feature_status"] = status
            row["feature_error"] = error
        except Exception as exc:
            row.update(_empty_raw_features())
            row["feature_status"] = "failed"
            row["feature_error"] = str(exc)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CALIBRATION_RAW_FEATURE_FIELDNAMES)
            writer.writerow(row)

        return row
