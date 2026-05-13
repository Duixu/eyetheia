import csv
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from utils.calibration_raw_features import (
    CALIBRATION_RAW_FEATURE_FIELDNAMES,
    CalibrationRawFeatureLogger,
    extract_calibration_raw_features,
)


def _fake_landmarks(count=478):
    landmarks = []
    for index in range(count):
        landmarks.append(
            SimpleNamespace(
                x=0.1 + (index % 20) * 0.01,
                y=0.2 + (index % 15) * 0.01,
                z=0.001 * index,
            )
        )

    landmarks[33] = SimpleNamespace(x=0.20, y=0.30, z=0.00)
    landmarks[133] = SimpleNamespace(x=0.30, y=0.30, z=0.00)
    landmarks[159] = SimpleNamespace(x=0.25, y=0.25, z=0.00)
    landmarks[160] = SimpleNamespace(x=0.23, y=0.27, z=0.00)
    landmarks[158] = SimpleNamespace(x=0.27, y=0.27, z=0.00)
    landmarks[144] = SimpleNamespace(x=0.25, y=0.35, z=0.00)

    landmarks[362] = SimpleNamespace(x=0.60, y=0.30, z=0.00)
    landmarks[263] = SimpleNamespace(x=0.70, y=0.30, z=0.00)
    landmarks[386] = SimpleNamespace(x=0.65, y=0.25, z=0.00)
    landmarks[387] = SimpleNamespace(x=0.63, y=0.27, z=0.00)
    landmarks[385] = SimpleNamespace(x=0.67, y=0.27, z=0.00)
    landmarks[373] = SimpleNamespace(x=0.65, y=0.35, z=0.00)

    for index in [474, 475, 476, 477]:
        if index < count:
            landmarks[index] = SimpleNamespace(x=0.25, y=0.30, z=0.10)
    for index in [469, 470, 471, 472]:
        if index < count:
            landmarks[index] = SimpleNamespace(x=0.65, y=0.30, z=0.10)

    return SimpleNamespace(landmark=landmarks)


def test_extract_calibration_raw_features_bbox_center_and_unit_vectors(monkeypatch):
    monkeypatch.setattr(
        "utils.calibration_raw_features._estimate_head_pose_degrees",
        lambda points_2d, frame_width, frame_height: (1.0, 2.0, 3.0),
    )

    features, status, error = extract_calibration_raw_features(
        _fake_landmarks(),
        (100, 200, 3),
        (1920, 1080),
    )

    assert status == "ok"
    assert error == ""
    assert features["head_yaw"] == 1.0
    assert features["head_pitch"] == 2.0
    assert features["head_roll"] == 3.0
    assert features["face_x1"] <= features["face_center_x"] <= features["face_x2"]
    assert features["face_y1"] <= features["face_center_y"] <= features["face_y2"]

    left_vec = np.array([
        features["left_eye_vec_x"],
        features["left_eye_vec_y"],
        features["left_eye_vec_z"],
    ])
    right_vec = np.array([
        features["right_eye_vec_x"],
        features["right_eye_vec_y"],
        features["right_eye_vec_z"],
    ])
    assert np.linalg.norm(left_vec) == pytest.approx(1.0)
    assert np.linalg.norm(right_vec) == pytest.approx(1.0)


def test_extract_calibration_raw_features_missing_iris_is_partial(monkeypatch):
    monkeypatch.setattr(
        "utils.calibration_raw_features._estimate_head_pose_degrees",
        lambda points_2d, frame_width, frame_height: (1.0, 2.0, 3.0),
    )

    features, status, error = extract_calibration_raw_features(
        _fake_landmarks(count=468),
        (100, 200, 3),
        (1920, 1080),
    )

    assert status == "partial"
    assert "eye_vector" in error
    assert math.isnan(features["left_eye_vec_x"])
    assert math.isnan(features["right_eye_vec_x"])


def test_calibration_raw_feature_logger_creates_distinct_sessions_and_header(tmp_path):
    first = CalibrationRawFeatureLogger(output_dir=tmp_path)
    second = CalibrationRawFeatureLogger(output_dir=tmp_path)

    assert first.csv_path != second.csv_path
    with open(first.csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        assert next(reader) == CALIBRATION_RAW_FEATURE_FIELDNAMES


def test_calibration_raw_feature_logger_writes_failed_feature_row(tmp_path):
    logger = CalibrationRawFeatureLogger(output_dir=tmp_path)
    logger.log_sample(
        sample_index=3,
        target_x_px=100,
        target_y_px=200,
        face_landmarks=SimpleNamespace(landmark=[]),
        image_shape=(100, 200, 3),
        screen_size=(1920, 1080),
    )

    with open(logger.csv_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["sample_index"] == "3"
    assert rows[0]["feature_status"] == "failed"
    assert "no face landmarks" in rows[0]["feature_error"]
