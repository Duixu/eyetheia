import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/utils/company_gaze_mapper.py")
)
spec = importlib.util.spec_from_file_location("company_gaze_mapper", MODULE_PATH)
company_gaze_mapper = importlib.util.module_from_spec(spec)
sys.modules["company_gaze_mapper"] = company_gaze_mapper
spec.loader.exec_module(company_gaze_mapper)
CompanyGazeMapper = company_gaze_mapper.CompanyGazeMapper
CompanyArcGazeMapper = company_gaze_mapper.CompanyArcGazeMapper


def _quadratic_target(pitch, yaw):
    p = pitch / 90.0
    y = yaw / 90.0
    x = 960 + 300 * p - 120 * y + 40 * p * y
    yy = 540 - 80 * p + 260 * y + 25 * y * y
    return x, yy


def test_company_gaze_mapper_quadratic_fit_predicts_screen_points():
    samples = []
    for pitch, yaw in [(-20, -20), (-20, 15), (0, 0), (10, -15), (20, 20), (30, -5)]:
        x_px, y_px = _quadratic_target(pitch, yaw)
        samples.append((pitch, yaw, x_px, y_px))

    mapper = CompanyGazeMapper.fit_from_samples(
        samples,
        degree=2,
        ridge=0.0,
        screen_size=(1920, 1080),
    )

    pred = mapper.predict(12, -7)
    expected = _quadratic_target(12, -7)

    assert np.allclose(pred, expected, atol=1e-6)
    assert mapper.mean_error_px() < 1e-6


def test_company_gaze_mapper_save_and_load():
    samples = [
        (-20, -20, 100, 100),
        (-20, 20, 100, 900),
        (20, -20, 1800, 100),
        (20, 20, 1800, 900),
        (0, 0, 960, 540),
        (10, -10, 1300, 300),
    ]
    mapper = CompanyGazeMapper.fit_from_samples(samples, degree=2, screen_size=(1920, 1080))
    save_path = Path(__file__).with_name("company_gaze_mapper_test_output.json")

    try:
        mapper.save(save_path)
        loaded = CompanyGazeMapper.load(save_path)

        assert loaded.screen_size == (1920, 1080)
        assert loaded.predict(5, -5) == pytest.approx(mapper.predict(5, -5))
    finally:
        if save_path.exists():
            save_path.unlink()


def test_company_gaze_mapper_requires_enough_samples():
    mapper = CompanyGazeMapper(degree=2)
    mapper.add_sample(0, 0, 960, 540)

    with pytest.raises(ValueError, match="requires at least 6 samples"):
        mapper.fit()


def _angles_from_attention_vector(vector):
    pitch = np.arcsin(-vector[1])
    yaw = np.arcsin(-vector[0] / np.cos(pitch))
    return float(pitch), float(yaw)


def test_company_arc_gaze_mapper_fits_company_screen_intersection_geometry():
    face_center = np.array([0.0, 0.0, 0.5])
    tvec = np.array([960.0, -540.0, 1000.0])
    origin = face_center + tvec
    targets = [
        (200.0, 120.0),
        (960.0, 120.0),
        (1720.0, 120.0),
        (200.0, 900.0),
        (960.0, 900.0),
        (1720.0, 900.0),
    ]
    samples = []
    for x_px, y_px in targets:
        target = np.array([x_px, -y_px, 0.0])
        attention_vec = target - origin
        attention_vec = attention_vec / np.linalg.norm(attention_vec)
        pitch, yaw = _angles_from_attention_vector(attention_vec)
        samples.append((pitch, yaw, x_px, y_px, face_center))

    mapper = CompanyArcGazeMapper.fit_from_samples(
        samples,
        pitch_yaw_unit="rad",
        screen_size=(1920, 1080),
    )

    assert mapper.mean_error_px() < 1e-3
    assert mapper.predict(samples[0][0], samples[0][1], face_center) == pytest.approx(
        targets[0],
        abs=1e-3,
    )


def test_company_arc_gaze_mapper_requires_six_samples():
    mapper = CompanyArcGazeMapper()
    mapper.add_sample(0.0, 0.0, 960.0, 540.0)

    with pytest.raises(ValueError, match="requires at least 6 samples"):
        mapper.fit()
