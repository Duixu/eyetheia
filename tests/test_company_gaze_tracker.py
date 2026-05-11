import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

sys.modules.setdefault(
    "cv2",
    SimpleNamespace(
        COLOR_BGR2RGB=1,
        cvtColor=lambda image, code: image,
    ),
)
sys.modules.setdefault(
    "mediapipe",
    SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=object))),
)

from company_gaze.company_gaze_tracker import CompanyGazeTracker
from utils.company_gaze_mapper import CompanyGazeMapper


class FakeEstimator:
    def is_landmarks_valid(self, landmarks):
        return True

    def forward(self, image, landmarks):
        return {
            "pitch": 0.1,
            "yaw": -0.2,
            "confidence": 0.75,
        }


class FakeFaceMesh:
    def process(self, image_rgb):
        landmarks = [
            SimpleNamespace(x=0.5, y=0.5, z=0.0),
            SimpleNamespace(x=0.6, y=0.5, z=0.0),
        ]
        return SimpleNamespace(
            multi_face_landmarks=[
                SimpleNamespace(landmark=landmarks),
            ]
        )


def test_company_gaze_tracker_predict_frame_maps_pitch_yaw_to_screen_pixels():
    mapper = CompanyGazeMapper.fit_from_samples(
        [
            (-0.2, -0.2, 100, 100),
            (-0.2, 0.2, 100, 900),
            (0.2, -0.2, 1800, 100),
            (0.2, 0.2, 1800, 900),
            (0.0, 0.0, 960, 540),
            (0.1, -0.2, 1200, 300),
        ],
        degree=2,
        pitch_yaw_unit="rad",
        ridge=0.0,
        screen_size=(1920, 1080),
    )

    tracker = object.__new__(CompanyGazeTracker)
    tracker.mapper = mapper
    tracker.estimator = FakeEstimator()

    prediction = tracker.predict_frame(
        np.zeros((32, 32, 3), dtype=np.uint8),
        FakeFaceMesh(),
    )

    assert prediction is not None
    assert prediction.raw_pitch == 0.1
    assert prediction.raw_yaw == -0.2
    assert prediction.confidence == 0.75
    assert prediction.x_px == mapper.predict(0.1, -0.2)[0]
    assert prediction.y_px == mapper.predict(0.1, -0.2)[1]
