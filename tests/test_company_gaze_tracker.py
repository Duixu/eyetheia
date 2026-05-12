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

class FakeEstimator:
    def is_landmarks_valid(self, landmarks):
        return True

    def forward(self, image, landmarks):
        return {
            "pitch": 0.1,
            "yaw": -0.2,
            "face_center": np.array([0.0, 0.0, 0.5]),
            "confidence": 0.75,
        }


class FakeMapper:
    fitted = True

    def __init__(self):
        self.last_args = None

    def predict(self, pitch, yaw, face_center=None):
        self.last_args = (pitch, yaw, face_center)
        return 123.0, 456.0


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
    tracker = object.__new__(CompanyGazeTracker)
    mapper = FakeMapper()
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
    assert prediction.x_px == 123.0
    assert prediction.y_px == 456.0
    assert mapper.last_args == (0.1, -0.2, (0.0, 0.0, 0.5))
