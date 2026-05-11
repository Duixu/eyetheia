from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from utils.company_gaze_mapper import CompanyGazeMapper
from utils.config import MID_X, MID_Y, SCREEN_HEIGHT, SCREEN_WIDTH


COMPANY_SWIN_MODEL_ID = "company_swin"
COMPANY_CALIBRATION_MAPPER = "mapper"
COMPANY_CALIBRATION_ARC = "arc"
DEFAULT_COMPANY_GAZE_WEIGHT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "Iter_19_swin_peg.pt")
)


@dataclass(frozen=True)
class CompanyGazePrediction:
    x_px: float
    y_px: float
    raw_pitch: float
    raw_yaw: float
    confidence: float
    model_name: str = COMPANY_SWIN_MODEL_ID


class CompanyGazeTracker:
    """
    Local OpenCV runner for the company Swin gaze model.

    The company model emits pitch/yaw angles.  This runner keeps the model
    weights fixed and learns a per-user calibration mapper from pitch/yaw to
    EyeTheia's common screen pixel space.
    """

    def __init__(
        self,
        calibration_point_count: int = 13,
        weight_path: str | None = None,
        calibration_mode: str = COMPANY_CALIBRATION_MAPPER,
        gaze_model_type: str = "swin",
    ) -> None:
        if calibration_mode not in (COMPANY_CALIBRATION_MAPPER, COMPANY_CALIBRATION_ARC):
            raise ValueError('company calibration mode must be "mapper" or "arc"')
        if calibration_mode == COMPANY_CALIBRATION_ARC:
            raise NotImplementedError(
                "Company ARC calibration is preserved in company_gaze.utils.arc_calibration, "
                "but the local EyeTheia integration currently runs the mapper flow."
            )

        from utils.utils import get_numbered_calibration_points

        self.calibration_point_count = calibration_point_count
        self.calibration_points = get_numbered_calibration_points(calibration_point_count)
        self.weight_path = weight_path or DEFAULT_COMPANY_GAZE_WEIGHT_PATH
        self.calibration_mode = calibration_mode

        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Company gaze weight file not found: {self.weight_path}")

        import torch
        from .gaze_estimator import GazeEstimator

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = GazeEstimator(
            gaze_model_type=gaze_model_type,
            device=str(self.device),
            weight_path=self.weight_path,
        )
        self.mapper: CompanyGazeMapper | None = None
        self.window_name = "EyeTheia Live Gaze Visualization"
        self.calibration_window_name = "Company Gaze Calibration"
        self.current_index = 0
        self.current_target: tuple[int, int] | None = None
        self.calibration_done = False
        self.gaze_filtered = True
        self.gaze_filter_x = self._new_filter()
        self.gaze_filter_y = self._new_filter()

        base_dim = min(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.point_radius = max(8, int(base_dim * 0.006))
        self.click_radius = max(10, int(base_dim * 0.012))

    def run(self, webcam: cv2.VideoCapture) -> None:
        self.run_calibration(webcam)
        if self.mapper is None or not self.mapper.fitted:
            raise RuntimeError("Company gaze mapper is not fitted; finish calibration first.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fullscreen = True

        gaze_x_px, gaze_y_px = float(MID_X), float(MID_Y)
        raw_pitch, raw_yaw, confidence = 0.0, 0.0, 0.0
        self.reset_gaze_filters()
        from utils.OneEuroTuner import OneEuroTuner

        tuner = OneEuroTuner(window_name="EyeTheia Controls")
        start_time = time.perf_counter()

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while True:
                success, img = webcam.read()
                if not success:
                    print("Error reading from the webcam.")
                    break

                freq, mincutoff, beta, dcutoff = tuner.update_filters(
                    self.gaze_filter_x, self.gaze_filter_y
                )

                prediction = self.predict_frame(img, face_mesh)
                if prediction is not None:
                    raw_pitch = prediction.raw_pitch
                    raw_yaw = prediction.raw_yaw
                    confidence = prediction.confidence
                    timestamp = time.perf_counter() - start_time
                    if self.gaze_filtered:
                        gaze_x_px, gaze_y_px = self.filter_gaze_pixels(
                            prediction.x_px,
                            prediction.y_px,
                            timestamp,
                        )
                    else:
                        gaze_x_px, gaze_y_px = prediction.x_px, prediction.y_px

                frame = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255
                cv2.circle(frame, (int(gaze_x_px), int(gaze_y_px)), 23, (0, 0, 0), 2)

                cv2.putText(
                    frame,
                    f"Model: Company Swin  pitch={raw_pitch:.3f}  yaw={raw_yaw:.3f}  conf={confidence:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"OneEuroFilter: freq={freq}Hz  mincutoff={mincutoff:.2f}  beta={beta:.3f}  dcutoff={dcutoff:.2f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Esc: toggle fullscreen  |  Q: quit",
                    (20, SCREEN_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 27:
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        self.window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("EyeTheia Controls")

    def run_calibration(self, webcam: cv2.VideoCapture) -> CompanyGazeMapper:
        print("\nCompany gaze calibration started")
        self.current_index = 0
        self.current_target = None
        self.calibration_done = False
        samples: list[tuple[float, float, float, float]] = []

        cv2.namedWindow(self.calibration_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.calibration_window_name, self._mouse_callback)

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while not self.calibration_done:
                calibration_frame = self._render_calibration_frame()
                cv2.setWindowProperty(
                    self.calibration_window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
                cv2.imshow(self.calibration_window_name, calibration_frame)
                cv2.waitKey(1)

                success, img = webcam.read()
                if not success:
                    raise RuntimeError("Error reading from the webcam.")

                if self.current_target is None:
                    continue

                target_x, target_y = self.current_target
                result = self._estimate_company_gaze(img, face_mesh)
                if result is None:
                    print("No valid face landmarks for this calibration point; please keep looking at the target.")
                    continue

                pitch = float(result["pitch"])
                yaw = float(result["yaw"])
                samples.append((pitch, yaw, float(target_x), float(target_y)))
                print(
                    f"Captured company gaze sample {len(samples)}/{len(self.calibration_points)}: "
                    f"pitch={pitch:.4f}, yaw={yaw:.4f}, target=({target_x}, {target_y})"
                )

                self.current_target = None
                self.current_index += 1
                if self.current_index >= len(self.calibration_points):
                    self.calibration_done = True

        cv2.destroyWindow(self.calibration_window_name)
        self.mapper = CompanyGazeMapper.fit_from_samples(
            samples,
            degree=2,
            pitch_yaw_unit="rad",
            screen_size=(SCREEN_WIDTH, SCREEN_HEIGHT),
        )
        print(
            "Company gaze calibration completed. "
            f"Mean mapper error: {self.mapper.mean_error_px():.2f}px"
        )
        return self.mapper

    def predict_frame(
        self,
        img: np.ndarray,
        face_mesh: Any,
    ) -> CompanyGazePrediction | None:
        if self.mapper is None or not self.mapper.fitted:
            raise RuntimeError("Company gaze mapper is not fitted; finish calibration first.")

        result = self._estimate_company_gaze(img, face_mesh)
        if result is None:
            return None

        pitch = float(result["pitch"])
        yaw = float(result["yaw"])
        x_px, y_px = self.mapper.predict(pitch, yaw)
        return CompanyGazePrediction(
            x_px=x_px,
            y_px=y_px,
            raw_pitch=pitch,
            raw_yaw=yaw,
            confidence=float(result.get("confidence", 0.0)),
        )

    def filter_gaze_pixels(self, gaze_x_px: float, gaze_y_px: float, timestamp: float) -> tuple[float, float]:
        filtered_x = float(self.gaze_filter_x.filter(float(gaze_x_px), timestamp))
        filtered_y = float(self.gaze_filter_y.filter(float(gaze_y_px), timestamp))
        return filtered_x, filtered_y

    def reset_gaze_filters(self) -> None:
        self.gaze_filter_x = self._new_filter()
        self.gaze_filter_y = self._new_filter()

    def _estimate_company_gaze(self, img: np.ndarray, face_mesh: Any) -> dict[str, Any] | None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_mp = face_mesh.process(img_rgb)
        if not img_mp.multi_face_landmarks:
            return None

        landmarks_np = self._landmarks_to_pixels(img_mp.multi_face_landmarks[0], img.shape)
        if not self.estimator.is_landmarks_valid(landmarks_np):
            return None
        return self.estimator.forward(img, landmarks_np)

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.calibration_done or self.current_target is not None:
            return

        expected_point = self.calibration_points[self.current_index]
        if self._within_radius((x, y), expected_point, self.click_radius):
            print(f"Correct click at point {self.current_index} ({x}, {y})")
            self.current_target = expected_point
        else:
            print(f"Incorrect click at ({x}, {y}). Please click on point {self.current_index + 1}.")

    def _render_calibration_frame(self) -> np.ndarray:
        frame = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255
        if not self.calibration_done and self.current_index < len(self.calibration_points):
            tx, ty = self.calibration_points[self.current_index]
            cv2.circle(frame, (int(tx), int(ty)), self.point_radius, (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Company Swin calibration {self.current_index + 1}/{len(self.calibration_points)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Look at the black circle, then click inside it.",
                (30, SCREEN_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _landmarks_to_pixels(face_landmarks: Any, image_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        return np.array(
            [[float(point.x * w), float(point.y * h)] for point in face_landmarks.landmark],
            dtype=np.float32,
        )

    @staticmethod
    def _within_radius(pt1: tuple[float, float], pt2: tuple[float, float], radius: float) -> bool:
        return bool(np.linalg.norm(np.array(pt1) - np.array(pt2)) <= radius)

    @staticmethod
    def _new_filter():
        from OneEuroFilter import OneEuroFilter

        return OneEuroFilter(freq=30, mincutoff=1.5, beta=0.02, dcutoff=1.0)
