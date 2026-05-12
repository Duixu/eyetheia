from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from utils.company_gaze_mapper import CompanyArcGazeMapper
from utils.config import BATCH_SIZE, EPOCH, LR, MID_X, MID_Y, SCREEN_HEIGHT, SCREEN_WIDTH


COMPANY_SWIN_MODEL_ID = "company_swin"
COMPANY_CALIBRATION_MAPPER = "mapper"
COMPANY_CALIBRATION_ARC = "arc"
COMPANY_CONFIRMATION_CLICK = "click"
COMPANY_CONFIRMATION_DWELL = "dwell"
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
        calibration_confirmation: str = COMPANY_CONFIRMATION_CLICK,
        eyetheia_finetune_tracker: Any | None = None,
        run_eyetheia_after_calibration: bool = False,
        dwell_seconds: float = 1.2,
    ) -> None:
        if calibration_mode not in (COMPANY_CALIBRATION_MAPPER, COMPANY_CALIBRATION_ARC):
            raise ValueError('company calibration mode must be "mapper" or "arc"')
        if calibration_confirmation not in (COMPANY_CONFIRMATION_CLICK, COMPANY_CONFIRMATION_DWELL):
            raise ValueError('company calibration confirmation must be "click" or "dwell"')
        from utils.utils import get_numbered_calibration_points

        self.calibration_point_count = calibration_point_count
        self.calibration_points = get_numbered_calibration_points(calibration_point_count)
        self.weight_path = weight_path or DEFAULT_COMPANY_GAZE_WEIGHT_PATH
        self.calibration_mode = calibration_mode
        self.calibration_confirmation = calibration_confirmation
        self.eyetheia_finetune_tracker = eyetheia_finetune_tracker
        self.run_eyetheia_after_calibration = run_eyetheia_after_calibration
        self.dwell_seconds = float(dwell_seconds)

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
        self.mapper: CompanyArcGazeMapper | None = None
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
        if self.run_eyetheia_after_calibration:
            if self.eyetheia_finetune_tracker is None:
                raise RuntimeError("EyeTheia tracker is not configured.")
            self.eyetheia_finetune_tracker.run_tracking_loop(webcam)
            return

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

    def run_calibration(self, webcam: cv2.VideoCapture) -> CompanyArcGazeMapper:
        print("\nCompany gaze calibration started")
        self.current_index = 0
        self.current_target = None
        self.calibration_done = False
        samples: list[tuple[float, float, float, float, tuple[float, float, float]]] = []
        eyetheia_samples = []
        dwell_started_at = time.perf_counter()

        cv2.namedWindow(self.calibration_window_name, cv2.WINDOW_NORMAL)
        if self.calibration_confirmation == COMPANY_CONFIRMATION_CLICK:
            cv2.setMouseCallback(self.calibration_window_name, self._mouse_callback)

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while not self.calibration_done:
                if self.calibration_confirmation == COMPANY_CONFIRMATION_DWELL:
                    self.current_target = self.calibration_points[self.current_index]

                dwell_elapsed = time.perf_counter() - dwell_started_at
                calibration_frame = self._render_calibration_frame(dwell_elapsed)
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
                if (
                    self.calibration_confirmation == COMPANY_CONFIRMATION_DWELL
                    and dwell_elapsed < self.dwell_seconds
                ):
                    continue

                target_x, target_y = self.current_target
                result, face_landmarks = self._estimate_company_gaze_with_landmarks(img, face_mesh)
                if result is None:
                    print("No valid face landmarks for this calibration point; please keep looking at the target.")
                    if self.calibration_confirmation == COMPANY_CONFIRMATION_DWELL:
                        dwell_started_at = time.perf_counter()
                    continue

                pitch = float(result["pitch"])
                yaw = float(result["yaw"])
                face_center = self._coerce_face_center(result.get("face_center"))
                samples.append((pitch, yaw, float(target_x), float(target_y), face_center))
                if self.eyetheia_finetune_tracker is not None and face_landmarks is not None:
                    eyetheia_samples.append(
                        self._build_eyetheia_calibration_sample(
                            img,
                            face_landmarks,
                            float(target_x),
                            float(target_y),
                        )
                    )
                print(
                    f"Captured company gaze sample {len(samples)}/{len(self.calibration_points)}: "
                    f"pitch={pitch:.4f}, yaw={yaw:.4f}, target=({target_x}, {target_y})"
                )

                self.current_target = None
                self.current_index += 1
                dwell_started_at = time.perf_counter()
                if self.current_index >= len(self.calibration_points):
                    self.calibration_done = True

        cv2.destroyWindow(self.calibration_window_name)
        self.mapper = CompanyArcGazeMapper.fit_from_samples(
            samples,
            pitch_yaw_unit="rad",
            screen_size=(SCREEN_WIDTH, SCREEN_HEIGHT),
        )
        print(
            "Company gaze calibration completed. "
            f"Mean mapper error: {self.mapper.mean_error_px():.2f}px"
        )
        if self.eyetheia_finetune_tracker is not None:
            self._fine_tune_eyetheia_tracker(eyetheia_samples)
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
        face_center = self._coerce_face_center(result.get("face_center"))
        x_px, y_px = self.mapper.predict(pitch, yaw, face_center=face_center)
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
        result, _ = self._estimate_company_gaze_with_landmarks(img, face_mesh)
        return result

    def _estimate_company_gaze_with_landmarks(
        self,
        img: np.ndarray,
        face_mesh: Any,
    ) -> tuple[dict[str, Any] | None, Any | None]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_mp = face_mesh.process(img_rgb)
        if not img_mp.multi_face_landmarks:
            return None, None

        face_landmarks = img_mp.multi_face_landmarks[0]
        landmarks_np = self._landmarks_to_pixels(face_landmarks, img.shape)
        if not self.estimator.is_landmarks_valid(landmarks_np):
            return None, face_landmarks
        return self.estimator.forward(img, landmarks_np), face_landmarks

    def _build_eyetheia_calibration_sample(
        self,
        img: np.ndarray,
        face_landmarks: Any,
        target_x: float,
        target_y: float,
    ):
        if self.eyetheia_finetune_tracker is None:
            raise RuntimeError("EyeTheia fine-tune tracker is not configured")

        features = self.eyetheia_finetune_tracker.extract_features(
            img,
            face_landmarks,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )
        from utils.utils import pixels_to_gaze_cm

        gaze_x, gaze_y = pixels_to_gaze_cm(target_x, target_y, SCREEN_WIDTH, SCREEN_HEIGHT)
        return features, (gaze_x, gaze_y)

    def _fine_tune_eyetheia_tracker(self, eyetheia_samples: list[Any]) -> None:
        if not eyetheia_samples:
            print("No EyeTheia calibration samples were collected; skipping EyeTheia fine-tuning.")
            return

        print(
            "\nFine-tuning EyeTheia baseline with images captured during company gaze calibration..."
        )
        from tracker.CalibrationDataset import CalibrationDataset

        self.eyetheia_finetune_tracker.reset_model()
        dataset = CalibrationDataset(eyetheia_samples)
        self.eyetheia_finetune_tracker.calibration.set_capture_points(eyetheia_samples)
        self.eyetheia_finetune_tracker.train(
            dataset,
            epochs=EPOCH,
            learning_rate=LR,
            batch_size=BATCH_SIZE,
        )
        mean_error, std_error = (
            self.eyetheia_finetune_tracker.calibration.evaluate_calibration_accuracy()
        )
        print(
            "EyeTheia fine-tuning from company calibration images completed. "
            f"Mean Error = {mean_error:.2f}, Std Dev = {std_error:.2f}"
        )

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.calibration_done or self.current_target is not None:
            return

        expected_point = self.calibration_points[self.current_index]
        if self._within_radius((x, y), expected_point, self.click_radius):
            print(f"Correct click at point {self.current_index} ({x}, {y})")
            self.current_target = expected_point
        else:
            print(f"Incorrect click at ({x}, {y}). Please click on point {self.current_index + 1}.")

    def _render_calibration_frame(self, dwell_elapsed: float = 0.0) -> np.ndarray:
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
            if self.calibration_confirmation == COMPANY_CONFIRMATION_DWELL:
                remaining = max(0.0, self.dwell_seconds - dwell_elapsed)
                instruction = f"Look at the black circle and hold still: {remaining:.1f}s"
            else:
                instruction = "Look at the black circle, then click inside it."
            cv2.putText(
                frame,
                instruction,
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
    def _coerce_face_center(value: Any) -> tuple[float, float, float]:
        if value is None:
            return (0.0, 0.0, 0.5)
        center = np.asarray(value, dtype=np.float64).reshape(-1)
        if center.size < 3:
            return (0.0, 0.0, 0.5)
        return (float(center[0]), float(center[1]), float(center[2]))

    @staticmethod
    def _new_filter():
        from OneEuroFilter import OneEuroFilter

        return OneEuroFilter(freq=30, mincutoff=1.5, beta=0.02, dcutoff=1.0)
