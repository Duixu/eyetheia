#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`Calibration` module
=========================

This module handles the calibration process for fine-tuning the gaze tracking model.
It captures gaze targets using mouse clicks and extracts corresponding features for fine-tuning.

:author: Pather Stevenson
:date: February 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import (
    pixels_to_gaze_cm,
    normalize_MPIIFaceGaze,
    get_numbered_calibration_points,
    euclidan_distance_radius,
)
from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, CALIBRATION_PTS
from tracker.CalibrationDataset import CalibrationDataset


class Calibration:
    """
    Handles the calibration process for fine-tuning the gaze tracking model.
    Uses mouse clicks to capture real gaze targets and extract corresponding features.
    """

    def __init__(self, gaze_tracker: "GazeTracker") -> None:
        """
        Initializes the Calibration object.

        :param gaze_tracker: Instance of GazeTracker to extract gaze features.
        :type gaze_tracker: GazeTracker
        """
        self.gaze_tracker = gaze_tracker
        self.capture_points: list[
            tuple[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                tuple[float, float]
            ]
        ] = []
        self.margin: int = 20
        self.current_target: tuple[int, int] | None = None
        self.window_name: str = "Calibration Window"
        self.calibration_done: bool = False
        self.calibration_points = get_numbered_calibration_points()
        self.current_index = 0

        self.base_dim = min(SCREEN_WIDTH, SCREEN_HEIGHT)

        self.point_radius: int = max(8, int(self.base_dim * 0.006))
        self.click_radius: int = max(10, int(self.base_dim * 0.012))

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """
        Mouse click callback function to capture gaze target in a specific order.
        The user must click on the calibration points sequentially.

        :param event: Type of mouse event.
        :type event: int
        :param x: X coordinate of mouse click.
        :type x: int
        :param y: Y coordinate of mouse click.
        :type y: int
        :param flags: Additional event parameters.
        :type flags: int
        :param param: Extra parameters (unused).
        :type param: any
        """
        if event != cv2.EVENT_LBUTTONDOWN or self.calibration_done:
            return

        expected_point = self.calibration_points[self.current_index]

        if euclidan_distance_radius((x, y), expected_point, self.click_radius):
            print(f"Correct click at point {self.current_index} ({x}, {y})")
            self.current_target = (x, y)

            self.current_index += 1
            if self.current_index >= len(self.calibration_points):
                self.calibration_done = True
        else:
            print(f"Incorrect click at ({x}, {y}). Please click on point {self.current_index}!")

    def set_capture_points(self, capture_points):
        self.capture_points = capture_points

    def _render_calibration_frame(self) -> np.ndarray:
        """
        Render a white fullscreen frame with the current calibration point.

        :return: Calibration display image.
        :rtype: np.ndarray
        """
        frame = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255

        if not self.calibration_done and self.current_index < len(self.calibration_points):
            tx, ty = self.calibration_points[self.current_index]

            cv2.circle(frame, (int(tx), int(ty)), self.point_radius, (0, 0, 0), -1)

            cv2.putText(
                frame,
                f"Click point {self.current_index + 1}/{len(self.calibration_points)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Please click inside the black circle.",
                (30, SCREEN_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return frame

    def evaluate_calibration_accuracy(self) -> tuple[float, float]:
        """
        Evaluates the accuracy of the fine-tuned gaze tracking model on calibration points.

        :return: Mean Euclidean distance error and standard deviation in cm.
        :rtype: tuple[float, float]
        """
        if not self.capture_points:
            print("No calibration data available for evaluation.")
            return 0.0, 0.0

        print("\nEvaluation of calibration started")

        total_errors = []
        self.gaze_tracker.model.eval()

        with torch.no_grad():
            for (face_input, left_eye_input, right_eye_input, face_grid_input), (gaze_x_true, gaze_y_true) in self.capture_points:
                face_input = face_input.to(self.gaze_tracker.device)
                left_eye_input = left_eye_input.to(self.gaze_tracker.device)
                right_eye_input = right_eye_input.to(self.gaze_tracker.device)
                face_grid_input = face_grid_input.to(self.gaze_tracker.device)

                # Model prediction
                gaze_prediction = self.gaze_tracker.model(face_input, left_eye_input, right_eye_input, face_grid_input)
                gaze_x_pred, gaze_y_pred = gaze_prediction.cpu().numpy().flatten()


                # Compute Euclidean error
                error =  np.linalg.norm(np.array([gaze_x_pred, gaze_y_pred]) - np.array([gaze_x_true, gaze_y_true]))
                total_errors.append(error)

                print(f"True Gaze: ({gaze_x_true:.2f}, {gaze_y_true:.2f}), "
                      f"Predicted Gaze: ({gaze_x_pred:.2f}, {gaze_y_pred:.2f}), "
                      f"Error: {error:.2f}")

        mean_error = np.mean(total_errors)
        std_error = np.std(total_errors)

        print(f"\nCalibration Accuracy: Mean Error = {mean_error:.2f}, Std Dev = {std_error:.2f}\n")

        return mean_error, std_error

    def run_calibration(self, webcam: cv2.VideoCapture) -> CalibrationDataset:
        """
        Runs the calibration process using mouse clicks to capture gaze targets.

        :param webcam: OpenCV VideoCapture object.
        :type webcam: cv2.VideoCapture
        :return: A CalibrationDataset object containing collected samples.
        :rtype: CalibrationDataset
        """
        print("\nCalibration started")

        self.capture_points = []
        self.current_target = None
        self.current_index = 0
        self.calibration_done = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while not self.calibration_done:
                calibration_frame = self._render_calibration_frame()

                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(self.window_name, calibration_frame)
                cv2.waitKey(1)

                success, img = webcam.read()
                if not success:
                    print("Error reading from the webcam.")
                    exit(1)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mp = face_mesh.process(img_rgb)

                if img_mp.multi_face_landmarks:
                    for face_landmarks in img_mp.multi_face_landmarks:
                        face_input, left_eye_input, right_eye_input, face_grid_input = (
                            self.gaze_tracker.extract_features(
                                img, face_landmarks, SCREEN_WIDTH, SCREEN_HEIGHT
                            )
                        )

                        if self.current_target:
                            user_x, user_y = self.current_target

                            match self.gaze_tracker.mp:
                                case "itracker_baseline.tar":
                                    gaze_x, gaze_y = pixels_to_gaze_cm(
                                        user_x, user_y, SCREEN_WIDTH, SCREEN_HEIGHT
                                    )
                                case "itracker_mpiiface.tar":
                                    gaze_x, gaze_y = normalize_MPIIFaceGaze(
                                        user_x, user_y, SCREEN_WIDTH, SCREEN_HEIGHT
                                    )
                                case _:
                                    raise ValueError("Invalid model_path")

                            self.capture_points.append(
                                (
                                    (face_input, left_eye_input, right_eye_input, face_grid_input),
                                    (gaze_x, gaze_y)
                                )
                            )

                            print(
                                f"Captured: Screen ({user_x}, {user_y}) "
                                f"→ Gaze ({gaze_x:.2f}, {gaze_y:.2f})"
                            )

                            self.current_target = None

                if len(self.capture_points) >= CALIBRATION_PTS:
                    self.calibration_done = True

        cv2.destroyWindow(self.window_name)
        print("\nCalibration completed.")

        return CalibrationDataset(self.capture_points)