#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`main` module
==================

Main entry point for the gaze tracking system.

This module initializes the webcam, loads the gaze tracking model, and starts the tracking process.

:author: Pather Stevenson
:date: February 2025
"""

import cv2
import numpy as np
import os
from tracker.GazeTracker import GazeTracker
from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT


def select_calibration_point_count() -> int | None:
    """
    Shows a blocking OpenCV selection screen before the normal tracking flow starts.

    :return: Selected calibration point count, or None if the user quits.
    :rtype: int | None
    """
    window_name = "EyeTheia Calibration Setup"
    options = [13, 9, 6]
    selected_point_count = {"value": None}

    frame_width = min(900, SCREEN_WIDTH)
    frame_height = min(520, SCREEN_HEIGHT)
    button_width = 180
    button_height = 90
    gap = 45
    start_x = (frame_width - (button_width * len(options) + gap * (len(options) - 1))) // 2
    button_y = 235

    buttons = []
    for index, point_count in enumerate(options):
        x1 = start_x + index * (button_width + gap)
        y1 = button_y
        x2 = x1 + button_width
        y2 = y1 + button_height
        buttons.append((point_count, (x1, y1, x2, y2)))

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for point_count, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_point_count["value"] = point_count
                break

    def render_frame() -> np.ndarray:
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        cv2.putText(
            frame,
            "Select calibration point count",
            (80, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Click an option, or press 1 / 2 / 3. Press Q to quit.",
            (80, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        for index, (point_count, (x1, y1, x2, y2)) in enumerate(buttons, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(
                frame,
                f"{point_count}",
                (x1 + 58, y1 + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Key {index}",
                (x1 + 54, y2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        return frame

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while selected_point_count["value"] is None:
        cv2.imshow(window_name, render_frame())
        key = cv2.waitKey(30) & 0xFF

        if key == ord("1"):
            selected_point_count["value"] = 13
        elif key == ord("2"):
            selected_point_count["value"] = 9
        elif key == ord("3"):
            selected_point_count["value"] = 6
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return selected_point_count["value"]

def main() -> None:
    """
    Main function to start the gaze tracking system.

    - Retrieves the webcam URL from environment variables.
    - Initializes the webcam (local or network stream).
    - Starts the gaze tracking process.
    """
    calibration_point_count = select_calibration_point_count()
    if calibration_point_count is None:
        print("Calibration setup cancelled.")
        return

    print(f"Selected {calibration_point_count} calibration points.")

    # Retrieve the webcam URL from environment variables
    webcam_url: str = os.getenv("WEBCAM_URL", "0")  # Default to "0" for local webcam

    # Use the URL if provided; otherwise, default to the local webcam
    webcam = cv2.VideoCapture(webcam_url if webcam_url != "0" else 0)

    if not webcam.isOpened():
        print("Unable to open webcam. Please check your device or URL.")
        return

    gaze_tracker = GazeTracker(
        model_path="itracker_baseline.tar",
        calibration_point_count=calibration_point_count,
    )

    try:
        gaze_tracker.run(webcam)
    except KeyboardInterrupt:
        print("\nTracking stopped by user.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
