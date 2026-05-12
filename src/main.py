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
from typing import Sequence, TypeVar

from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT



CALIBRATION_ORIGINAL_CLICK = "original_click"
CALIBRATION_COMPANY_GAZE = "company_gaze"
MODEL_EYETHEIA_BASELINE = "eyetheia_baseline"
COMPANY_SWIN_MODEL_ID = "company_swin"
T = TypeVar("T")


def _key_to_option(key: int, options: Sequence[T]) -> T | None:
    if ord("1") <= key <= ord("9"):
        index = key - ord("1")
        if index < len(options):
            return options[index]
    return None


def select_startup_option(
    window_name: str,
    title: str,
    options: Sequence[tuple[T, str]],
) -> T | None:
    """
    Shows a blocking OpenCV option selection screen.

    :return: Selected option value, or None if the user quits.
    """
    selected = {"value": None}
    option_values = [value for value, _ in options]

    frame_width = min(980, SCREEN_WIDTH)
    frame_height = min(520, SCREEN_HEIGHT)
    button_width = 240 if len(options) <= 2 else 180
    button_height = 90
    gap = 45 if len(options) <= 3 else 25
    start_x = (frame_width - (button_width * len(options) + gap * (len(options) - 1))) // 2
    button_y = 235

    buttons = []
    for index, (value, label) in enumerate(options):
        x1 = start_x + index * (button_width + gap)
        y1 = button_y
        x2 = x1 + button_width
        y2 = y1 + button_height
        buttons.append((value, label, (x1, y1, x2, y2)))

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for value, _, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected["value"] = value
                break

    def render_frame() -> np.ndarray:
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        cv2.putText(
            frame,
            title,
            (80, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Click an option, or press the matching number key. Press Q to quit.",
            (80, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        for index, (_, label, (x1, y1, x2, y2)) in enumerate(buttons, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            text_x = x1 + max(12, (button_width - text_size[0]) // 2)
            text_y = y1 + 55
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
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

    while selected["value"] is None:
        cv2.imshow(window_name, render_frame())
        key = cv2.waitKey(30) & 0xFF

        key_selection = _key_to_option(key, option_values)
        if key_selection is not None:
            selected["value"] = key_selection
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return selected["value"]


def select_model() -> str | None:
    return select_startup_option(
        window_name="EyeTheia Model Setup",
        title="Select gaze model",
        options=[
            (MODEL_EYETHEIA_BASELINE, "EyeTheia"),
            (COMPANY_SWIN_MODEL_ID, "Company"),
        ],
    )


def select_calibration_workflow() -> str | None:
    return select_startup_option(
        window_name="EyeTheia Calibration Mode",
        title="Select calibration mode",
        options=[
            (CALIBRATION_ORIGINAL_CLICK, "EyeTheia click"),
            (CALIBRATION_COMPANY_GAZE, "Company dwell"),
        ],
    )


def select_calibration_point_count() -> int | None:
    return select_startup_option(
        window_name="EyeTheia Calibration Setup",
        title="Select calibration point count",
        options=[
            (13, "13"),
            (9, "9"),
            (6, "6"),
        ],
    )

#新增整合成一次
def select_startup_config() -> tuple[str, str, int] | None:
    """
    Shows a single OpenCV startup configuration screen.

    Allows the user to select:
    1. Gaze model
    2. Calibration workflow
    3. Calibration point count

    :return: (selected_model, calibration_workflow, calibration_point_count), or None if cancelled.
    """
    window_name = "EyeTheia Startup Setup"

    frame_width = min(1500, SCREEN_WIDTH)
    frame_height = min(1000, SCREEN_HEIGHT)

    selected = {
        "model": MODEL_EYETHEIA_BASELINE,
        "workflow": CALIBRATION_ORIGINAL_CLICK,
        "points": 13,
        "confirmed": False,
        "cancelled": False,
    }

    groups = [
        (
            "model",
            "1. Select gaze model",
            [
                (MODEL_EYETHEIA_BASELINE, "EyeTheia"),
                (COMPANY_SWIN_MODEL_ID, "Company"),
            ],
        ),
        (
            "workflow",
            "2. Select calibration mode",
            [
                (CALIBRATION_ORIGINAL_CLICK, "click"),
                (CALIBRATION_COMPANY_GAZE, "gaze"),
            ],
        ),
        (
            "points",
            "3. Select calibration point count",
            [
                (13, "13 points"),
                (9, "9 points"),
                (6, "6 points"),
            ],
        ),
    ]

    buttons = []
    start_y = 200
    row_gap = 190
    button_width = 280
    button_height = 80
    button_gap = 50

    for row_index, (group_key, group_title, options) in enumerate(groups):
        y1 = start_y + row_index * row_gap + 45
        total_width = len(options) * button_width + (len(options) - 1) * button_gap
        start_x = (frame_width - total_width) // 2

        for option_index, (value, label) in enumerate(options):
            x1 = start_x + option_index * (button_width + button_gap)
            x2 = x1 + button_width
            y2 = y1 + button_height
            buttons.append((group_key, value, label, (x1, y1, x2, y2)))

    confirm_button = (frame_width // 2 - 180, frame_height - 105, frame_width // 2 - 20, frame_height - 45)
    cancel_button = (frame_width // 2 + 20, frame_height - 105, frame_width // 2 + 180, frame_height - 45)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for group_key, value, _, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected[group_key] = value
                return

        cx1, cy1, cx2, cy2 = confirm_button
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            selected["confirmed"] = True
            return

        qx1, qy1, qx2, qy2 = cancel_button
        if qx1 <= x <= qx2 and qy1 <= y <= qy2:
            selected["cancelled"] = True
            return

    def draw_button(
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        label: str,
        active: bool = False,
    ) -> None:
        x1, y1, x2, y2 = rect

        if active:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 235, 210), -1)
            border_color = (0, 120, 0)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)
            border_color = (0, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        text_x = x1 + max(10, (x2 - x1 - text_size[0]) // 2)
        text_y = y1 + 38

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def render_frame() -> np.ndarray:
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        cv2.putText(
            frame,
            "EyeTheia Startup Setup",
            (70, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Select all options on this page, then click Start. Press Q to quit.",
            (70, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        for row_index, (group_key, group_title, _) in enumerate(groups):
            title_y = start_y + row_index * row_gap + 20
            cv2.putText(
                frame,
                group_title,
                (80, title_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        for group_key, value, label, rect in buttons:
            active = selected[group_key] == value
            draw_button(frame, rect, label, active)

        draw_button(frame, confirm_button, "Start", active=True)
        draw_button(frame, cancel_button, "Cancel", active=False)

        summary = (
            f"Current: model={selected['model']}, "
            f"mode={selected['workflow']}, "
            f"points={selected['points']}"
        )
        cv2.putText(
            frame,
            summary,
            (70, frame_height - 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

        return frame

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #新增改变窗口大小
    cv2.resizeWindow(window_name, frame_width, frame_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    while not selected["confirmed"]:
        if selected["cancelled"]:
            cv2.destroyWindow(window_name)
            return None

        cv2.imshow(window_name, render_frame())
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return None
        elif key == 13 or key == 10:
            selected["confirmed"] = True

    cv2.destroyWindow(window_name)

    return (
        selected["model"],
        selected["workflow"],
        selected["points"],
    )
def main() -> None:
    """
    Main function to start the gaze tracking system.

    - Retrieves the webcam URL from environment variables.
    - Initializes the webcam (local or network stream).
    - Starts the gaze tracking process.
    """
    # selected_model = select_model()
    # if selected_model is None:
    #     print("Model setup cancelled.")
    #     return

    # calibration_workflow = select_calibration_workflow()
    # if calibration_workflow is None:
    #     print("Calibration mode setup cancelled.")
    #     return

    # calibration_point_count = select_calibration_point_count()
    # if calibration_point_count is None:
    #     print("Calibration setup cancelled.")
    #     return
    #新增main() 里面原来的三次选择整合成一次
    startup_config = select_startup_config()
    if startup_config is None:
        print("Startup setup cancelled.")
        return

    selected_model, calibration_workflow, calibration_point_count = startup_config


    print(f"Selected model: {selected_model}.")
    print(f"Selected calibration mode: {calibration_workflow}.")
    print(f"Selected {calibration_point_count} calibration points.")

    # Retrieve the webcam URL from environment variables
    webcam_url: str = os.getenv("WEBCAM_URL", "0")  # Default to "0" for local webcam

    # Use the URL if provided; otherwise, default to the local webcam
    webcam = cv2.VideoCapture(webcam_url if webcam_url != "0" else 0)

    if not webcam.isOpened():
        print("Unable to open webcam. Please check your device or URL.")
        return

    if selected_model == MODEL_EYETHEIA_BASELINE and calibration_workflow == CALIBRATION_ORIGINAL_CLICK:
        from tracker.GazeTracker import GazeTracker

        gaze_tracker = GazeTracker(
            model_path="itracker_baseline.tar",
            calibration_point_count=calibration_point_count,
        )
    elif selected_model == MODEL_EYETHEIA_BASELINE and calibration_workflow == CALIBRATION_COMPANY_GAZE:
        from company_gaze import CompanyGazeTracker
        from tracker.GazeTracker import GazeTracker

        eyetheia_finetune_tracker = GazeTracker(
            model_path="itracker_baseline.tar",
            calibration_point_count=calibration_point_count,
        )
        gaze_tracker = CompanyGazeTracker(
            calibration_point_count=calibration_point_count,
            eyetheia_finetune_tracker=eyetheia_finetune_tracker,
            calibration_confirmation="dwell",
            run_eyetheia_after_calibration=True,
        )
    elif selected_model == COMPANY_SWIN_MODEL_ID and calibration_workflow == CALIBRATION_ORIGINAL_CLICK:
        from company_gaze import CompanyGazeTracker

        gaze_tracker = CompanyGazeTracker(
            calibration_point_count=calibration_point_count,
            calibration_mode="arc",
            calibration_confirmation="click",
        )
    elif selected_model == COMPANY_SWIN_MODEL_ID and calibration_workflow == CALIBRATION_COMPANY_GAZE:
        from company_gaze import CompanyGazeTracker

        gaze_tracker = CompanyGazeTracker(
            calibration_point_count=calibration_point_count,
            calibration_mode="arc",
            calibration_confirmation="dwell",
        )
    else:
        raise ValueError(
            f"Unknown model/calibration workflow: {selected_model}/{calibration_workflow}"
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
