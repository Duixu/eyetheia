# -*- coding: utf-8 -*-

"""
:mod:`OneEuroTuner` module
=========================

This module provides a lightweight OpenCV-based user interface to dynamically
tune the parameters of One Euro Filters during runtime.

It allows real-time adjustment of:
- freq
- mincutoff
- beta
- dcutoff

Default values match the EyeTheia runtime configuration:
freq = 30 Hz
mincutoff = 1.2
beta = 0.02
dcutoff = 1.0

:author: Pather Stevenson
:date: January 2026
"""

import cv2


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class OneEuroTuner:
    """
    OpenCV UI for tuning OneEuroFilter parameters in real time.
    """

    def __init__(self, window_name: str):

        self.window_name = window_name
        self._last_params = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 560, 220)

        # Default values aligned with GazeTracker filters
        cv2.createTrackbar("freq (Hz)", self.window_name, 30, 150, lambda x: None)
        cv2.createTrackbar("mincutoff x100", self.window_name, 150, 500, lambda x: None)
        cv2.createTrackbar("beta x1000", self.window_name, 20, 1000, lambda x: None)
        cv2.createTrackbar("dcutoff x100", self.window_name, 100, 500, lambda x: None)

    def read(self) -> tuple[float, float, float, float]:

        freq = _clamp(
            cv2.getTrackbarPos("freq (Hz)", self.window_name),
            1,
            120,
        )

        mincutoff = _clamp(
            cv2.getTrackbarPos("mincutoff x100", self.window_name) / 100.0,
            0.01,
            5.0,
        )

        beta = _clamp(
            cv2.getTrackbarPos("beta x1000", self.window_name) / 1000.0,
            0.0,
            1.0,
        )

        dcutoff = _clamp(
            cv2.getTrackbarPos("dcutoff x100", self.window_name) / 100.0,
            0.01,
            5.0,
        )

        return freq, mincutoff, beta, dcutoff

    def update_filters(self, filter_x, filter_y) -> tuple[float, float, float, float]:

        params = self.read()
        freq, mincutoff, beta, dcutoff = params

        if params != self._last_params:

            filter_x.setParameters(
                freq=freq,
                mincutoff=mincutoff,
                beta=beta,
                dcutoff=dcutoff,
            )
            filter_y.setParameters(
                freq=freq,
                mincutoff=mincutoff,
                beta=beta,
                dcutoff=dcutoff,
            )

            self._last_params = params

        return params