# -*- coding: utf-8 -*-

"""
:mod:`OneEuroTuner` module
=========================

This module provides a lightweight OpenCV-based user interface to dynamically
tune the parameters of One Euro Filters during runtime.

It is designed to be used alongside a fullscreen gaze visualization window,
while keeping all UI-related logic (trackbars, parameter reading) separated
from the core gaze tracking logic.

The tuner allows real-time adjustment of:
- One Euro Filter parameters (freq, mincutoff, beta, dcutoff)
- An additional exponential smoothing factor (alpha)

This module does not perform any filtering by itself; it only updates
existing OneEuroFilter instances passed by the caller.

:author: Pather Stevenson
:date: January 2026
"""

import cv2


def _clamp(v: float, lo: float, hi: float) -> float:
    """
    Clamps a value between a lower and an upper bound.

    :param v: Input value.
    :type v: float
    :param lo: Lower bound.
    :type lo: float
    :param hi: Upper bound.
    :type hi: float
    :return: Clamped value.
    :rtype: float
    """
    return max(lo, min(hi, v))


class OneEuroTuner:
    """
    UI helper class for live tuning of One Euro Filter parameters using OpenCV trackbars.

    This class manages a dedicated OpenCV window containing sliders that allow
    real-time modification of One Euro Filter parameters without resetting their
    internal state.

    The window is intended to remain separate from the fullscreen visualization
    window to ensure stable interaction and focus handling.
    """

    def __init__(self, window_name: str):
        """
        Initializes the One Euro tuning UI and creates the associated OpenCV trackbars.

        :param window_name: Name of the OpenCV window used for controls.
        :type window_name: str
        """
        self.window_name = window_name
        self._last_params = None
        self._alpha = 0.2

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 560, 260)

        cv2.createTrackbar("freq (Hz)", self.window_name, 10, 120, lambda x: None)
        cv2.createTrackbar("mincutoff x100", self.window_name, 70, 500, lambda x: None)
        cv2.createTrackbar("beta x1000", self.window_name, 7, 1000, lambda x: None)
        cv2.createTrackbar("dcutoff x100", self.window_name, 100, 500, lambda x: None)
        cv2.createTrackbar("alpha x100", self.window_name, 20, 100, lambda x: None)

    def read(self) -> tuple:
        """
        Reads the current values from the OpenCV trackbars and converts them
        to meaningful floating-point parameters.

        :return: Tuple containing (freq, mincutoff, beta, dcutoff, alpha).
        :rtype: tuple[float, float, float, float, float]
        """
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

        self._alpha = _clamp(
            cv2.getTrackbarPos("alpha x100", self.window_name) / 100.0,
            0.0,
            1.0,
        )

        return freq, mincutoff, beta, dcutoff, self._alpha

    def update_filters(self, filter_x, filter_y) -> tuple:
        """
        Updates the parameters of two OneEuroFilter instances if the UI values
        have changed since the last call.

        The internal state of the filters is preserved (no reset).

        :param filter_x: OneEuroFilter instance for the X coordinate.
        :param filter_y: OneEuroFilter instance for the Y coordinate.
        :return: Tuple containing the smoothing alpha and the active filter parameters.
        :rtype: tuple[float, tuple[float, float, float, float]]
        """
        freq, mincutoff, beta, dcutoff, alpha = self.read()
        params = (freq, mincutoff, beta, dcutoff)

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

        return alpha, params

    @property
    def alpha(self) -> float:
        """
        Returns the current additional exponential smoothing factor.

        :return: Smoothing alpha in [0, 1].
        :rtype: float
        """
        return self._alpha
