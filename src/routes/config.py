#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`config` module
=====================

This module defines configuration endpoints for the gaze tracking API.
It exposes endpoints to update the screen resolution used for
coordinate normalization and gaze prediction, and to toggle
whether gaze predictions should be filtered with One Euro.

:author: Pather Stevenson
:date: October 2025
"""

from fastapi import APIRouter, Form, Depends
from routes.dependency import get_screen, get_tracker

router = APIRouter(prefix="/config", tags=['config'])


@router.post("/update_screen")
async def update_screen_size(
    width: int = Form(...),
    height: int = Form(...),
    screen=Depends(get_screen),
):
    """
    Update the screen size (width, height) used by gaze normalization.

    :param width: New screen width in pixels.
    :param height: New screen height in pixels.
    :type width: int
    :type height: int

    :return: Confirmation message with updated values.
    :rtype: dict
    """
    if isinstance(screen, tuple):
        screen = list(screen)

    screen[0] = width
    screen[1] = height

    return {"message": "Screen size updated", "width": width, "height": height}


@router.post("/set_gaze_filtered")
async def set_gaze_filtered(
    enabled: bool = Form(...),
    gaze_tracker=Depends(get_tracker),
):
    """
    Enable or disable One Euro filtering for real-time gaze output.

    :param enabled: True to enable filtering, False to disable it.
    :type enabled: bool

    :return: Confirmation message with current filtering state.
    :rtype: dict
    """
    gaze_tracker.gaze_filtered = enabled

    if hasattr(gaze_tracker, "reset_gaze_filters"):
        gaze_tracker.reset_gaze_filters()

    return {
        "message": "gaze_filtered updated",
        "gaze_filtered": gaze_tracker.gaze_filtered
    }


@router.get("/gaze_filtered")
async def get_gaze_filtered(
    gaze_tracker=Depends(get_tracker),
):
    """
    Return whether One Euro filtering is enabled for real-time gaze output.

    :return: Current filtering state.
    :rtype: dict
    """
    return {
        "gaze_filtered": getattr(gaze_tracker, "gaze_filtered", True)
    }