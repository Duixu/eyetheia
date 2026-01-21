#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`calibration_ws` module
============================

This module defines the WebSocket-based calibration endpoint for EyeTheia.

This WebSocket interface allows incremental transmission of calibration samples (image + landmarks + screen
coordinates), real-time progress feedback, and controlled triggering of the
fine-tuning procedure once all required calibration points have been received.

Each WebSocket connection maintains its own isolated calibration state to avoid
cross-session interference. Collected calibration samples are processed using
the currently injected :class:`GazeTracker` instance.

Message protocol (JSON-based):
------------------------------
Client → Server:
    - ``{"type": "calib_start", "screen": {"w": int, "h": int}}``
    - ``{"type": "calib_point", "i": int, "x_pixel": float, "y_pixel": float,
          "image_base64": str, "landmarks": list[dict]}``

Server → Client:
    - ``{"type": "ready", "expected": int}``
    - ``{"type": "ack", "i": int, "count": int, "total": int}``
    - ``{"type": "progress", "stage": str}``
    - ``{"type": "result", "message": str, "total_points": int}``
    - ``{"type": "error", "detail": str}``

Stages emitted during calibration:
----------------------------------
    - ``reset_model``
    - ``before_eval``
    - ``training``
    - ``after_eval``

:author: Pather Stevenson
:date: February 2026
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import json

from tracker.CalibrationDataset import CalibrationDataset
from utils.utils import decode_image_bytes, FaceLandmarks, pixels_to_gaze_cm, normalize_MPIIFaceGaze
from routes.dependency import get_tracker, get_capture_points, get_screen
from utils.ws_codec import unpack_ws_message


router = APIRouter()

@router.websocket("/ws/calibration")
async def ws_calibration(
    ws: WebSocket,
    gaze_tracker=Depends(get_tracker),
    capture_points=Depends(get_capture_points),
    screen=Depends(get_screen),
):
    await ws.accept()

    local_points = []
    screen_width, screen_height = screen

    try:
        while True:
            data = await ws.receive_bytes()
            meta, img_bytes = unpack_ws_message(data)
            msg_type = meta.get("type")

            if msg_type == "calib_start":
                scr = meta.get("screen")
                if scr:
                    screen_width = int(scr["w"])
                    screen_height = int(scr["h"])
                local_points.clear()
                await ws.send_text(json.dumps({"type":"ready","expected":17}))
                continue

            if msg_type != "calib_point":
                await ws.send_text(json.dumps({"type":"error","detail":f"Unknown message type: {msg_type}"}))
                continue

            index = int(meta["i"])
            x_pixel = meta["x_pixel"]
            y_pixel = meta["y_pixel"]
            landmarks_data = meta["landmarks"]

            try:
                img = decode_image_bytes(img_bytes)
            except ValueError as e:
                await ws.send_text(json.dumps({"type":"error","detail":str(e),"i":index}))
                continue

            face_landmarks = FaceLandmarks(landmarks_data)
            face_input, left_eye_input, right_eye_input, face_grid_input = gaze_tracker.extract_features(
                img, face_landmarks, screen_width, screen_height
            )

            match gaze_tracker.mp:
                case "itracker_baseline.tar":
                    x, y = pixels_to_gaze_cm(x_pixel, y_pixel, screen_width, screen_height)
                case "itracker_mpiiface.tar":
                    x, y = normalize_MPIIFaceGaze(x_pixel, y_pixel, screen_width, screen_height)
                case _:
                    await ws.send_text(json.dumps({"type":"error","detail":"Invalid model path"}))
                    continue

            local_points.append(((face_input, left_eye_input, right_eye_input, face_grid_input), (x, y)))

            await ws.send_text(json.dumps({"type":"ack","i":index,"count":len(local_points),"total":17}))

            if len(local_points) == 17:
                await ws.send_text(json.dumps({"type":"progress","stage":"reset_model"}))
                gaze_tracker.reset_model()

                await ws.send_text(json.dumps({"type":"progress","stage":"before_eval"}))
                gaze_tracker.calibration.set_capture_points(local_points)
                gaze_tracker.calibration.evaluate_calibration_accuracy()

                await ws.send_text(json.dumps({"type":"progress","stage":"training"}))
                gaze_tracker.train(CalibrationDataset(local_points))

                await ws.send_text(json.dumps({"type":"progress","stage":"after_eval"}))
                gaze_tracker.calibration.evaluate_calibration_accuracy()

                capture_points.clear()
                capture_points.extend(local_points)

                await ws.send_text(json.dumps({"type":"result","message":"Calibration completed","total_points":17}))

    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type":"error","detail":str(e)}))
