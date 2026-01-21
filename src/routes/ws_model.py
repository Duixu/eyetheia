#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`model_ws` module
======================

This module defines the WebSocket-based real-time gaze prediction endpoint.

It enables low-latency, continuous gaze estimation by maintaining a persistent
connection between the client and the server. Each incoming frame (image +
facial landmarks) is processed independently using the injected
:class:`GazeTracker` instance, and the predicted gaze coordinates are streamed
back to the client in screen pixel space.

Message protocol (JSON-based):
------------------------------
Client → Server:
    - ``{"type": "screen", "w": int, "h": int}``
    - ``{"type": "frame", "image_base64": str, "landmarks": list[dict]}``

Server → Client:
    - ``{"type": "screen_ack", "w": int, "h": int}``
    - ``{"type": "pred", "x_px": float, "y_px": float}``
    - ``{"type": "error", "detail": str}``

:author: Pather Stevenson
:date: February 2026
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import json

from routes.dependency import get_tracker, get_screen
from utils.utils import FaceLandmarks, decode_image_bytes, denormalized_MPIIFaceGaze, gaze_cm_to_pixels
from utils.ws_codec import unpack_ws_message

router = APIRouter()

@router.websocket("/ws/predict_gaze")
async def ws_predict_gaze(
    ws: WebSocket,
    gaze_tracker=Depends(get_tracker),
    screen=Depends(get_screen),
):
    await ws.accept()
    screen_width, screen_height = screen

    try:
        while True:
            data = await ws.receive_bytes()
            meta, img_bytes = unpack_ws_message(data)

            msg_type = meta.get("type")
            if msg_type == "screen":
                screen_width = int(meta["w"])
                screen_height = int(meta["h"])
                await ws.send_text(json.dumps({
                    "type": "screen_ack",
                    "w": screen_width,
                    "h": screen_height
                }))
                continue

            if msg_type != "frame":
                await ws.send_text(json.dumps({
                    "type": "error",
                    "detail": f"Unknown message type: {msg_type}"
                }))
                continue

            try:
                img = decode_image_bytes(img_bytes)
            except ValueError as e:
                await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                continue

            face_landmarks = FaceLandmarks(meta["landmarks"])

            face_input, left_eye_input, right_eye_input, face_grid_input = (
                gaze_tracker.extract_features(img, face_landmarks, screen_width, screen_height)
            )

            pred = gaze_tracker.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)
            if not isinstance(pred, (list, tuple)) or len(pred) < 2:
                await ws.send_text(json.dumps({"type": "error", "detail": "Model returned invalid gaze vector"}))
                continue

            match gaze_tracker.mp:
                case "itracker_baseline.tar":
                    x_px, y_px = gaze_cm_to_pixels(pred[0], pred[1], screen_width, screen_height)
                case "itracker_mpiiface.tar":
                    x_px, y_px = denormalized_MPIIFaceGaze(pred[0], pred[1], screen_width, screen_height)
                case _:
                    await ws.send_text(json.dumps({"type":"error","detail":"Invalid model path"}))
                    continue

            await ws.send_text(json.dumps({"type":"pred","x_px": float(x_px), "y_px": float(y_px)}))

    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))