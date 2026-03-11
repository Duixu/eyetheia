#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`model_ws` module
======================

This module defines the WebSocket-based real-time gaze prediction endpoint.

It enables low-latency, continuous gaze estimation by maintaining a persistent
connection between the client and the server. Incoming frames (JPEG bytes +
facial landmarks) are processed using the injected :class:`GazeTracker` instance,
and predicted gaze coordinates are streamed back to the client in screen pixel
space.

To preserve real-time behavior under load, the server uses a "latest-only"
buffer: if new frames arrive while one is still pending, the pending frame is
overwritten and older frames are effectively dropped.

Message protocol:
-----------------
Client → Server (binary):
    Each message is a binary payload containing:
        - a JSON-encoded metadata header (``meta``)
        - optional JPEG image bytes (``image_bytes``)

    Supported ``meta`` messages:
        - ``{"type": "screen", "w": int, "h": int}``
        - ``{"type": "frame", "landmarks": list[dict]}`` + JPEG bytes payload

Server → Client (text / JSON):
    - ``{"type": "screen_ack", "w": int, "h": int}``
    - ``{"type": "pred", "x_px": float, "y_px": float}``
    - ``{"type": "error", "detail": str}``

:author: Pather Stevenson
:date: February 2026
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import json
import asyncio
from typing import Optional, Tuple
import time

from routes.dependency import get_tracker, get_screen
from utils.utils import FaceLandmarks, decode_image_bytes, denormalized_MPIIFaceGaze, gaze_cm_to_pixels
from utils.ws_codec import unpack_ws_message

router = APIRouter()


def _process_frame_sync(
    meta: dict,
    img_bytes: bytes,
    gaze_tracker,
    screen_width: int,
    screen_height: int,
    start_time: float,
) -> Tuple[float, float]:
    """
    Synchronous heavy pipeline executed off the event loop (thread).
    Returns (x_px, y_px) in screen pixel space.
    """
    img = decode_image_bytes(img_bytes)
    face_landmarks = FaceLandmarks(meta["landmarks"])

    face_input, left_eye_input, right_eye_input, face_grid_input = (
        gaze_tracker.extract_features(img, face_landmarks, screen_width, screen_height)
    )

    pred = gaze_tracker.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)
    if not isinstance(pred, (list, tuple)) or len(pred) < 2:
        raise ValueError("Model returned invalid gaze vector")

    if gaze_tracker.mp == "itracker_baseline.tar":
        x_px, y_px = gaze_cm_to_pixels(pred[0], pred[1], screen_width, screen_height)
    elif gaze_tracker.mp == "itracker_mpiiface.tar":
        x_px, y_px = denormalized_MPIIFaceGaze(pred[0], pred[1], screen_width, screen_height)
    else:
        raise ValueError("Invalid model path")

    if getattr(gaze_tracker, "gaze_filtered", True):
        timestamp = time.perf_counter() - start_time
        x_px, y_px = gaze_tracker.filter_gaze_pixels(x_px, y_px, timestamp)

    return float(x_px), float(y_px)


@router.websocket("/ws/predict_gaze")
async def ws_predict_gaze(
    ws: WebSocket,
    gaze_tracker=Depends(get_tracker),
    screen=Depends(get_screen),
):
    await ws.accept()
    screen_width, screen_height = screen
    start_time = time.perf_counter()

    # Latest-only buffer for frames. If a new frame arrives while one is pending,
    # we overwrite the pending frame (real-time behavior).
    latest_frame: Optional[Tuple[dict, bytes]] = None
    latest_lock = asyncio.Lock()
    new_frame_evt = asyncio.Event()

    async def receiver_loop():
        nonlocal screen_width, screen_height, latest_frame
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

                # Store only the most recent frame (overwrite previous if any).
                async with latest_lock:
                    latest_frame = (meta, img_bytes)
                    new_frame_evt.set()

        except WebSocketDisconnect:
            # Let outer scope handle shutdown.
            raise

    async def worker_loop():
        nonlocal latest_frame
        while True:
            # Wait until at least one frame is available.
            await new_frame_evt.wait()

            # Grab the latest frame and clear the event.
            async with latest_lock:
                frame = latest_frame
                latest_frame = None
                new_frame_evt.clear()

            if frame is None:
                continue

            meta, img_bytes = frame

            try:
                # Offload heavy CPU/ML work to a thread to avoid blocking the event loop.
                x_px, y_px = await asyncio.to_thread(
                    _process_frame_sync,
                    meta,
                    img_bytes,
                    gaze_tracker,
                    screen_width,
                    screen_height,
                    start_time,
                )

                await ws.send_text(json.dumps({
                    "type": "pred",
                    "x_px": x_px,
                    "y_px": y_px,
                }))

            except ValueError as e:
                await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))

    try:
        recv_task = asyncio.create_task(receiver_loop())
        work_task = asyncio.create_task(worker_loop())

        done, pending = await asyncio.wait(
            {recv_task, work_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for t in pending:
            t.cancel()

        # Propagate exception if any.
        for t in done:
            exc = t.exception()
            if exc:
                raise exc

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
        except Exception:
            pass
