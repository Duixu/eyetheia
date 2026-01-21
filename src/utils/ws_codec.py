#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`ws_codec` module
======================

Binary WebSocket framing utilities.

Frame format:
    [meta_len: uint32 BE][meta_json: utf-8][payload_bytes]

:author: Pather Stevenson
:date: January 2026
"""

import json
from typing import Tuple, Dict, Any


def unpack_ws_message(data: bytes) -> Tuple[Dict[str, Any], bytes]:
    """
    Unpack a binary WebSocket message into (meta, payload_bytes).

    :param data: Raw WebSocket binary message.
    :type data: bytes
    :return: Tuple (meta dict, payload bytes).
    :rtype: tuple[dict, bytes]
    :raises ValueError: If message is malformed.
    """
    if len(data) < 4:
        raise ValueError("WS message too short (missing meta length header)")

    meta_len = int.from_bytes(data[:4], "big")
    if meta_len <= 0:
        raise ValueError("Invalid meta length")

    start = 4
    end = start + meta_len
    if end > len(data):
        raise ValueError("Meta length exceeds message size")

    meta = json.loads(data[start:end].decode("utf-8"))
    payload = data[end:]
    return meta, payload
