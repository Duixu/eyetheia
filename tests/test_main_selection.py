import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

sys.modules.setdefault(
    "cv2",
    SimpleNamespace(
        EVENT_LBUTTONDOWN=1,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=16,
        WINDOW_NORMAL=0,
        destroyWindow=lambda window_name: None,
        getTextSize=lambda text, font, scale, thickness: ((len(text) * 8, 16), 0),
        imshow=lambda window_name, frame: None,
        namedWindow=lambda window_name, mode: None,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        setMouseCallback=lambda window_name, callback: None,
        waitKey=lambda delay: ord("q"),
    ),
)
sys.modules.setdefault(
    "mediapipe",
    SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=object))),
)

from main import COMPANY_SWIN_MODEL_ID, MODEL_EYETHEIA_BASELINE, _key_to_option


def test_key_to_option_returns_matching_option():
    options = [MODEL_EYETHEIA_BASELINE, COMPANY_SWIN_MODEL_ID]

    assert _key_to_option(ord("1"), options) == MODEL_EYETHEIA_BASELINE
    assert _key_to_option(ord("2"), options) == COMPANY_SWIN_MODEL_ID


def test_key_to_option_rejects_out_of_range_key():
    assert _key_to_option(ord("3"), [13, 9]) is None
    assert _key_to_option(ord("q"), [13, 9, 6]) is None
