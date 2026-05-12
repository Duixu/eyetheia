"""Company gaze model integration."""

from .company_gaze_tracker import (
    COMPANY_CALIBRATION_ARC,
    COMPANY_CALIBRATION_MAPPER,
    COMPANY_CONFIRMATION_CLICK,
    COMPANY_CONFIRMATION_DWELL,
    COMPANY_SWIN_MODEL_ID,
    DEFAULT_COMPANY_GAZE_WEIGHT_PATH,
    CompanyGazePrediction,
    CompanyGazeTracker,
)

__all__ = [
    "COMPANY_CALIBRATION_ARC",
    "COMPANY_CALIBRATION_MAPPER",
    "COMPANY_CONFIRMATION_CLICK",
    "COMPANY_CONFIRMATION_DWELL",
    "COMPANY_SWIN_MODEL_ID",
    "DEFAULT_COMPANY_GAZE_WEIGHT_PATH",
    "CompanyGazePrediction",
    "CompanyGazeTracker",
]
