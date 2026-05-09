"""
Map company gaze angle outputs to EyeTheia-style screen coordinates.

The company gaze pipeline exposes pitch/yaw angles, while EyeTheia compares
models in screen pixel space.  This mapper learns a small calibration layer:

    (pitch, yaw) -> (x_px, y_px)

Use the same calibration targets as EyeTheia, collect the company model output
at each target, fit the mapper, then run both models against screen pixels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CompanyGazeSample:
    """One calibration pair from company model output to a screen target."""

    pitch: float
    yaw: float
    x_px: float
    y_px: float


class CompanyGazeMapper:
    """
    Calibrates company pitch/yaw outputs into screen pixel coordinates.

    Parameters
    ----------
    degree:
        Polynomial degree for the mapping.  Use 1 for linear, 2 for the
        default quadratic mapping.  Degree 2 needs at least 6 samples.
    pitch_yaw_unit:
        Unit used by incoming company outputs.  The public company API returns
        degrees, so the default is "deg".  Use "rad" if you tap the raw model
        output before the company's degree conversion.
    ridge:
        Small L2 regularization strength for a stable least-squares fit.
    screen_size:
        Optional (width, height).  If present, predictions are clipped to the
        visible screen area.
    """

    def __init__(
        self,
        degree: int = 2,
        pitch_yaw_unit: str = "deg",
        ridge: float = 1e-6,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        if pitch_yaw_unit not in ("deg", "rad"):
            raise ValueError('pitch_yaw_unit must be "deg" or "rad"')
        if ridge < 0:
            raise ValueError("ridge must be non-negative")

        self.degree = degree
        self.pitch_yaw_unit = pitch_yaw_unit
        self.ridge = float(ridge)
        self.screen_size = screen_size
        self.samples: list[CompanyGazeSample] = []
        self.coefficients: np.ndarray | None = None  # shape: (features, 2)

    @property
    def fitted(self) -> bool:
        return self.coefficients is not None

    @property
    def feature_count(self) -> int:
        return 3 if self.degree == 1 else 6

    def add_sample(self, pitch: float, yaw: float, x_px: float, y_px: float) -> None:
        """Add one calibration sample."""
        values = (pitch, yaw, x_px, y_px)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("sample values must be finite numbers")
        self.samples.append(
            CompanyGazeSample(
                pitch=float(pitch),
                yaw=float(yaw),
                x_px=float(x_px),
                y_px=float(y_px),
            )
        )

    def extend_samples(
        self,
        samples: Iterable[CompanyGazeSample | Sequence[float] | dict],
    ) -> None:
        """Add many samples.

        Accepted forms:
        - CompanyGazeSample
        - (pitch, yaw, x_px, y_px)
        - {"pitch": ..., "yaw": ..., "x_px": ..., "y_px": ...}
        """
        for sample in samples:
            if isinstance(sample, CompanyGazeSample):
                self.add_sample(sample.pitch, sample.yaw, sample.x_px, sample.y_px)
            elif isinstance(sample, dict):
                self.add_sample(
                    sample["pitch"],
                    sample["yaw"],
                    sample.get("x_px", sample.get("x")),
                    sample.get("y_px", sample.get("y")),
                )
            else:
                pitch, yaw, x_px, y_px = sample
                self.add_sample(pitch, yaw, x_px, y_px)

    def fit(self) -> np.ndarray:
        """Fit the calibration mapping and return coefficients."""
        if len(self.samples) < self.feature_count:
            raise ValueError(
                f"degree {self.degree} mapping requires at least "
                f"{self.feature_count} samples, got {len(self.samples)}"
            )

        angles = np.array([(s.pitch, s.yaw) for s in self.samples], dtype=np.float64)
        targets = np.array([(s.x_px, s.y_px) for s in self.samples], dtype=np.float64)
        design = self._design_matrix(angles[:, 0], angles[:, 1])

        if self.ridge == 0:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        else:
            reg = self.ridge * np.eye(design.shape[1], dtype=np.float64)
            reg[0, 0] = 0.0  # Do not regularize the intercept.
            coeffs = np.linalg.solve(design.T @ design + reg, design.T @ targets)

        self.coefficients = coeffs
        return coeffs.copy()

    def predict(self, pitch: float, yaw: float, clip: bool = True) -> tuple[float, float]:
        """Predict screen pixel coordinates from pitch/yaw."""
        if self.coefficients is None:
            raise RuntimeError("CompanyGazeMapper must be fitted or loaded before predict()")

        design = self._design_matrix(
            np.array([float(pitch)], dtype=np.float64),
            np.array([float(yaw)], dtype=np.float64),
        )
        x_px, y_px = (design @ self.coefficients)[0]

        if clip and self.screen_size is not None:
            width, height = self.screen_size
            x_px = float(np.clip(x_px, 0, max(width - 1, 0)))
            y_px = float(np.clip(y_px, 0, max(height - 1, 0)))

        return float(x_px), float(y_px)

    def predict_from_gaze(self, gaze: dict, clip: bool = True) -> tuple[float, float]:
        """Predict from a company API dict containing pitch/yaw fields."""
        return self.predict(float(gaze["pitch"]), float(gaze["yaw"]), clip=clip)

    def residuals(self) -> np.ndarray:
        """Return per-sample prediction errors in pixels."""
        if self.coefficients is None:
            raise RuntimeError("CompanyGazeMapper must be fitted before residuals()")
        angles = np.array([(s.pitch, s.yaw) for s in self.samples], dtype=np.float64)
        targets = np.array([(s.x_px, s.y_px) for s in self.samples], dtype=np.float64)
        preds = self._design_matrix(angles[:, 0], angles[:, 1]) @ self.coefficients
        return preds - targets

    def mean_error_px(self) -> float:
        """Return mean Euclidean calibration error in screen pixels."""
        errors = self.residuals()
        return float(np.linalg.norm(errors, axis=1).mean())

    def save(self, path: str | Path) -> None:
        """Save mapper configuration, samples, and fitted coefficients as JSON."""
        if self.coefficients is None:
            raise RuntimeError("fit the mapper before saving it")

        payload = {
            "degree": self.degree,
            "pitch_yaw_unit": self.pitch_yaw_unit,
            "ridge": self.ridge,
            "screen_size": list(self.screen_size) if self.screen_size is not None else None,
            "coefficients": self.coefficients.tolist(),
            "samples": [sample.__dict__ for sample in self.samples],
        }

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompanyGazeMapper":
        """Load a saved mapper."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        screen_size = payload.get("screen_size")
        mapper = cls(
            degree=int(payload["degree"]),
            pitch_yaw_unit=payload.get("pitch_yaw_unit", "deg"),
            ridge=float(payload.get("ridge", 1e-6)),
            screen_size=tuple(screen_size) if screen_size is not None else None,
        )
        mapper.extend_samples(payload.get("samples", []))
        mapper.coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        if mapper.coefficients.shape != (mapper.feature_count, 2):
            raise ValueError("saved coefficients do not match mapper degree")
        return mapper

    @classmethod
    def fit_from_samples(
        cls,
        samples: Iterable[CompanyGazeSample | Sequence[float] | dict],
        degree: int = 2,
        pitch_yaw_unit: str = "deg",
        ridge: float = 1e-6,
        screen_size: tuple[int, int] | None = None,
    ) -> "CompanyGazeMapper":
        """Create, populate, and fit a mapper in one call."""
        mapper = cls(
            degree=degree,
            pitch_yaw_unit=pitch_yaw_unit,
            ridge=ridge,
            screen_size=screen_size,
        )
        mapper.extend_samples(samples)
        mapper.fit()
        return mapper

    def _design_matrix(self, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        pitch, yaw = self._normalize_angles(pitch, yaw)
        if self.degree == 1:
            return np.column_stack([np.ones_like(pitch), pitch, yaw])
        return np.column_stack(
            [
                np.ones_like(pitch),
                pitch,
                yaw,
                pitch * pitch,
                pitch * yaw,
                yaw * yaw,
            ]
        )

    def _normalize_angles(
        self,
        pitch: np.ndarray,
        yaw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.pitch_yaw_unit == "deg":
            # Keep values near [-1, 1] for better conditioning.
            return pitch / 90.0, yaw / 90.0
        return pitch, yaw
