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


@dataclass(frozen=True)
class CompanyArcGazeSample(CompanyGazeSample):
    """One ARC calibration sample with optional face-center geometry."""

    face_center: tuple[float, float, float] = (0.0, 0.0, 0.5)


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


class CompanyArcGazeMapper:
    """
    Company-style ARC calibration mapper.

    This follows the company's original gaze calibration path:
    pitch/yaw -> attention vector -> solve screen rotation/translation with
    calibration points -> intersect the gaze ray with the screen plane z=0.
    """

    def __init__(
        self,
        pitch_yaw_unit: str = "rad",
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        if pitch_yaw_unit not in ("deg", "rad"):
            raise ValueError('pitch_yaw_unit must be "deg" or "rad"')
        self.pitch_yaw_unit = pitch_yaw_unit
        self.screen_size = screen_size
        self.samples: list[CompanyArcGazeSample] = []
        self.rotation_matrix: np.ndarray | None = None
        self.tvec: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.rotation_matrix is not None and self.tvec is not None

    def add_sample(
        self,
        pitch: float,
        yaw: float,
        x_px: float,
        y_px: float,
        face_center: Sequence[float] | None = None,
    ) -> None:
        values = (pitch, yaw, x_px, y_px)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("sample values must be finite numbers")
        center_source = (0.0, 0.0, 0.5) if face_center is None else face_center
        center = tuple(float(v) for v in center_source)
        if len(center) != 3 or not all(math.isfinite(v) for v in center):
            raise ValueError("face_center must contain three finite numbers")
        self.samples.append(
            CompanyArcGazeSample(
                pitch=float(pitch),
                yaw=float(yaw),
                x_px=float(x_px),
                y_px=float(y_px),
                face_center=center,
            )
        )

    def extend_samples(
        self,
        samples: Iterable[CompanyArcGazeSample | Sequence[float] | dict],
    ) -> None:
        for sample in samples:
            if isinstance(sample, CompanyArcGazeSample):
                self.add_sample(sample.pitch, sample.yaw, sample.x_px, sample.y_px, sample.face_center)
            elif isinstance(sample, dict):
                self.add_sample(
                    sample["pitch"],
                    sample["yaw"],
                    sample.get("x_px", sample.get("x")),
                    sample.get("y_px", sample.get("y")),
                    sample.get("face_center"),
                )
            else:
                if len(sample) == 4:
                    pitch, yaw, x_px, y_px = sample
                    face_center = None
                else:
                    pitch, yaw, x_px, y_px, face_center = sample
                self.add_sample(pitch, yaw, x_px, y_px, face_center)

    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.samples) < 6:
            raise ValueError(
                f"ARC mapping requires at least 6 samples, got {len(self.samples)}"
            )

        from scipy.optimize import leastsq

        realworld_points = np.array(
            [[sample.x_px, -sample.y_px, 0.0] for sample in self.samples],
            dtype=np.float64,
        )
        attention_vectors = np.array(
            [
                self.angles_to_attention_vector(sample.pitch, sample.yaw)
                for sample in self.samples
            ],
            dtype=np.float64,
        )
        face_centers = np.array(
            [sample.face_center for sample in self.samples],
            dtype=np.float64,
        )

        def equations(params: np.ndarray) -> np.ndarray:
            quat = params[:4]
            tvec = params[4:7]
            rotation = self._quaternion_to_rotation_matrix(quat)
            residuals = np.zeros(len(self.samples) * 2 + 1, dtype=np.float64)

            for index, (face_center, attention_vec, target) in enumerate(
                zip(face_centers, attention_vectors, realworld_points)
            ):
                face_center_rotated = rotation @ face_center + tvec
                attention_vec_rotated = rotation @ attention_vec
                dot = self._intersect_screen_plane(face_center_rotated, attention_vec_rotated)
                residuals[2 * index] = dot[0] - target[0]
                residuals[2 * index + 1] = dot[1] - target[1]

            residuals[-1] = np.dot(quat, quat) - 1.0
            return residuals

        width, height = self.screen_size or self._infer_screen_size()
        initial = np.array([0.0, 0.0, 1.0, 0.0, width / 2.0, -height / 2.0, max(width, height) * 0.5])
        result = leastsq(equations, initial)

        self.rotation_matrix = self._quaternion_to_rotation_matrix(result[0][:4])
        self.tvec = result[0][4:7].astype(np.float64)
        return self.rotation_matrix.copy(), self.tvec.copy()

    def predict(
        self,
        pitch: float,
        yaw: float,
        face_center: Sequence[float] | None = None,
        clip: bool = True,
    ) -> tuple[float, float]:
        if self.rotation_matrix is None or self.tvec is None:
            raise RuntimeError("CompanyArcGazeMapper must be fitted before predict()")

        center_source = (0.0, 0.0, 0.5) if face_center is None else face_center
        center = np.asarray(center_source, dtype=np.float64)
        attention_vec = self.angles_to_attention_vector(float(pitch), float(yaw))
        face_center_rotated = self.rotation_matrix @ center + self.tvec
        attention_vec_rotated = self.rotation_matrix @ attention_vec
        dot = self._intersect_screen_plane(face_center_rotated, attention_vec_rotated)

        x_px = float(dot[0])
        y_px = float(-dot[1])

        if clip and self.screen_size is not None:
            width, height = self.screen_size
            x_px = float(np.clip(x_px, 0, max(width - 1, 0)))
            y_px = float(np.clip(y_px, 0, max(height - 1, 0)))

        return x_px, y_px

    def predict_from_gaze(self, gaze: dict, clip: bool = True) -> tuple[float, float]:
        return self.predict(
            float(gaze["pitch"]),
            float(gaze["yaw"]),
            gaze.get("face_center"),
            clip=clip,
        )

    def residuals(self) -> np.ndarray:
        if self.rotation_matrix is None or self.tvec is None:
            raise RuntimeError("CompanyArcGazeMapper must be fitted before residuals()")
        errors = []
        for sample in self.samples:
            pred = self.predict(sample.pitch, sample.yaw, sample.face_center, clip=False)
            errors.append((pred[0] - sample.x_px, pred[1] - sample.y_px))
        return np.asarray(errors, dtype=np.float64)

    def mean_error_px(self) -> float:
        errors = self.residuals()
        return float(np.linalg.norm(errors, axis=1).mean())

    def save(self, path: str | Path) -> None:
        if self.rotation_matrix is None or self.tvec is None:
            raise RuntimeError("fit the mapper before saving it")
        payload = {
            "mapper_type": "company_arc",
            "pitch_yaw_unit": self.pitch_yaw_unit,
            "screen_size": list(self.screen_size) if self.screen_size is not None else None,
            "rotation_matrix": self.rotation_matrix.tolist(),
            "tvec": self.tvec.tolist(),
            "samples": [
                {
                    "pitch": sample.pitch,
                    "yaw": sample.yaw,
                    "x_px": sample.x_px,
                    "y_px": sample.y_px,
                    "face_center": list(sample.face_center),
                }
                for sample in self.samples
            ],
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompanyArcGazeMapper":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        screen_size = payload.get("screen_size")
        mapper = cls(
            pitch_yaw_unit=payload.get("pitch_yaw_unit", "rad"),
            screen_size=tuple(screen_size) if screen_size is not None else None,
        )
        mapper.extend_samples(payload.get("samples", []))
        mapper.rotation_matrix = np.asarray(payload["rotation_matrix"], dtype=np.float64)
        mapper.tvec = np.asarray(payload["tvec"], dtype=np.float64)
        if mapper.rotation_matrix.shape != (3, 3) or mapper.tvec.shape != (3,):
            raise ValueError("saved ARC calibration has invalid rotation/tvec shape")
        return mapper

    @classmethod
    def fit_from_samples(
        cls,
        samples: Iterable[CompanyArcGazeSample | Sequence[float] | dict],
        pitch_yaw_unit: str = "rad",
        screen_size: tuple[int, int] | None = None,
    ) -> "CompanyArcGazeMapper":
        mapper = cls(pitch_yaw_unit=pitch_yaw_unit, screen_size=screen_size)
        mapper.extend_samples(samples)
        mapper.fit()
        return mapper

    def angles_to_attention_vector(self, pitch: float, yaw: float) -> np.ndarray:
        if self.pitch_yaw_unit == "deg":
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        attention_vec = -np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
                math.cos(pitch) * math.cos(yaw),
            ],
            dtype=np.float64,
        )
        return attention_vec / np.linalg.norm(attention_vec)

    def _infer_screen_size(self) -> tuple[float, float]:
        if not self.samples:
            return 1920.0, 1080.0
        width = max(sample.x_px for sample in self.samples) + 1.0
        height = max(sample.y_px for sample in self.samples) + 1.0
        return float(width), float(height)

    @staticmethod
    def _quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = quat
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _intersect_screen_plane(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        denominator = direction[2]
        if abs(float(denominator)) < 1e-9:
            raise ValueError("gaze ray is parallel to the screen plane")
        t = -origin[2] / denominator
        return origin + direction * t
