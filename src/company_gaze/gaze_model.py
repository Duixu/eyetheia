from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from .base_model import BaseModel
from .swin import GazeNet


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GAZE_WEIGHT_PATH = os.path.join(PACKAGE_ROOT, "models", "Iter_19_swin_peg.pt")


class GazeStrategy:
    def forward(self, input_data: Any) -> torch.Tensor:
        raise NotImplementedError

    def get_type(self) -> str:
        raise NotImplementedError

    def get_model(self) -> torch.nn.Module | None:
        raise NotImplementedError

    def load_weights(self, weight_path: str | None = None) -> None:
        raise NotImplementedError


class SwinGazeStrategy(GazeStrategy):
    """PyTorch strategy for the company Swin gaze checkpoint."""

    def __init__(
        self,
        device: str | torch.device,
        weight_path: str | None = None,
        backbone: str = "swin_tiny_patch4_window7_224",
    ) -> None:
        self.device = torch.device(device)
        self.weight_path = weight_path or DEFAULT_GAZE_WEIGHT_PATH
        self.model = GazeNet(backbone=backbone, pretrained=False).to(self.device)
        self.model.eval()

    def forward(self, input_data: Any) -> torch.Tensor:
        x = input_data["face"] if isinstance(input_data, dict) else input_data
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"expected torch.Tensor or np.ndarray, got {type(x)!r}")
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input tensor, got shape {tuple(x.shape)}")

        x = x.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x)

    def get_type(self) -> str:
        return "pitch_yaw"

    def get_model(self) -> torch.nn.Module:
        return self.model

    def load_weights(self, weight_path: str | None = None) -> None:
        weight_path = weight_path or self.weight_path
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Company gaze weight file not found: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "net" in checkpoint:
                checkpoint = checkpoint["net"]

        if isinstance(checkpoint, dict):
            checkpoint = {
                key.replace("module.", "", 1): value
                for key, value in checkpoint.items()
            }
        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint.to(self.device)
            self.model.eval()
            return

        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()


class GazeEstimationModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        task: str = "gaze_estimation",
        device: str | torch.device | None = None,
        weight_path: str | None = None,
    ) -> None:
        super().__init__(model_name, task)
        if device is not None:
            self.device = torch.device(device)
        self.weight_path = weight_path
        self.strategy: GazeStrategy | None = None
        self.is_initialized = False
        self._init_strategy()

    def _init_strategy(self) -> None:
        name = self.model_name.lower()
        if any(token in name for token in ("swin", "torch", "pt")):
            self.strategy = SwinGazeStrategy(self.device, weight_path=self.weight_path)
        else:
            raise ValueError(
                f"Unsupported company gaze model type: {self.model_name}. "
                'Use "swin" for the Iter_19_swin_peg.pt checkpoint.'
            )
        self.is_initialized = True

    def build_model(self) -> torch.nn.Module | None:
        return self.strategy.get_model() if self.strategy else None

    def forward(self, input_data: Any) -> torch.Tensor:
        if self.strategy is None:
            raise RuntimeError("GazeEstimationModel is not initialized")
        return self.strategy.forward(input_data)

    def load_weights(self, weight_path: str | None = None) -> None:
        if self.strategy is None:
            raise RuntimeError("GazeEstimationModel is not initialized")
        self.strategy.load_weights(weight_path)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "device": str(self.device),
            "is_initialized": self.is_initialized,
            "model_type": self.strategy.get_type() if self.strategy else None,
        }
