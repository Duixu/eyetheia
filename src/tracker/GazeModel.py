#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`GazeModel` module
========================

This module defines the `GazeModel` class and its supporting models used for gaze tracking.
It includes separate feature extractors for eyes, face, and face grid, which are then combined 
to predict the user's gaze position.

:author: Pather Stevenson
:date: February 2025
"""

import torch
import torch.nn as nn
from typing import List, Tuple

class FeatureImageModel(nn.Module):
    """
    A convolutional feature extraction model used for both eyes (with shared weights) 
    and the face (with unique weights).
    """

    def __init__(self) -> None:
        """
        Initializes the feature extraction model with convolutional layers.
        """
        super(FeatureImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Extracted features as a flattened tensor.
        """
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):
    """
    A model for processing face images and extracting relevant features.
    """

    def __init__(self) -> None:
        """
        Initializes the face image model, which consists of a feature extractor 
        followed by fully connected layers.
        """
        super(FaceImageModel, self).__init__()
        self.conv = FeatureImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for face feature extraction.

        :param x: Input tensor representing the face image.
        :return: Extracted face features as a tensor.
        """
        x = self.conv(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    """
    A model for processing the face grid, which provides positional context.
    """

    def __init__(self, gridSize: int = 25) -> None:
        """
        Initializes the face grid model with fully connected layers.

        :param gridSize: The size of the face grid (default: 25).
        """
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for face grid processing.

        :param x: Input tensor representing the face grid.
        :return: Extracted face grid features as a tensor.
        """
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def _make_mlp(in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """
    Simple MLP builder: Linear -> ReLU -> Dropout (xN) -> Linear.
    """
    layers: List[nn.Module] = []
    d = in_dim

    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        d = h

    layers.append(nn.Linear(d, out_dim))

    return nn.Sequential(*layers)

class FiLMGate(nn.Module):
    """
    Context-conditioned FiLM + gating.
    Given a context vector 'ctx', produce (gamma, beta) and a gate to modulate features.
    """

    def __init__(self, ctx_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = _make_mlp(ctx_dim, [256], 2 * feat_dim, dropout=0.1)
        self.to_gate = _make_mlp(ctx_dim, [256], feat_dim, dropout=0.1)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(ctx)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        gate = torch.sigmoid(self.to_gate(ctx))

        out = (1.0 + gamma) * feat + beta
        out = gate * out
        return out

class GazeModel(nn.Module):
    """
    The main gaze tracking model that combines eye, face, and face grid features 
    to predict gaze direction.
    """

    def __init__(self) -> None:
        """
        Initializes the GazeModel with submodules for eyes, face, and face grid processing.
        """
        super(GazeModel, self).__init__()
        self.eyeModel = FeatureImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()

        # Fully connected layer to join both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )

        # Fully connected layers to integrate all features and predict gaze direction
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        faces: torch.Tensor,
        eyesLeft: torch.Tensor,
        eyesRight: torch.Tensor,
        faceGrids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the GazeModel.

        :param faces: Tensor representing the face images.
        :param eyesLeft: Tensor representing the left eye images.
        :param eyesRight: Tensor representing the right eye images.
        :param faceGrids: Tensor representing the face grid data.
        :return: Predicted gaze coordinates as a tensor (x, y).
        """
        # Process eye images
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)

        # Concatenate eye features and apply fully connected layers
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Process face and face grid data
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Concatenate all features and apply final layers
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)

        return x

class EyeTheiaUFModel(nn.Module):
    """
    EyeTheia-UF (Uncertainty-aware Fusion)

    - Keeps the same inputs as GazeModel: face, left eye, right eye, face grid
    - Uses face-grid embedding as context to modulate eye and face features (FiLM + gating)
    - Outputs (xy, logvar) for heteroscedastic regression
    """

    def __init__(self, gridSize: int = 25) -> None:
        super().__init__()

        # Reuse baseline submodules
        self.eyeModel = FeatureImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel(gridSize=gridSize)

        # Same eyes join as baseline
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )

        # UF: context-driven modulation
        ctx_dim = 128  # FaceGridModel output
        self.film_eyes = FiLMGate(ctx_dim=ctx_dim, feat_dim=128)
        self.film_face = FiLMGate(ctx_dim=ctx_dim, feat_dim=64)

        # Fusion trunk -> latent embedding
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Heads: mean + log-variance
        self.head_xy = nn.Linear(128, 2)
        self.head_logvar = nn.Linear(128, 2)  # log(sigma^2)

    def forward(
        self,
        faces: torch.Tensor,
        eyesLeft: torch.Tensor,
        eyesRight: torch.Tensor,
        faceGrids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Eyes
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        xEyes = torch.cat((xEyeL, xEyeR), dim=1)
        xEyes = self.eyesFC(xEyes)  # (B,128)

        # Face + Grid
        xFace = self.faceModel(faces)      # (B,64)
        xGrid = self.gridModel(faceGrids)  # (B,128) -> context

        # UF modulation
        xEyes = self.film_eyes(xEyes, xGrid)
        xFace = self.film_face(xFace, xGrid)

        # Fusion
        z = torch.cat((xEyes, xFace, xGrid), dim=1)
        z = self.fusion(z)  # (B,128)

        # Heads
        xy = self.head_xy(z)
        logvar = self.head_logvar(z).clamp(-8.0, 4.0)

        return xy, logvar
    
def heteroscedastic_gaussian_loss(
    xy_pred: torch.Tensor,
    logvar: torch.Tensor,
    xy_true: torch.Tensor,
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL loss (diagonal covariance).

    logvar = log(sigma^2)
    L = exp(-logvar) * (err^2) + logvar
    """
    err2 = (xy_true - xy_pred) ** 2
    loss = torch.exp(-logvar) * err2 + logvar
    return loss.mean()


EyetheiaV1 = GazeModel