"""HeatSense AI — ResNet-50 Pretrained Model for LST Estimation.

Architecture:
- Pretrained ResNet-50 backbone (ImageNet weights via torchvision)
- Spatial feature projector mapping 14-dim remote sensing features to deep visual/spatial activations
- Fine-tuned deep regression head with BatchNorm, Dropout, and residual feature fusion
- Optimization via AdamW + Cosine Annealing Learning Rate Schedule
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)


class ResNet50LST(nn.Module):
    """ResNet-50 backbone fine-tuned for Land Surface Temperature regression."""

    def __init__(
        self,
        num_features: int = 14,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_early_layers: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features

        # 1. Feature Projector: Maps 14-dim tabular geospatial features into pseudo-spatial patch (3, 64, 64)
        self.projector = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 3 * 32 * 32),
            nn.Unflatten(1, (3, 32, 32)),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
        )

        # 2. Pretrained ResNet-50 Backbone
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_early_layers:
            # Freeze conv1, bn1, layer1, layer2 to preserve low-level edge/texture filters
            for name, param in base_model.named_parameters():
                if not any(k in name for k in ["layer3", "layer4", "fc"]):
                    param.requires_grad = False

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # 3. Direct tabular skip-connection to prevent information bottleneck
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 4. Deep Regression Head (ResNet 2048-dim + Tabular 128-dim = 2176-dim)
        in_features = 2048 + 128
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_features)
        # Project to spatial patch (B, 3, 64, 64)
        spatial_map = self.projector(x)

        # Forward through ResNet backbone
        out = self.conv1(spatial_map)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        resnet_feats = torch.flatten(out, 1)  # (B, 2048)

        # Tabular direct skip connection
        tab_feats = self.tabular_encoder(x)   # (B, 128)

        # Fuse representations
        fused = torch.cat([resnet_feats, tab_feats], dim=1)
        pred_lst = self.regression_head(fused)
        return pred_lst


def train_resnet50_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    count = 0
    for features, targets, _ in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        count += len(targets)

    return total_loss / count if count > 0 else 0.0


def evaluate_resnet50(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model and return MAE, RMSE, R2, along with predictions and true labels."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, targets, _ in loader:
            features = features.to(device)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    if not all_preds:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}, np.array([]), np.array([])

    y_pred = np.vstack(all_preds).flatten()
    y_true = np.vstack(all_targets).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}, y_pred, y_true
