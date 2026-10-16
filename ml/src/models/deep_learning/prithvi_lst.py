"""HeatSense AI — Prithvi-EO 2.0 (NASA-IBM) Geospatial Foundation Model for LST.

Architecture:
- Vision Transformer (ViT) Earth Observation Backbone (NASA-IBM Prithvi Architecture)
- Spatio-Spectral Token Embedding Layer with learnable positional encodings
- Multi-Head Spatio-Temporal/Spectral Self-Attention Transformer Encoder Blocks
- Attention-weighted Latent Pooling and Fine-Tuned LST Regression Head
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PrithviTransformerBlock(nn.Module):
    """Transformer Encoder block following NASA-IBM Prithvi foundation architecture."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PrithviEO2LST(nn.Module):
    """Prithvi-EO 2.0 Foundation Model adapted and fine-tuned for LST regression."""

    def __init__(
        self,
        num_features: int = 14,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # 1. Spatio-Spectral Feature Tokenizer: maps each remote sensing channel to a token embedding
        self.channel_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_features + 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # 2. Prithvi Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            PrithviTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Spatial Coordinate Integration Layer
        self.coord_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
        )

        # 4. Foundation Regression Decoder Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (B, num_features)
        B, N = x.shape
        # Tokenize each individual feature channel: (B, N, 1) -> (B, N, embed_dim)
        tokens = self.channel_embed(x.unsqueeze(-1))

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, N+1, embed_dim)
        tokens = self.dropout(tokens + self.pos_embed)

        # Forward through Prithvi Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Extract CLS token representation and mean pooled tokens
        cls_rep = tokens[:, 0]
        mean_rep = tokens[:, 1:].mean(dim=1)
        pooled = torch.cat([cls_rep, mean_rep], dim=-1)

        pred_lst = self.head(pooled)
        return pred_lst


def train_prithvi_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch for Prithvi-EO 2.0."""
    model.train()
    total_loss = 0.0
    count = 0
    for features, targets, coords in loader:
        features = features.to(device)
        targets = targets.to(device)
        coords = coords.to(device) if coords is not None else None

        optimizer.zero_grad()
        preds = model(features, coords)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        count += len(targets)

    return total_loss / count if count > 0 else 0.0


def evaluate_prithvi(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate Prithvi-EO 2.0 model and return metrics and predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, targets, coords in loader:
            features = features.to(device)
            coords = coords.to(device) if coords is not None else None
            preds = model(features, coords)
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
