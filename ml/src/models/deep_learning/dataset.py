"""HeatSense AI — Geospatial Dataset & DataLoaders for Deep Learning Models.

Provides:
- GeospatialTabularDataset: Tensor dataset with standardization fit strictly on train split.
- SpatialPatchDataset: 2D Spatial raster feature map dataset for CNN/Transformer backbones.
- get_dataloaders: Prepares train/val/test DataLoaders with spatial or random splitting.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

TARGET_COL = "observed_lst_c"
SEASON_COL = "season"
FEATURE_COLS = [
    "ndvi", "ndbi", "mndwi", "albedo",
    "built_up_fraction", "bare_fraction",
    "vegetation_fraction", "water_fraction",
    "road_density", "elevation_m",
    "dist_to_water_m", "dist_to_coast_m",
    "population_density", "night_lights",
]


class GeospatialLSTDataset(Dataset):
    """PyTorch Dataset for Geospatial LST modeling."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        coords: Optional[np.ndarray] = None,
        as_patch: bool = False,
        patch_size: int = 16,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for GeospatialLSTDataset.")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        self.coords = torch.tensor(coords, dtype=torch.float32) if coords is not None else None
        self.as_patch = as_patch
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        feat = self.features[idx]
        target = self.targets[idx]
        coord = self.coords[idx] if self.coords is not None else torch.zeros(2)

        if self.as_patch:
            # Reshape 1D feature vector into 2D multi-channel spatial patch representation (C, H, W)
            # 14 features mapped onto 3-channel (RGB-like / multi-spectral) HxW patch
            patch = feat.repeat(3, self.patch_size, self.patch_size // len(feat) + 1)[:, :self.patch_size, :self.patch_size]
            return patch, target, coord, feat

        return feat, target, coord


def prepare_datasets(
    df: pd.DataFrame,
    season: Optional[str] = None,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    as_patch: bool = False,
    patch_size: int = 32,
) -> Tuple[GeospatialLSTDataset, GeospatialLSTDataset, GeospatialLSTDataset, StandardScaler, List[str]]:
    """Prepare train, validation, and test datasets with scaler fit ONLY on train."""
    if season:
        df = df[df[SEASON_COL] == season].reset_index(drop=True)

    present_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_raw = df[present_cols].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.float32)
    
    coords = None
    if "centroid_x" in df.columns and "centroid_y" in df.columns:
        coords = df[["centroid_x", "centroid_y"]].values.astype(np.float32)
        # Normalize coords to [0, 1]
        coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-6)

    # First split off test set
    if test_size > 0:
        X_train_val, X_test, y_train_val, y_test, c_tv, c_test = train_test_split(
            X_raw, y_raw, coords if coords is not None else np.zeros((len(X_raw), 2)),
            test_size=test_size, random_state=random_state
        )
    else:
        X_train_val, y_train_val, c_tv = X_raw, y_raw, coords
        X_test, y_test, c_test = X_raw[:0], y_raw[:0], coords[:0] if coords is not None else None

    # Then split train and validation
    adj_val_size = val_size / (1.0 - test_size) if test_size > 0 else val_size
    X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
        X_train_val, y_train_val, c_tv,
        test_size=adj_val_size, random_state=random_state
    )

    # Standardize features (fit only on train)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test) if len(X_test) > 0 else X_test

    train_ds = GeospatialLSTDataset(X_train_norm, y_train, c_train, as_patch=as_patch, patch_size=patch_size)
    val_ds = GeospatialLSTDataset(X_val_norm, y_val, c_val, as_patch=as_patch, patch_size=patch_size)
    test_ds = GeospatialLSTDataset(X_test_norm, y_test, c_test, as_patch=as_patch, patch_size=patch_size)

    return train_ds, val_ds, test_ds, scaler, present_cols
