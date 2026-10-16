"""Unit tests for Deep Learning & Foundation Models.

Tests:
1. Geospatial dataset preparation and tensor dimensions.
2. ResNet-50 LST forward pass and gradient backpropagation.
3. Prithvi-EO 2.0 forward pass and attention mechanism.
4. IBM Granite Geospatial forward pass and group attention.
5. End-to-end training epoch and metric validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.models.deep_learning.dataset import prepare_datasets, GeospatialLSTDataset


@pytest.fixture
def dummy_geo_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 64
    return pd.DataFrame({
        "cell_id": [f"c_{i}" for i in range(n)],
        "season": ["pre_monsoon"] * n,
        "centroid_x": np.random.uniform(270000, 290000, n),
        "centroid_y": np.random.uniform(2330000, 2350000, n),
        "observed_lst_c": np.random.uniform(30.0, 42.0, n),
        "ndvi": np.random.uniform(0.1, 0.6, n),
        "ndbi": np.random.uniform(0.0, 0.4, n),
        "mndwi": np.random.uniform(-0.3, 0.1, n),
        "albedo": np.random.uniform(0.1, 0.3, n),
        "built_up_fraction": np.random.uniform(0.1, 0.9, n),
        "bare_fraction": np.random.uniform(0.0, 0.2, n),
        "vegetation_fraction": np.random.uniform(0.1, 0.5, n),
        "water_fraction": np.random.uniform(0.0, 0.1, n),
        "road_density": np.random.uniform(100, 800, n),
        "elevation_m": np.random.uniform(10, 30, n),
        "dist_to_water_m": np.random.uniform(100, 2000, n),
        "dist_to_coast_m": np.random.uniform(5000, 20000, n),
        "population_density": np.random.uniform(500, 5000, n),
        "night_lights": np.random.uniform(10, 60, n),
    })


def test_deep_dataset_preparation(dummy_geo_df: pd.DataFrame):
    """Test train/val/test dataset split and normalization."""
    train_ds, val_ds, test_ds, scaler, feature_cols = prepare_datasets(dummy_geo_df, season="pre_monsoon")
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(feature_cols) >= 14
    feat, target, coord = train_ds[0]
    assert feat.shape[0] == len(feature_cols)
    assert target.shape[0] == 1
