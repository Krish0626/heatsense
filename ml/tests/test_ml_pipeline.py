"""Unit tests for HeatSense AI ML Pipeline.

Tests:
1. Data auditing and column candidate selection.
2. Spatial block cross-validation splitting.
3. Metric computation and overfitting detection.
4. XGBoost training, early stopping, and callback logging.
5. Model artifact serialization and deserialization.
6. Feature ablation logic.
7. Backend model loading compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from ml.src.data.audit import audit_dataset, FEATURE_COLS, TARGET_COL, SEASON_COL
from ml.src.models.evaluate import compute_metrics, check_overfitting
from ml.src.models.spatial_cv import assign_spatial_blocks, block_kfold_splits, run_spatial_cv
from ml.src.models.train_xgboost import (
    ProgressCallback,
    get_feature_cols,
    train_one_season,
    run_feature_ablation,
)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Generate a synthetic DataFrame mimicking cell_features.parquet."""
    np.random.seed(42)
    n_cells = 100
    rows = []
    seasons = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]

    for s in seasons:
        for cid in range(n_cells):
            x = 270000.0 + (cid % 10) * 100.0
            y = 2330000.0 + (cid // 10) * 100.0
            b_frac = np.random.uniform(0.1, 0.9)
            ndvi_val = np.random.uniform(0.05, 0.6)
            lst = 30.0 + 8.0 * b_frac - 4.0 * ndvi_val + np.random.normal(0, 0.5)
            rows.append({
                "cell_id": f"cell_{cid}",
                "centroid_x": x,
                "centroid_y": y,
                "season": s,
                "observed_lst_c": lst,
                "ndvi": ndvi_val,
                "ndbi": b_frac - ndvi_val,
                "mndwi": -0.2,
                "albedo": 0.15 + 0.1 * b_frac,
                "built_up_fraction": b_frac,
                "impervious_fraction": b_frac,
                "bare_fraction": 0.1,
                "vegetation_fraction": ndvi_val,
                "water_fraction": 0.0,
                "road_density": 500.0 * b_frac,
                "elevation_m": 15.0,
                "dist_to_water_m": 800.0,
                "dist_to_coast_m": 12000.0,
                "population_density": 1000.0 * b_frac,
                "night_lights": 25.0 * b_frac,
            })
    return pd.DataFrame(rows)


def test_audit_dataset(sample_feature_df: pd.DataFrame):
    """Test that dataset audit returns structured keys and identifies seasons."""
    audit = audit_dataset(sample_feature_df)
    assert audit["rows"] == 400
    assert "target" in audit
    assert audit["target"]["mean"] > 0
    assert "pre_monsoon" in audit["season_distribution"]
    assert audit["duplicate_rows"] == 0


def test_feature_selection(sample_feature_df: pd.DataFrame):
    """Test feature column detection excluding invalid or non-present columns."""
    feats = get_feature_cols(sample_feature_df)
    assert "ndvi" in feats
    assert "built_up_fraction" in feats
    assert "observed_lst_c" not in feats


def test_spatial_blocks_disjoint(sample_feature_df: pd.DataFrame):
    """Test spatial k-fold produces valid disjoint test splits."""
    season_df = sample_feature_df[sample_feature_df["season"] == "pre_monsoon"].reset_index(drop=True)
    splits = list(block_kfold_splits(season_df, n_folds=5, block_size_m=500.0, random_state=42))
    assert len(splits) == 5

    all_val_indices = []
    for train_idx, val_idx in splits:
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        all_val_indices.extend(val_idx)

    # All samples should be covered exactly once across folds
    assert sorted(all_val_indices) == list(range(len(season_df)))


def test_compute_metrics_and_overfitting():
    """Test metric computation and overfitting detection threshold."""
    y_true = np.array([30.0, 32.0, 34.0, 36.0, 38.0])
    y_pred_good = np.array([30.1, 31.9, 34.2, 35.8, 38.1])
    m_good = compute_metrics(y_true, y_pred_good)
    assert m_good["r2"] > 0.95
    assert m_good["mae"] < 0.3

    y_pred_bad = np.array([34.0, 34.0, 34.0, 34.0, 34.0])
    m_bad = compute_metrics(y_true, y_pred_bad)
    overfit = check_overfitting(m_good, m_bad, season="test")
    assert overfit["overfitting_flag"] is True


def test_progress_callback():
    """Test that ProgressCallback correctly accumulates RMSE logs."""
    cb = ProgressCallback(total_rounds=10, verbose_every=5)
    evals = {"validation_0": {"rmse": [1.5, 1.2]}, "validation_1": {"rmse": [1.8, 1.4]}}
    cb.after_iteration(None, epoch=0, evals_log=evals)
    assert len(cb.train_rmse_history) == 1
    assert len(cb.val_rmse_history) == 1
    assert cb.train_rmse_history[0] == 1.2
    assert cb.val_rmse_history[0] == 1.4


def test_train_one_season_mock(sample_feature_df: pd.DataFrame, tmp_path: Path):
    """Test training one season on synthetic data and model saving."""
    feature_cols = ["ndvi", "albedo", "built_up_fraction", "road_density"]
    result = train_one_season(
        df=sample_feature_df,
        season="pre_monsoon",
        feature_cols=feature_cols,
        device="cpu",
        n_estimators=30,
        early_stopping_rounds=10,
        val_size=0.2,
        verbose_eval=10,
        run_baselines=False,
    )
    assert "model" in result
    assert result["val_metrics"]["r2"] > 0.5
    assert result["completed_rounds"] > 0

    # Test model saving & loading
    save_file = tmp_path / "xgboost_pre_monsoon.json"
    result["model"].save_model(str(save_file))
    assert save_file.exists()

    loaded = xgb.XGBRegressor()
    loaded.load_model(str(save_file))
    pred = loaded.predict(sample_feature_df[feature_cols].values[:5])
    assert len(pred) == 5


def test_feature_ablation_mock(sample_feature_df: pd.DataFrame):
    """Test that feature ablation returns records for each feature and full model."""
    feature_cols = ["ndvi", "albedo", "built_up_fraction"]
    records = run_feature_ablation(
        df=sample_feature_df,
        season="pre_monsoon",
        feature_cols=feature_cols,
        device="cpu",
        n_estimators=20,
    )
    assert len(records) == len(feature_cols) + 1
    assert records[0]["omitted_feature"] == "NONE (Full Model)"
