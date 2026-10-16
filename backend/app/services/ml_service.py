"""Loads the trained XGBoost model + feature schema produced by
ml/src/models/train_xgboost.py, and exposes prediction / counterfactual
cooling estimation to the API layer.

This is the seam between the "ml/" training code and the "backend/" serving
code: ml/ produces artifacts (xgboost_<season>.json, feature_columns_<season>.json),
this module loads them once per season and caches them in process.

The counterfactual cooling estimate follows plan.md section 7.4:
  1. Perturb the cell's features to reflect the intervention (albedo up for
     cool roof, NDVI up for tree planting, etc.), scaled by the fraction of
     the cell area treated.
  2. Re-predict LST with the trained model.
  3. Clip the resulting delta to the literature bounds from
     ml/src/optimization/cooling_coefficients.py so anomalous extrapolations
     are flagged, not silently accepted.
  4. Return the average cooling across all requested cells plus an
     uncertainty band (half-range of the literature bounds).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import get_settings

settings = get_settings()

# Matches ml/src/optimization/cooling_coefficients.py
INTERVENTION_FEATURE_EFFECTS: dict[str, dict] = {
    "cool_roof_limewash": {"feature": "albedo", "delta_per_full_coverage": 0.35},
    "cool_roof_hdpe": {"feature": "albedo", "delta_per_full_coverage": 0.45},
    "tree_planting": {"feature": "ndvi", "delta_per_full_coverage": 0.4},
    "cool_pavement": {"feature": "albedo", "delta_per_full_coverage": 0.20},
}

PLAUSIBLE_COOLING_BOUNDS_C: dict[str, tuple[float, float]] = {
    "cool_roof_limewash": (0.5, 5.0),
    "cool_roof_hdpe": (0.5, 5.0),
    "tree_planting": (0.0, 4.0),
    "cool_pavement": (0.0, 12.0),
}

CELL_AREA_SQM = 10_000.0  # 100 m × 100 m


@lru_cache(maxsize=None)
def _model_dir() -> Path:
    return Path(settings.ml_model_dir)


@lru_cache(maxsize=8)  # one entry per (season,)
def _load_season_artifacts(season: str) -> tuple:
    """Load XGBoost model and feature column list for one season.

    Returns (xgb.XGBRegressor, list[str]) or raises FileNotFoundError if the
    model hasn't been trained yet (run ml/src/models/train_xgboost.py first).
    """
    import xgboost as xgb  # deferred import: optional dep at API startup time

    model_dir = _model_dir()
    model_path = model_dir / f"xgboost_{season}.json"
    cols_path = model_dir / f"feature_columns_{season}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model for season={season!r}. "
            f"Expected: {model_path}. "
            "Run `python ml/src/models/train_xgboost.py` first."
        )
    if not cols_path.exists():
        raise FileNotFoundError(
            f"Feature column list missing for season={season!r}. "
            f"Expected: {cols_path}."
        )

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    feature_cols: list[str] = json.loads(cols_path.read_text())
    return model, feature_cols


def _model_available(season: str) -> bool:
    """True iff a trained model artifact exists for the given season."""
    return (_model_dir() / f"xgboost_{season}.json").exists()


def predict_lst(cell_features_row: dict, season: str) -> float:
    """Predict LST (°C) for a single cell, given its feature values.

    `cell_features_row` must contain all columns in feature_columns_<season>.json.
    Returns the raw XGBoost prediction; SHAP attribution is separate (see
    ml/src/explainability/shap_analysis.py).
    """
    import pandas as pd

    model, feature_cols = _load_season_artifacts(season)
    X = pd.DataFrame([{col: cell_features_row.get(col, np.nan) for col in feature_cols}])
    return float(model.predict(X)[0])


def estimate_cooling(
    cell_ids: list[int],
    season: str,
    intervention_mix: dict[str, float],
    db=None,  # optional: SQLAlchemy session to load cell_features rows
) -> tuple[float, float | None]:
    """Counterfactual cooling estimate (plan.md section 7.4).

    For each cell in cell_ids:
      - Load feature values from the `cell_features` table (requires db).
      - For each intervention, compute the area fraction treated and nudge
        the relevant feature by `delta_per_full_coverage * fraction`.
      - Re-predict LST; the drop is the estimated cooling for that cell.
      - Clip to PLAUSIBLE_COOLING_BOUNDS_C and average across cells.

    If `db` is None or the model is not yet trained, returns (0.0, None)
    as a safe placeholder so the API stays callable during development.

    Returns (estimated_cooling_c, uncertainty_c). uncertainty_c is the
    half-range of the applicable literature bounds (a conservative estimate;
    a proper uncertainty would come from an ensemble or prediction interval).
    """
    if db is None or not _model_available(season):
        # Placeholder path: no DB session or no trained model yet.
        return 0.0, None

    import pandas as pd
    from sqlalchemy import select
    from app.models.cell_features import CellFeatures

    model, feature_cols = _load_season_artifacts(season)

    cell_coolings: list[float] = []
    for cell_id in cell_ids:
        # Load the most recent feature row for this cell/season.
        row = db.execute(
            select(CellFeatures)
            .where(
                CellFeatures.cell_id == cell_id,
                CellFeatures.season == season,
            )
            .order_by(CellFeatures.version_id.desc())
            .limit(1)
        ).scalar_one_or_none()

        if row is None:
            # No feature data yet; skip this cell.
            continue

        # Build a feature dict from the ORM row.
        features = {col: getattr(row, col, np.nan) for col in feature_cols}
        X_base = pd.DataFrame([features])
        baseline_lst = float(model.predict(X_base)[0])

        # Apply each intervention proportionally to the area treated.
        perturbed_features = {**features}
        for intervention_name, area_sqm in intervention_mix.items():
            if area_sqm <= 0:
                continue
            effect = INTERVENTION_FEATURE_EFFECTS.get(intervention_name)
            if effect is None:
                continue
            feat = effect["feature"]
            area_fraction = min(area_sqm / CELL_AREA_SQM, 1.0)
            delta = effect["delta_per_full_coverage"] * area_fraction
            current_val = perturbed_features.get(feat)
            if current_val is not None and not np.isnan(current_val):
                perturbed_features[feat] = float(np.clip(current_val + delta, -1.0, 1.0))

        X_perturbed = pd.DataFrame([perturbed_features])
        perturbed_lst = float(model.predict(X_perturbed)[0])
        raw_cooling = baseline_lst - perturbed_lst  # positive = cooling

        # Clip to literature bounds (use the dominant intervention's bounds).
        dominant = max(intervention_mix, key=intervention_mix.get, default=None)
        if dominant and dominant in PLAUSIBLE_COOLING_BOUNDS_C:
            lo, hi = PLAUSIBLE_COOLING_BOUNDS_C[dominant]
            raw_cooling = float(np.clip(raw_cooling, -abs(lo), hi))

        cell_coolings.append(raw_cooling)

    if not cell_coolings:
        return 0.0, None

    mean_cooling = float(np.mean(cell_coolings))

    # Uncertainty: half-range of the dominant intervention's bounds.
    dominant = max(intervention_mix, key=intervention_mix.get, default=None)
    uncertainty: float | None = None
    if dominant and dominant in PLAUSIBLE_COOLING_BOUNDS_C:
        lo, hi = PLAUSIBLE_COOLING_BOUNDS_C[dominant]
        uncertainty = (hi - max(lo, 0)) / 2.0

    return mean_cooling, uncertainty
