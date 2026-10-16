"""HeatSense AI — Spatial block cross-validation.

Assigns grid cells to spatial folds based on their UTM coordinates.
Cells are grouped into square blocks of `block_size_m` × `block_size_m`
metres; blocks are then assigned round-robin to folds.

This ensures that training and validation sets are spatially separated,
preventing inflated R² due to spatial autocorrelation between neighbours.

Run standalone:
    python src/models/spatial_cv.py \\
        --features data/processed/cell_features.parquet \\
        --out-dir reports

Produces:
    reports/spatial_cv_results.csv
    reports/spatial_cv_summary.csv
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "observed_lst_c"
DEFAULT_FEATURE_COLS = [
    "ndvi", "ndbi", "mndwi", "albedo",
    "built_up_fraction", "bare_fraction",
    "vegetation_fraction", "water_fraction",
    "road_density", "elevation_m",
    "dist_to_water_m", "dist_to_coast_m",
    "population_density", "night_lights",
]


# ─────────────────────────────────────────────────────────────────────────────
# Spatial block assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_spatial_blocks(
    df: pd.DataFrame,
    block_size_m: float = 2000.0,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.Series:
    """Assign each row a block_id based on its UTM coordinates.

    Returns a Series of integer block IDs aligned with df's index.
    """
    x_block = ((df[x_col] - df[x_col].min()) // block_size_m).astype(int)
    y_block = ((df[y_col] - df[y_col].min()) // block_size_m).astype(int)
    # Encode (x_block, y_block) as a single integer
    n_x = int(x_block.max()) + 1
    block_ids = x_block + y_block * n_x
    return block_ids.rename("block_id")


def block_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_size_m: float = 2000.0,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, val_indices) for each spatial fold.

    Block assignment is deterministic (no shuffle of blocks within fold).
    Each unique block goes entirely to train or validation — never split.
    """
    block_ids = assign_spatial_blocks(df, block_size_m=block_size_m, x_col=x_col, y_col=y_col)
    unique_blocks = np.sort(block_ids.unique())

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_blocks)
    fold_assignments = {b: i % n_folds for i, b in enumerate(shuffled)}

    block_fold = block_ids.map(fold_assignments)

    for fold_idx in range(n_folds):
        val_mask = block_fold == fold_idx
        train_mask = ~val_mask
        yield (
            np.where(train_mask.values)[0],
            np.where(val_mask.values)[0],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main spatial CV runner
# ─────────────────────────────────────────────────────────────────────────────

def run_spatial_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    season: str,
    n_folds: int = 5,
    block_size_m: float = 2000.0,
    device: str = "cpu",
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
    verbose_eval: int = 0,
) -> list[dict]:
    """Run spatial block K-fold CV for one season.

    Returns a list of per-fold result dicts.
    """
    import xgboost as xgb

    season_df = df[df["season"] == season].reset_index(drop=True)
    X = season_df[feature_cols].values
    y = season_df[target_col].values

    logger.info(
        f"  Spatial CV — season={season}, samples={len(season_df)}, "
        f"folds={n_folds}, block_size={block_size_m:.0f} m"
    )

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        block_kfold_splits(
            season_df,
            n_folds=n_folds,
            block_size_m=block_size_m,
            random_state=random_state,
        ),
        start=1,
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            device=device,
            tree_method="hist",
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose_eval if verbose_eval > 0 else False,
        )

        val_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, val_pred)

        fold_results.append(
            {
                "season": season,
                "fold": fold_idx,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else n_estimators,
                **{k: round(v, 6) for k, v in metrics.items()},
            }
        )
        logger.info(
            f"    Fold {fold_idx}/{n_folds}: val_samples={len(val_idx)}, "
            f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
        )

    return fold_results


def summarise_cv(fold_results: list[dict]) -> dict:
    """Aggregate fold results into mean/std summary."""
    import pandas as pd

    df = pd.DataFrame(fold_results)
    summary = {}
    for metric in ["mae", "rmse", "r2"]:
        summary[f"mean_{metric}"] = round(float(df[metric].mean()), 6)
        summary[f"std_{metric}"] = round(float(df[metric].std()), 6)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="HeatSense spatial block CV")
    parser.add_argument("--features", type=Path, default=Path("data/processed/cell_features.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--block-size", type=float, default=2000.0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    if not args.features.exists():
        logger.error(f"Features not found: {args.features}")
        sys.exit(1)

    df = pd.read_parquet(args.features)
    feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns and c != "impervious_fraction"]

    all_fold_results = []
    all_summaries = []

    for season in df["season"].unique():
        fold_results = run_spatial_cv(
            df=df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            season=season,
            n_folds=args.n_folds,
            block_size_m=args.block_size,
            device=args.device,
            n_estimators=args.n_estimators,
        )
        summary = summarise_cv(fold_results)
        summary["season"] = season
        all_fold_results.extend(fold_results)
        all_summaries.append(summary)

        logger.info(
            f"  Summary [{season}]: mean_R²={summary['mean_r2']:.4f}±{summary['std_r2']:.4f}, "
            f"mean_RMSE={summary['mean_rmse']:.4f}±{summary['std_rmse']:.4f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(all_fold_results)
    cv_df.to_csv(args.out_dir / "spatial_cv_results.csv", index=False)
    logger.info(f"Saved: {args.out_dir}/spatial_cv_results.csv")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(args.out_dir / "spatial_cv_summary.csv", index=False)
    logger.info(f"Saved: {args.out_dir}/spatial_cv_summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
