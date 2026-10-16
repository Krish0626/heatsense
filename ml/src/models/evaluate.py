"""HeatSense AI — Model evaluation utilities.

Provides reusable functions for computing and reporting metrics,
residual analysis, and feature importance — shared across train_xgboost.py,
spatial_cv.py, and any future model scripts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict[str, float]:
    """Return MAE, RMSE, R² for a prediction set."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    if label:
        logger.info(f"  [{label}] MAE={mae:.4f} °C  RMSE={rmse:.4f} °C  R²={r2:.4f}")
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)}


def check_overfitting(train_metrics: dict, val_metrics: dict, season: str = "") -> dict:
    """Flag potential overfitting based on train vs validation R² gap."""
    r2_gap = train_metrics["r2"] - val_metrics["r2"]
    rmse_ratio = val_metrics["rmse"] / train_metrics["rmse"] if train_metrics["rmse"] > 0 else 1.0
    # Threshold: val R² more than 0.10 below train R², or RMSE ratio > 1.3
    overfit = r2_gap > 0.10 or rmse_ratio > 1.3
    result = {
        "season": season,
        "train_r2": round(train_metrics["r2"], 4),
        "val_r2": round(val_metrics["r2"], 4),
        "r2_gap": round(r2_gap, 4),
        "train_rmse": round(train_metrics["rmse"], 4),
        "val_rmse": round(val_metrics["rmse"], 4),
        "rmse_ratio_val_train": round(rmse_ratio, 4),
        "overfitting_flag": overfit,
        "threshold_note": "r2_gap > 0.10 or val_rmse/train_rmse > 1.3",
    }
    if overfit:
        logger.warning(
            f"  OVERFITTING DETECTED [{season}]: train_R²={train_metrics['r2']:.4f} "
            f"val_R²={val_metrics['r2']:.4f} gap={r2_gap:.4f}"
        )
    else:
        logger.info(
            f"  No overfitting detected [{season}]: train_R²={train_metrics['r2']:.4f} "
            f"val_R²={val_metrics['r2']:.4f} gap={r2_gap:.4f}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    season: str,
    out_path: Path,
) -> None:
    """Plot train vs validation RMSE over boosting rounds.

    Args:
        history: dict with keys 'train_rmse' and 'val_rmse', each a list.
        season: season label for title.
        out_path: file path to save PNG.
    """
    train_rmse = history.get("train_rmse", [])
    val_rmse = history.get("val_rmse", [])

    if not train_rmse:
        logger.warning(f"No training history to plot for {season}")
        return

    rounds = list(range(1, len(train_rmse) + 1))
    best_iter = history.get("best_iteration", len(rounds))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, train_rmse, label="Train RMSE", linewidth=1.5, color="#2563EB")
    if val_rmse:
        ax.plot(rounds, val_rmse, label="Validation RMSE", linewidth=1.5, color="#DC2626")
        ax.axvline(
            x=best_iter + 1, linestyle="--", color="#16A34A", linewidth=1.2,
            label=f"Best iteration = {best_iter + 1}",
        )
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title(f"HeatSense AI — Training Curves [{season}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved training curve: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Residual analysis plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    season: str,
    plots_dir: Path,
) -> None:
    """Generate residual histogram, actual-vs-predicted, and residual-vs-predicted plots."""
    residuals = y_true - y_pred
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Residual histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40, color="#3B82F6", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (°C)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution [{season}]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = plots_dir / f"{season}_residuals.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved residuals histogram: {path}")

    # 2. Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#3B82F6")
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="#DC2626", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual LST (°C)")
    ax.set_ylabel("Predicted LST (°C)")
    ax.set_title(f"Actual vs Predicted [{season}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = plots_dir / f"{season}_actual_vs_predicted.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved actual-vs-predicted: {path}")

    # 3. Residual vs predicted
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15, color="#8B5CF6")
    ax.axhline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted LST (°C)")
    ax.set_ylabel("Residual (°C)")
    ax.set_title(f"Residual vs Predicted [{season}]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = plots_dir / f"{season}_residual_vs_predicted.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved residual-vs-predicted: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance plots
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_importance(
    model,
    feature_cols: list[str],
    season: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Save feature importance (gain, weight) and generate bar plot."""
    import xgboost as xgb

    booster = model.get_booster()
    importance_types = ["gain", "weight"]
    records = []
    for imp_type in importance_types:
        scores = booster.get_score(importance_type=imp_type)
        for feat in feature_cols:
            records.append(
                {
                    "feature": feat,
                    "importance_type": imp_type,
                    "score": scores.get(feat, 0.0),
                }
            )

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"feature_importance_{season}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved feature importance: {csv_path}")

    # Plot gain importance
    gain_df = df[df["importance_type"] == "gain"].sort_values("score")
    if not gain_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(gain_df["feature"], gain_df["score"], color="#2563EB", alpha=0.85)
        ax.set_xlabel("Gain Importance")
        ax.set_title(f"Feature Importance (Gain) [{season}]")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plot_path = out_dir / f"{season}_feature_importance.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved feature importance plot: {plot_path}")

    return df
