"""HeatSense AI — SHAP analysis for trained seasonal XGBoost models.

Generates:
  - Global SHAP bar chart (mean |SHAP| per feature)
  - SHAP beeswarm plot
  - SHAP dependence plots for top-N features
  - Per-cell SHAP values saved as parquet (for DB import)

Important: SHAP values explain the model's learned associations.
A high SHAP value for built_up_fraction means the model links it to
higher predicted LST in these data. This is NOT a causal claim.

Usage:
    python src/explainability/shap_analysis.py \\
        --features data/processed/cell_features.parquet \\
        --models-dir models_store \\
        --out-dir reports
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL  = "observed_lst_c"
SEASON_COL  = "season"
FEATURE_COLS = [
    "ndvi", "ndbi", "mndwi", "albedo",
    "built_up_fraction", "bare_fraction",
    "vegetation_fraction", "water_fraction",
    "road_density", "elevation_m",
    "dist_to_water_m", "dist_to_coast_m",
    "population_density", "night_lights",
]


def analyse_season(
    df: pd.DataFrame,
    season: str,
    models_dir: Path,
    out_dir: Path,
    max_display: int = 14,
) -> None:
    import shap
    import xgboost as xgb

    season_df = df[df[SEASON_COL] == season].reset_index(drop=True)
    feature_cols_path = models_dir / f"feature_columns_{season}.json"
    model_path = models_dir / f"xgboost_{season}.json"

    if not model_path.exists():
        logger.warning(f"No model found for season={season}. Skipping SHAP.")
        return
    if not feature_cols_path.exists():
        logger.warning(f"No feature columns file for season={season}. Skipping SHAP.")
        return

    feature_cols = json.loads(feature_cols_path.read_text())
    present_cols = [c for c in feature_cols if c in season_df.columns]
    X = season_df[present_cols]

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    logger.info(f"  Computing SHAP values for season={season} ({len(X)} rows) …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Global bar chart ──────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=present_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_importance.index, feat_importance.values, color="#2563EB", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (°C)")
    ax.set_title(
        f"SHAP Feature Importance [{season}]\n"
        "(model association, NOT physical causation)"
    )
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = plots_dir / f"{season}_shap_bar.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved SHAP bar chart: {path}")

    # ── 2. Beeswarm plot ─────────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            shap_values, X, feature_names=present_cols,
            max_display=max_display, show=False, plot_type="dot",
        )
        plt.title(f"SHAP Beeswarm [{season}]")
        plt.tight_layout()
        path = plots_dir / f"{season}_shap_beeswarm.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved SHAP beeswarm: {path}")
    except Exception as exc:
        logger.warning(f"  Beeswarm plot failed: {exc}")

    # ── 3. Dependence plots for top-3 features ────────────────────────────────
    top_features = feat_importance.sort_values(ascending=False).index[:3].tolist()
    for feat in top_features:
        if feat not in present_cols:
            continue
        try:
            feat_idx = present_cols.index(feat)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(
                X[feat].values, shap_values[:, feat_idx],
                alpha=0.35, s=12, color="#7C3AED",
            )
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel(feat)
            ax.set_ylabel(f"SHAP value for {feat} (°C)")
            ax.set_title(f"SHAP Dependence: {feat} [{season}]")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = plots_dir / f"{season}_shap_dep_{feat}.png"
            fig.savefig(str(path), dpi=150)
            plt.close(fig)
            logger.info(f"  Saved SHAP dependence: {path}")
        except Exception as exc:
            logger.warning(f"  Dependence plot failed for {feat}: {exc}")

    # ── 4. Save per-cell SHAP values as parquet ───────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in present_cols])
    shap_df.insert(0, SEASON_COL, season)
    if "cell_id" in season_df.columns:
        shap_df.insert(0, "cell_id", season_df["cell_id"].values)
    parquet_path = out_dir / f"shap_values_{season}.parquet"
    shap_df.to_parquet(str(parquet_path), index=False)
    logger.info(f"  Saved SHAP parquet: {parquet_path}")

    # Summary
    logger.info(f"  Top-3 features by |SHAP| [{season}]: {top_features}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="HeatSense SHAP analysis")
    parser.add_argument("--features",    type=Path, default=Path("data/processed/cell_features.parquet"))
    parser.add_argument("--models-dir",  type=Path, default=Path("models_store"))
    parser.add_argument("--out-dir",     type=Path, default=Path("reports"))
    parser.add_argument("--season",      type=str,  default=None)
    args = parser.parse_args()

    if not args.features.exists():
        logger.error(f"Features not found: {args.features}")
        sys.exit(1)

    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        sys.exit(1)

    df = pd.read_parquet(args.features)
    seasons = [args.season] if args.season else sorted(df[SEASON_COL].unique().tolist())

    for season in seasons:
        logger.info(f"=== SHAP analysis: {season} ===")
        analyse_season(df, season, args.models_dir, args.out_dir)

    logger.info("SHAP analysis complete.")


if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    for p in [str(_here), str(_here.parent), str(_here.parents[1])]:
        if p not in sys.path:
            sys.path.insert(0, p)
    main()
