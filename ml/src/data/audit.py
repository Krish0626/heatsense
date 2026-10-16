"""HeatSense AI -- Dataset auditing and validation utilities.

Run standalone:
    python src/data/audit.py --features data/processed/cell_features.parquet

Produces:
    ml/reports/dataset_audit.json
    ml/reports/dataset_audit.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Column groups ------------------------------------------------------------
FEATURE_COLS = [
    "ndvi", "ndbi", "mndwi", "albedo",
    "built_up_fraction", "impervious_fraction",
    "bare_fraction", "vegetation_fraction", "water_fraction",
    "road_density", "elevation_m",
    "dist_to_water_m", "dist_to_coast_m",
    "population_density", "night_lights",
]
TARGET_COL = "observed_lst_c"
SEASON_COL = "season"
CELL_ID_COL = "cell_id"
SPATIAL_COLS = ["centroid_x", "centroid_y"]


def _iqr_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return a structured audit dictionary for the given DataFrame."""
    audit: dict = {}

    # -- Basic shape ----------------------------------------------------------
    audit["rows"] = len(df)
    audit["columns"] = list(df.columns)
    audit["n_columns"] = len(df.columns)
    audit["dtypes"] = df.dtypes.astype(str).to_dict()

    # -- Missing / inf --------------------------------------------------------
    audit["missing_values"] = df.isnull().sum().to_dict()
    num_df = df.select_dtypes(include=[np.number])
    audit["infinite_values"] = {
        c: int(np.isinf(df[c]).sum())
        for c in num_df.columns
        if np.isinf(df[c]).sum() > 0
    }

    # -- Duplicate checks -----------------------------------------------------
    audit["duplicate_rows"] = int(df.duplicated().sum())
    if CELL_ID_COL in df.columns and SEASON_COL in df.columns:
        audit["duplicate_cell_season"] = int(
            df.duplicated(subset=[CELL_ID_COL, SEASON_COL]).sum()
        )

    # -- Season distribution --------------------------------------------------
    if SEASON_COL in df.columns:
        audit["season_distribution"] = df[SEASON_COL].value_counts().to_dict()

    # -- Spatial coverage -----------------------------------------------------
    if CELL_ID_COL in df.columns:
        audit["unique_cells"] = int(df[CELL_ID_COL].nunique())
    if all(c in df.columns for c in SPATIAL_COLS):
        x_range = df["centroid_x"].max() - df["centroid_x"].min()
        y_range = df["centroid_y"].max() - df["centroid_y"].min()
        audit["spatial_extent_m"] = {
            "x_range": round(x_range, 1),
            "y_range": round(y_range, 1),
        }

    # -- Target statistics ----------------------------------------------------
    if TARGET_COL in df.columns:
        t = df[TARGET_COL]
        audit["target"] = {
            "column": TARGET_COL,
            "mean": round(float(t.mean()), 4),
            "median": round(float(t.median()), 4),
            "std": round(float(t.std()), 4),
            "min": round(float(t.min()), 4),
            "max": round(float(t.max()), 4),
            "n_outliers_iqr": _iqr_outliers(t),
        }
        if SEASON_COL in df.columns:
            per_season = {}
            for s, g in df.groupby(SEASON_COL):
                per_season[s] = {
                    "mean": round(float(g[TARGET_COL].mean()), 4),
                    "std": round(float(g[TARGET_COL].std()), 4),
                    "min": round(float(g[TARGET_COL].min()), 4),
                    "max": round(float(g[TARGET_COL].max()), 4),
                }
            audit["target_by_season"] = per_season

    # -- Feature statistics ---------------------------------------------------
    feature_info = {}
    for col in FEATURE_COLS:
        if col not in df.columns:
            feature_info[col] = {"status": "MISSING FROM DATASET"}
            continue
        s = df[col]
        feature_info[col] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "missing": int(s.isnull().sum()),
            "n_outliers_iqr": _iqr_outliers(s),
            "corr_with_target": round(float(s.corr(df[TARGET_COL])), 6)
            if TARGET_COL in df.columns
            else None,
        }
    audit["features"] = feature_info

    # -- High correlations (multicollinearity) --------------------------------
    present_feats = [c for c in FEATURE_COLS if c in df.columns]
    corr_mat = df[present_feats].corr().abs()
    high_corr_pairs = []
    for i in range(len(present_feats)):
        for j in range(i + 1, len(present_feats)):
            c = float(corr_mat.iloc[i, j])
            if c >= 0.85:
                high_corr_pairs.append(
                    {
                        "feature_a": present_feats[i],
                        "feature_b": present_feats[j],
                        "pearson_abs": round(c, 6),
                    }
                )
    high_corr_pairs.sort(key=lambda x: -x["pearson_abs"])
    audit["high_correlation_pairs"] = high_corr_pairs

    # -- valid_obs_count ------------------------------------------------------
    if "valid_obs_count" in df.columns:
        audit["valid_obs_count"] = {
            "min": int(df["valid_obs_count"].min()),
            "max": int(df["valid_obs_count"].max()),
            "cells_below_5": int((df["valid_obs_count"] < 5).sum()),
        }

    return audit


def print_audit(audit: dict) -> str:
    """Render a human-readable text version of the audit."""
    lines = [
        "=" * 60,
        "HEATSENSE AI -- DATASET AUDIT",
        "=" * 60,
        "",
        f"Rows    : {audit['rows']:,}",
        f"Columns : {audit['n_columns']}",
        f"Duplicate rows           : {audit.get('duplicate_rows', 0)}",
        f"Duplicate cell+season    : {audit.get('duplicate_cell_season', 0)}",
        "",
        "-- Season distribution --",
    ]
    for s, n in audit.get("season_distribution", {}).items():
        lines.append(f"  {s:<20}: {n:,}")

    lines += ["", "-- Target: observed_lst_c --"]
    if "target" in audit:
        t = audit["target"]
        lines.append(f"  Mean   : {t['mean']} C")
        lines.append(f"  Median : {t['median']} C")
        lines.append(f"  Std    : {t['std']} C")
        lines.append(f"  Min    : {t['min']} C")
        lines.append(f"  Max    : {t['max']} C")
        lines.append(f"  Outliers (IQR) : {t['n_outliers_iqr']}")

    lines += ["", "-- Target by season --"]
    for s, v in audit.get("target_by_season", {}).items():
        lines.append(f"  {s:<16}: mean={v['mean']} std={v['std']} [{v['min']} to {v['max']}]")

    lines += ["", "-- Spatial coverage --"]
    lines.append(f"  Unique grid cells : {audit.get('unique_cells', 'N/A')}")
    se = audit.get("spatial_extent_m", {})
    if se:
        lines.append(f"  X extent : {se['x_range']:.0f} m ({se['x_range']/1000:.1f} km)")
        lines.append(f"  Y extent : {se['y_range']:.0f} m ({se['y_range']/1000:.1f} km)")

    lines += ["", "-- Missing values --"]
    missing = audit.get("missing_values", {})
    any_missing = False
    for col, n in missing.items():
        if n > 0:
            lines.append(f"  {col}: {n}")
            any_missing = True
    if not any_missing:
        lines.append("  None -- all features complete.")

    lines += ["", "-- Infinite values --"]
    inf_vals = audit.get("infinite_values", {})
    if inf_vals:
        for col, n in inf_vals.items():
            lines.append(f"  {col}: {n} Inf values")
    else:
        lines.append("  None.")

    lines += ["", "-- Feature correlations with target --"]
    for col, info in audit.get("features", {}).items():
        if isinstance(info, dict) and "corr_with_target" in info:
            lines.append(f"  {col:<25}: r={info['corr_with_target']:+.4f}  outliers={info['n_outliers_iqr']}")

    lines += ["", "-- High inter-feature correlations (|r| ≥ 0.85) --"]
    for pair in audit.get("high_correlation_pairs", []):
        lines.append(
            f"  {pair['feature_a']} vs {pair['feature_b']}: |r|={pair['pearson_abs']:.4f}"
        )

    lines += ["", "=" * 60]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="HeatSense dataset auditor")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/cell_features.parquet"),
        help="Path to cell_features.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write audit reports",
    )
    args = parser.parse_args()

    if not args.features.exists():
        logger.error(f"Features file not found: {args.features}")
        sys.exit(1)

    logger.info(f"Loading: {args.features}")
    df = pd.read_parquet(args.features)

    logger.info("Running audit …")
    audit = audit_dataset(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "dataset_audit.json"
    txt_path = args.out_dir / "dataset_audit.txt"

    json_path.write_text(json.dumps(audit, indent=2, default=str))
    logger.info(f"Saved: {json_path}")

    # Save feature correlation matrix
    present_feats = [c for c in FEATURE_COLS if c in df.columns]
    if present_feats:
        corr_df = df[present_feats].corr()
        corr_csv = args.out_dir / "feature_correlation.csv"
        corr_df.to_csv(corr_csv)
        logger.info(f"Saved: {corr_csv}")

        # Correlation heatmap plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plots_dir = args.out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax)
            ax.set_xticks(range(len(present_feats)))
            ax.set_yticks(range(len(present_feats)))
            ax.set_xticklabels(present_feats, rotation=90, fontsize=8)
            ax.set_yticklabels(present_feats, fontsize=8)
            ax.set_title("Feature Correlation Matrix", pad=20)
            plt.tight_layout()
            heat_path = plots_dir / "feature_correlation_heatmap.png"
            fig.savefig(str(heat_path), dpi=150)
            plt.close(fig)
            logger.info(f"Saved: {heat_path}")
        except Exception as exc:
            logger.warning(f"Could not save correlation heatmap: {exc}")

    text = print_audit(audit)
    txt_path.write_text(text, encoding="utf-8")
    logger.info(f"Saved: {txt_path}")
    print()
    import sys
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()

