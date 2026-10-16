# HeatSense AI — Getting Started

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| Node.js | 20+ |
| PostgreSQL + PostGIS | 16 + PostGIS 3.4 |

---

## 1 — Database

```bash
# Create the DB and enable PostGIS:
psql -U postgres -c "CREATE USER heatsense WITH PASSWORD 'heatsense';"
psql -U postgres -c "CREATE DATABASE heatsense OWNER heatsense;"
psql -U postgres -d heatsense -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Create all tables and seed the interventions catalogue:
cd backend
pip install -r requirements.txt
python -m app.db.init_db
```

---

## 2 — ML pipeline (requires GEE access + real data)

```bash
cd ml
pip install -r requirements.txt

# 1. Export LST / Sentinel-2 / land cover rasters from Google Earth Engine:
python src/data/gee_extraction.py  # follow prompts; outputs to data/raw/gee_exports/

# 2. Build the 100 m × 100 m grid:
python src/data/grid_builder.py \
    --boundary data/raw/surat_boundary.geojson \
    --out data/processed/grid_cells.geojson

# 3. Assemble the feature table:
python src/data/feature_engineering.py \
    --grid data/processed/grid_cells.geojson \
    --gee-dir data/raw/gee_exports \
    --buildings data/raw/open_buildings.geojson \
    --roads data/raw/osm_roads.geojson \
    --out data/processed/cell_features.parquet

# 4. Train XGBoost models (one per season):
python src/models/train_xgboost.py \
    --features data/processed/cell_features.parquet \
    --out-dir models_store

# 5. Compute SHAP values and write to DB:
python src/explainability/shap_analysis.py \
    --features data/processed/cell_features.parquet \
    --models-dir models_store

# 6. Compute feasibility caps and write to DB:
python src/data/feasibility_caps.py \
    --grid data/processed/grid_cells.geojson \
    --version-id 1
```

---

## 3 — Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs

---

## 4 — Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173 (proxies `/api` → `localhost:8000`)

---

## 5 — Tests

```bash
cd backend
pytest        # runs all tests (no DB required for schema/validation tests)
pytest -m db  # runs DB-dependent tests (requires PostgreSQL)
```

---

## Pipeline order summary

```
GEE exports
    → grid_builder.py   (grid_cells.geojson)
    → feature_engineering.py  (cell_features.parquet)
    → train_xgboost.py  (models_store/xgboost_<season>.json)
    → shap_analysis.py  (populates heat_predictions.shap_values)
    → feasibility_caps.py  (populates feasibility_caps)
    → FastAPI + Frontend
```
