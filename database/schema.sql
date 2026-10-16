-- HeatSense AI database schema
-- PostgreSQL + PostGIS. Run after `CREATE EXTENSION postgis;`
-- Fixes applied vs. the mid-sem deck (see plan.md section 2.2 and 9.1):
--   * is_hotspot and rooftop area moved from planning_areas to grid_cells /
--     heat_predictions, since both are per-cell-per-season facts.
--   * feasibility_caps and recommendations are split into separate tables.
--   * scenario runs and Pareto runs each get their own tables.
--   * a data_versions table tracks which imagery window + model version
--     produced a given prediction, so re-runs don't silently overwrite history.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- Administrative hierarchy: City > Zones > Planning Areas > Grid Cells
-- ---------------------------------------------------------------------------

CREATE TABLE cities (
    city_id     SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    state       VARCHAR(100) NOT NULL,
    boundary    GEOMETRY(MultiPolygon, 4326) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_cities_boundary ON cities USING GIST (boundary);

CREATE TABLE zones (
    zone_id     SERIAL PRIMARY KEY,
    city_id     INTEGER NOT NULL REFERENCES cities(city_id) ON DELETE CASCADE,
    name        VARCHAR(100),
    geom        GEOMETRY(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX idx_zones_geom ON zones USING GIST (geom);
CREATE INDEX idx_zones_city ON zones (city_id);

CREATE TABLE planning_areas (
    area_id     SERIAL PRIMARY KEY,
    zone_id     INTEGER NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
    name        VARCHAR(100),
    geom        GEOMETRY(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX idx_planning_areas_geom ON planning_areas USING GIST (geom);
CREATE INDEX idx_planning_areas_zone ON planning_areas (zone_id);

-- 100m x 100m analytical cells. Built in EPSG:32643 (UTM 43N) then stored as
-- 4326 for serving; see ml/src/data/grid_builder.py.
CREATE TABLE grid_cells (
    cell_id             SERIAL PRIMARY KEY,
    area_id             INTEGER NOT NULL REFERENCES planning_areas(area_id) ON DELETE CASCADE,
    geom                GEOMETRY(Polygon, 4326) NOT NULL,
    centroid            GEOMETRY(Point, 4326) NOT NULL,
    building_rooftop_sqm DOUBLE PRECISION,   -- moved here from planning_areas
    road_area_sqm        DOUBLE PRECISION,
    open_ground_sqm       DOUBLE PRECISION
);
CREATE INDEX idx_grid_cells_geom ON grid_cells USING GIST (geom);
CREATE INDEX idx_grid_cells_centroid ON grid_cells USING GIST (centroid);
CREATE INDEX idx_grid_cells_area ON grid_cells (area_id);

-- ---------------------------------------------------------------------------
-- Data versioning: which imagery window + model version produced a run
-- ---------------------------------------------------------------------------

CREATE TABLE data_versions (
    version_id      SERIAL PRIMARY KEY,
    label           VARCHAR(50) NOT NULL,       -- e.g. 'v1-2019-2025'
    imagery_start   DATE NOT NULL,
    imagery_end     DATE NOT NULL,
    model_name      VARCHAR(50) NOT NULL,       -- e.g. 'xgboost'
    model_version   VARCHAR(50) NOT NULL,       -- e.g. '2026-09-27'
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Per-cell, per-season feature values (NDVI, albedo, LULC fractions, etc.)
-- ---------------------------------------------------------------------------

CREATE TYPE season_type AS ENUM ('winter', 'pre_monsoon', 'monsoon', 'post_monsoon');

CREATE TABLE cell_features (
    feature_id          SERIAL PRIMARY KEY,
    cell_id             INTEGER NOT NULL REFERENCES grid_cells(cell_id) ON DELETE CASCADE,
    version_id          INTEGER NOT NULL REFERENCES data_versions(version_id),
    season              season_type NOT NULL,
    ndvi                DOUBLE PRECISION,
    ndbi                DOUBLE PRECISION,
    mndwi               DOUBLE PRECISION,
    albedo              DOUBLE PRECISION,
    built_up_fraction   DOUBLE PRECISION,
    impervious_fraction DOUBLE PRECISION,
    bare_fraction       DOUBLE PRECISION,
    vegetation_fraction DOUBLE PRECISION,
    water_fraction      DOUBLE PRECISION,
    road_density        DOUBLE PRECISION,       -- m of road per m^2 of cell
    elevation_m         DOUBLE PRECISION,
    dist_to_water_m     DOUBLE PRECISION,
    dist_to_coast_m     DOUBLE PRECISION,
    population_density  DOUBLE PRECISION,       -- people per km^2, from GHS-POP
    night_lights        DOUBLE PRECISION,        -- optional, VIIRS
    valid_obs_count     INTEGER,                 -- cloud-free scenes contributing to this cell/season
    UNIQUE (cell_id, version_id, season)
);
CREATE INDEX idx_cell_features_cell_season ON cell_features (cell_id, season);

-- ---------------------------------------------------------------------------
-- Model predictions: observed + predicted LST, anomaly, hotspot flag, SHAP
-- ---------------------------------------------------------------------------

CREATE TABLE heat_predictions (
    prediction_id       SERIAL PRIMARY KEY,
    cell_id             INTEGER NOT NULL REFERENCES grid_cells(cell_id) ON DELETE CASCADE,
    version_id          INTEGER NOT NULL REFERENCES data_versions(version_id),
    season              season_type NOT NULL,
    observed_lst_c       DOUBLE PRECISION,
    predicted_lst_c      DOUBLE PRECISION NOT NULL,
    city_anomaly_c       DOUBLE PRECISION,       -- predicted_lst - city median for that season
    rural_anomaly_c      DOUBLE PRECISION,       -- predicted_lst - rural reference ring (SUHI-style)
    is_hotspot           BOOLEAN NOT NULL DEFAULT FALSE,  -- moved here from planning_areas
    hotspot_method       VARCHAR(30),            -- 'percentile' | 'getis_ord_gi'
    hotspot_rank         INTEGER,
    shap_values          JSONB,                  -- {"ndvi": -1.2, "built_up_fraction": 0.9, ...}
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (cell_id, version_id, season)
);
CREATE INDEX idx_heat_predictions_cell_season ON heat_predictions (cell_id, season);
CREATE INDEX idx_heat_predictions_hotspot ON heat_predictions (season, is_hotspot) WHERE is_hotspot;
CREATE INDEX idx_heat_predictions_shap ON heat_predictions USING GIN (shap_values);

-- ---------------------------------------------------------------------------
-- Interventions catalogue (cool roof, tree planting, cool pavement, ...)
-- ---------------------------------------------------------------------------

CREATE TABLE interventions (
    intervention_id     SERIAL PRIMARY KEY,
    name                VARCHAR(50) NOT NULL UNIQUE,   -- 'cool_roof_limewash', 'tree_planting', ...
    surface_type         VARCHAR(20) NOT NULL,          -- 'rooftop' | 'road' | 'open_ground'
    unit_cost            DOUBLE PRECISION NOT NULL,      -- per m^2 or per unit, see unit_label
    unit_label            VARCHAR(20) NOT NULL DEFAULT 'per_sqm',
    annual_maintenance    DOUBLE PRECISION DEFAULT 0,
    lifetime_years        DOUBLE PRECISION,
    cooling_coefficient   DOUBLE PRECISION,               -- estimated deg C per unit area treated, see ml/src/optimization
    source                TEXT                            -- citation, e.g. Takacs et al. 2025
);

-- Feasibility caps: how much of each intervention physically fits in a cell
CREATE TABLE feasibility_caps (
    cap_id              SERIAL PRIMARY KEY,
    cell_id             INTEGER NOT NULL REFERENCES grid_cells(cell_id) ON DELETE CASCADE,
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    version_id          INTEGER NOT NULL REFERENCES data_versions(version_id),
    cap_sqm             DOUBLE PRECISION NOT NULL,
    assumptions         JSONB,                  -- e.g. {"usable_roof_fraction": 0.6}
    UNIQUE (cell_id, intervention_id, version_id)
);
CREATE INDEX idx_feasibility_caps_cell ON feasibility_caps (cell_id);

-- Recommendations: system-generated ranked suggestions per hotspot cell
CREATE TABLE recommendations (
    recommendation_id   SERIAL PRIMARY KEY,
    cell_id             INTEGER NOT NULL REFERENCES grid_cells(cell_id) ON DELETE CASCADE,
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    season              season_type NOT NULL,
    recommended_area_sqm DOUBLE PRECISION NOT NULL,
    estimated_cooling_c  DOUBLE PRECISION,
    estimated_cost        DOUBLE PRECISION,
    rank                  INTEGER,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_recommendations_cell ON recommendations (cell_id, season);

-- ---------------------------------------------------------------------------
-- What-if scenario simulations
-- ---------------------------------------------------------------------------

CREATE TABLE scenarios (
    scenario_id         SERIAL PRIMARY KEY,
    name                VARCHAR(100),
    cell_ids            INTEGER[] NOT NULL,       -- one or more cells included in the run
    season              season_type NOT NULL,
    intervention_mix    JSONB NOT NULL,           -- {"cool_roof_limewash": 1200, "tree_planting": 300}
    created_by          VARCHAR(100),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE scenario_results (
    result_id           SERIAL PRIMARY KEY,
    scenario_id         INTEGER NOT NULL REFERENCES scenarios(scenario_id) ON DELETE CASCADE,
    estimated_cooling_c  DOUBLE PRECISION,
    cooling_uncertainty_c DOUBLE PRECISION,
    total_cost            DOUBLE PRECISION,
    annualized_cost       DOUBLE PRECISION,
    cost_per_degree       DOUBLE PRECISION,
    binding_constraints   JSONB,                  -- which caps/budget were hit
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Pareto / NSGA-II optimization runs
-- ---------------------------------------------------------------------------

CREATE TABLE pareto_runs (
    run_id              SERIAL PRIMARY KEY,
    cell_ids             INTEGER[] NOT NULL,      -- candidate cells included (e.g. top-N hotspots)
    season               season_type NOT NULL,
    objectives            JSONB NOT NULL,          -- ["cost", "cooling", "equity"]
    algorithm             VARCHAR(30) NOT NULL DEFAULT 'nsga2',
    population_size       INTEGER,
    generations           INTEGER,
    budget_cap            DOUBLE PRECISION,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pareto_solutions (
    solution_id         SERIAL PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES pareto_runs(run_id) ON DELETE CASCADE,
    allocation          JSONB NOT NULL,           -- {"cell_id": {"intervention_id": area_sqm, ...}, ...}
    total_cost           DOUBLE PRECISION,
    total_cooling         DOUBLE PRECISION,
    equity_score          DOUBLE PRECISION,
    is_named_solution     VARCHAR(30)             -- 'cheapest_useful' | 'best_value' | 'max_cooling' | NULL
);
CREATE INDEX idx_pareto_solutions_run ON pareto_solutions (run_id);

-- ---------------------------------------------------------------------------
-- Auth (JWT-based, per the deck's FastAPI + Pydantic v2 gateway)
-- ---------------------------------------------------------------------------

CREATE TABLE users (
    user_id             SERIAL PRIMARY KEY,
    email               VARCHAR(255) NOT NULL UNIQUE,
    hashed_password     VARCHAR(255) NOT NULL,
    full_name           VARCHAR(100),
    role                VARCHAR(30) NOT NULL DEFAULT 'planner',  -- 'planner' | 'admin'
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
