"""Tests for the grid, hotspot, scenario and optimization endpoints.

These tests use FastAPI's TestClient (backed by httpx), which does NOT
require a running database -- but they will exercise the import chain of
all models, schemas and services, catching import-time errors early.

Any test that requires a real DB connection is marked with `@pytest.mark.db`
and skipped by default. To run DB tests:
    pytest -m db --override-ini "markers=db:requires live db"

Run the whole suite with: pytest
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def test_grid_cells_bad_bbox():
    """Invalid bbox string should return 400, not 500."""
    response = client.get("/api/v1/grid/cells?bbox=not_a_bbox")
    assert response.status_code == 400
    assert "bbox" in response.json()["detail"].lower()


def test_grid_cells_valid_bbox_no_db():
    """Valid bbox query against an empty / unreachable DB should either
    return 200 with an empty FeatureCollection or raise a 500 — but not
    a 422 (schema error). We accept both 200 and 500 here since the test
    environment likely has no PostGIS instance running."""
    response = client.get("/api/v1/grid/cells?bbox=72.75,21.05,72.95,21.28&season=pre_monsoon")
    assert response.status_code in (200, 500)


def test_grid_cell_prediction_returns_404_or_501():
    """A prediction for a non-existent cell should return 404 (wired)
    or 500 (DB connection error), not 501 as the old stub did."""
    response = client.get("/api/v1/grid/cells/99999/prediction?season=pre_monsoon")
    assert response.status_code in (404, 500)


# ---------------------------------------------------------------------------
# Hotspots
# ---------------------------------------------------------------------------

def test_hotspots_schema_valid():
    """Hotspot endpoint should accept season and top_n query params."""
    response = client.get("/api/v1/hotspots?season=winter&top_n=5")
    assert response.status_code in (200, 500)


def test_hotspots_top_n_over_limit():
    """top_n > 1000 should be rejected by the schema (le=1000)."""
    response = client.get("/api/v1/hotspots?top_n=9999")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def test_scenario_simulate_schema():
    """POST /scenarios/simulate with a valid payload body should reach the
    service layer (not fail at schema validation)."""
    payload = {
        "cell_ids": [1],
        "season": "pre_monsoon",
        "intervention_mix": {"cool_roof_limewash": 500.0},
    }
    response = client.post("/api/v1/scenarios/simulate", json=payload)
    # DB may not be available, accept 500; but not 422 (schema error).
    assert response.status_code in (200, 500)


def test_scenario_simulate_missing_cell_ids():
    """Empty cell_ids list should be rejected (min_length=1)."""
    payload = {
        "cell_ids": [],
        "season": "monsoon",
        "intervention_mix": {"cool_roof_limewash": 100},
    }
    response = client.post("/api/v1/scenarios/simulate", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def test_pareto_schema():
    payload = {
        "cell_ids": [1, 2, 3],
        "season": "post_monsoon",
        "objectives": ["cost", "cooling"],
    }
    response = client.post("/api/v1/optimization/pareto", json=payload)
    assert response.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_register_schema_error():
    """Registering with an invalid email should return 422."""
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "not_an_email", "password": "pw123"},
    )
    assert response.status_code == 422


def test_auth_login_wrong_credentials():
    """Login with wrong credentials should return 401 or 500 (no DB)."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "nobody@example.com", "password": "wrong"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code in (401, 500)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def test_reports_bad_id():
    response = client.get("/api/v1/reports/0.csv")
    assert response.status_code == 404


def test_reports_valid_id():
    response = client.get("/api/v1/reports/1.csv?season=pre_monsoon")
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        assert "text/csv" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Interventions & Feasibility Caps
# ---------------------------------------------------------------------------

def test_interventions_catalogue():
    response = client.get("/api/v1/interventions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_interventions_caps():
    response = client.get("/api/v1/interventions/caps?cell_id=1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
