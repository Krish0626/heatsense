"""Viewport-based grid queries and hotspot listing.

Backs GET /api/v1/grid/cells (FR-02: dynamic viewport-based 100m grid
loading) and GET /api/v1/hotspots (FR-04: automated hotspot detection and
ranking). Keeping this out of the route handler makes it reusable from the
scenario/optimization services, which also need "cells in this area".
"""
import json

from geoalchemy2.functions import ST_AsGeoJSON, ST_Intersects, ST_MakeEnvelope
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.geography import GridCell
from app.models.heat_prediction import HeatPrediction
from app.schemas.common import Season


def get_cells_in_bbox(
    db: Session, min_lon: float, min_lat: float, max_lon: float, max_lat: float, limit: int = 2000
) -> list[GridCell]:
    """Return grid cells whose geometry intersects the given WGS84 bounding box.

    Uses PostGIS ST_MakeEnvelope + ST_Intersects, which hits the GiST index on
    grid_cells.geom built by database/schema.sql.
    """
    envelope = ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
    stmt = (
        select(GridCell)
        .where(ST_Intersects(GridCell.geom, envelope))
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def cell_to_geojson_feature(
    db: Session,
    cell: GridCell,
    prediction: HeatPrediction | None = None,
) -> dict:
    """Serialize one GridCell to a GeoJSON Feature dict.

    Geometry is retrieved via ST_AsGeoJSON(geom) so we never need to load the
    full WKB into Python -- PostGIS does the conversion server-side.
    Properties include the prediction fields (LST, anomaly, hotspot flag) when
    a HeatPrediction row is provided.
    """
    geom_json_str = db.execute(
        select(func.cast(ST_AsGeoJSON(GridCell.geom), type_=None))
        .where(GridCell.cell_id == cell.cell_id)
    ).scalar_one()

    props: dict = {
        "cell_id": cell.cell_id,
        "area_id": cell.area_id,
        "building_rooftop_sqm": cell.building_rooftop_sqm,
        "road_area_sqm": cell.road_area_sqm,
        "open_ground_sqm": cell.open_ground_sqm,
    }
    if prediction is not None:
        props.update(
            {
                "predicted_lst_c": prediction.predicted_lst_c,
                "city_anomaly_c": prediction.city_anomaly_c,
                "rural_anomaly_c": prediction.rural_anomaly_c,
                "is_hotspot": prediction.is_hotspot,
                "hotspot_rank": prediction.hotspot_rank,
            }
        )

    return {
        "type": "Feature",
        "geometry": json.loads(geom_json_str),
        "properties": props,
    }


def get_prediction_for_cell(
    db: Session, cell_id: int, season: Season
) -> HeatPrediction | None:
    """Return the latest HeatPrediction for a cell/season, or None."""
    stmt = (
        select(HeatPrediction)
        .where(
            HeatPrediction.cell_id == cell_id,
            HeatPrediction.season == season.value,
        )
        .order_by(HeatPrediction.version_id.desc())
        .limit(1)
    )
    return db.execute(stmt).scalar_one_or_none()


def get_hotspots(db: Session, season: Season, top_n: int = 100) -> list[HeatPrediction]:
    """Return the top-N hotspot cells for a season, ranked by hotspot_rank."""
    stmt = (
        select(HeatPrediction)
        .where(HeatPrediction.season == season.value, HeatPrediction.is_hotspot.is_(True))
        .order_by(HeatPrediction.hotspot_rank.asc())
        .limit(top_n)
    )
    return list(db.execute(stmt).scalars().all())


def top_shap_drivers(shap_values: dict[str, float] | None, n: int = 3) -> list[str]:
    """Return the n feature names with the largest absolute SHAP contribution."""
    if not shap_values:
        return []
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [name for name, _ in ranked[:n]]
