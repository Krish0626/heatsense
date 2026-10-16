"""Request/response schemas for GET /api/v1/grid/cells and
GET /api/v1/grid/cells/{id}/prediction."""
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import Season


class BBox(BaseModel):
    """Viewport bounding box, WGS84 degrees."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class GridCellOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cell_id: int
    area_id: int
    geom_geojson: dict = Field(..., description="Cell polygon as GeoJSON")
    building_rooftop_sqm: float | None = None
    road_area_sqm: float | None = None
    open_ground_sqm: float | None = None


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cell_id: int
    season: Season
    observed_lst_c: float | None = None
    predicted_lst_c: float
    city_anomaly_c: float | None = None
    rural_anomaly_c: float | None = None
    is_hotspot: bool
    hotspot_rank: int | None = None
    shap_values: dict[str, float] | None = Field(
        default=None,
        description="Feature name -> SHAP contribution in degrees C for this cell/season",
    )
