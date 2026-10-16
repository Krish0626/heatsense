"""Response schema for GET /api/v1/hotspots."""
from pydantic import BaseModel

from app.schemas.common import Season


class HotspotOut(BaseModel):
    cell_id: int
    season: Season
    predicted_lst_c: float
    rural_anomaly_c: float | None
    hotspot_rank: int | None
    hotspot_method: str | None
    top_drivers: list[str] = []  # top 3 SHAP feature names, filled by the service layer


class HotspotListOut(BaseModel):
    season: Season
    count: int
    hotspots: list[HotspotOut]
