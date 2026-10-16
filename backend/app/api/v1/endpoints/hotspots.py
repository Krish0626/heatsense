"""GET /api/v1/hotspots -- ranked hotspot list per season (FR-04)."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.schemas.common import Season
from app.schemas.hotspot import HotspotListOut, HotspotOut
from app.services import spatial_service

router = APIRouter(prefix="/hotspots", tags=["hotspots"])


@router.get("", response_model=HotspotListOut)
def list_hotspots(
    season: Season = Season.pre_monsoon,
    top_n: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
):
    predictions = spatial_service.get_hotspots(db, season, top_n=top_n)
    hotspots = [
        HotspotOut(
            cell_id=p.cell_id,
            season=season,
            predicted_lst_c=p.predicted_lst_c,
            rural_anomaly_c=p.rural_anomaly_c,
            hotspot_rank=p.hotspot_rank,
            hotspot_method=p.hotspot_method,
            top_drivers=spatial_service.top_shap_drivers(p.shap_values),
        )
        for p in predictions
    ]
    return HotspotListOut(season=season, count=len(hotspots), hotspots=hotspots)
