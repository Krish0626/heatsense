"""GET /api/v1/grid/cells and GET /api/v1/grid/cells/{id}/prediction.

Matches FR-01 (multi-tier navigation) and FR-02 (dynamic viewport-based
100m grid loading) from the SRS in plan.md / the mid-sem deck.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.schemas.common import Season
from app.schemas.grid import PredictionOut
from app.services import spatial_service

router = APIRouter(prefix="/grid", tags=["grid"])


@router.get("/cells")
def list_cells(
    bbox: str = Query(..., description="min_lon,min_lat,max_lon,max_lat"),
    season: Season = Season.pre_monsoon,
    db: Session = Depends(get_db),
):
    """Return grid cells intersecting the viewport bbox as a GeoJSON
    FeatureCollection. Each feature carries the predicted LST and hotspot
    flag for the requested season (if a prediction row exists), enabling
    the frontend choropleth and hotspot overlay.

    Limit: 2000 cells per request. Use smaller viewport / higher zoom for
    dense areas. For production, prefer server-rendered vector tiles over
    per-move GeoJSON fetches (plan.md section 9.3).
    """
    try:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be 'min_lon,min_lat,max_lon,max_lat'")

    cells = spatial_service.get_cells_in_bbox(db, min_lon, min_lat, max_lon, max_lat)

    features = []
    for cell in cells:
        prediction = spatial_service.get_prediction_for_cell(db, cell.cell_id, season)
        features.append(
            spatial_service.cell_to_geojson_feature(db, cell, prediction)
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "cell_count": len(features),
    }


@router.get("/cells/{cell_id}/prediction", response_model=PredictionOut)
def get_cell_prediction(
    cell_id: int,
    season: Season = Season.pre_monsoon,
    db: Session = Depends(get_db),
):
    """Return the LST prediction + SHAP breakdown for one cell/season.
    Used by the ShapWaterfall panel when the user clicks a map cell.
    """
    prediction = spatial_service.get_prediction_for_cell(db, cell_id, season)
    if prediction is None:
        raise HTTPException(
            404,
            detail=(
                f"No prediction for cell_id={cell_id} season={season.value}. "
                "Run the ML pipeline (ml/src/models/train_xgboost.py) and "
                "shap_analysis.py to populate heat_predictions."
            ),
        )
    return PredictionOut(
        cell_id=prediction.cell_id,
        season=season,
        observed_lst_c=prediction.observed_lst_c,
        predicted_lst_c=prediction.predicted_lst_c,
        city_anomaly_c=prediction.city_anomaly_c,
        rural_anomaly_c=prediction.rural_anomaly_c,
        is_hotspot=prediction.is_hotspot,
        hotspot_rank=prediction.hotspot_rank,
        shap_values=prediction.shap_values,
    )
