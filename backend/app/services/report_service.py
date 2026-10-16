"""Zone and planning-area aggregation for the CSV export endpoint.

Aggregates grid-cell-level heat predictions up to planning-area level:
mean predicted LST, hotspot count, top SHAP drivers, and recommended
interventions per planning area. See plan.md section 9.2 (FR-08).
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.geography import GridCell, PlanningArea, Zone
from app.models.heat_prediction import HeatPrediction
from app.models.feasibility import Recommendation
from app.models.intervention import Intervention
from app.schemas.common import Season


def zone_summary_for_season(db: Session, season: Season) -> list[dict]:
    """Return one row per planning area with:
      - area_id, area_name, zone_name
      - mean_predicted_lst_c
      - hotspot_count (cells flagged is_hotspot for this season)
      - cell_count (total cells in the area)
      - top_intervention: the most-frequently recommended intervention name
        for cells in this area, or None if no recommendations exist yet.

    Returns rows ordered by mean_predicted_lst_c descending (hottest first).
    """
    # Aggregation: mean LST and hotspot count per planning area, for the
    # requested season, joining via grid_cells.
    agg_stmt = (
        select(
            GridCell.area_id,
            func.count(GridCell.cell_id).label("cell_count"),
            func.avg(HeatPrediction.predicted_lst_c).label("mean_predicted_lst_c"),
            func.sum(
                func.cast(HeatPrediction.is_hotspot, type_=None)
            ).label("hotspot_count"),
        )
        .join(HeatPrediction, HeatPrediction.cell_id == GridCell.cell_id)
        .where(HeatPrediction.season == season.value)
        .group_by(GridCell.area_id)
        .subquery()
    )

    # Pull planning area + zone names alongside the aggregation.
    stmt = (
        select(
            PlanningArea.area_id,
            PlanningArea.name.label("area_name"),
            Zone.name.label("zone_name"),
            agg_stmt.c.cell_count,
            agg_stmt.c.mean_predicted_lst_c,
            agg_stmt.c.hotspot_count,
        )
        .join(agg_stmt, agg_stmt.c.area_id == PlanningArea.area_id)
        .join(Zone, Zone.zone_id == PlanningArea.zone_id)
        .order_by(agg_stmt.c.mean_predicted_lst_c.desc())
    )
    rows = db.execute(stmt).all()

    # Pull top recommended intervention per planning area (mode of
    # intervention_id in recommendations for cells in each area).
    # Done as a Python aggregation over a second query to keep SQL readable.
    rec_stmt = (
        select(
            GridCell.area_id,
            Intervention.name.label("intervention_name"),
            func.count().label("rec_count"),
        )
        .join(Recommendation, Recommendation.cell_id == GridCell.cell_id)
        .join(Intervention, Intervention.intervention_id == Recommendation.intervention_id)
        .where(Recommendation.season == season.value)
        .group_by(GridCell.area_id, Intervention.name)
    )
    rec_rows = db.execute(rec_stmt).all()

    # Build area_id -> top intervention name map.
    from collections import defaultdict
    area_rec: dict[int, dict[str, int]] = defaultdict(dict)
    for r in rec_rows:
        area_rec[r.area_id][r.intervention_name] = r.rec_count
    top_intervention: dict[int, str | None] = {
        aid: max(counts, key=counts.get)
        for aid, counts in area_rec.items()
    }

    result = []
    for row in rows:
        result.append(
            {
                "area_id": row.area_id,
                "area_name": row.area_name or f"Area {row.area_id}",
                "zone_name": row.zone_name or "Unknown Zone",
                "cell_count": int(row.cell_count or 0),
                "mean_predicted_lst_c": (
                    round(float(row.mean_predicted_lst_c), 2)
                    if row.mean_predicted_lst_c is not None
                    else None
                ),
                "hotspot_count": int(row.hotspot_count or 0),
                "top_intervention": top_intervention.get(row.area_id),
            }
        )
    return result
