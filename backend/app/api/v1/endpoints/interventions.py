"""GET /api/v1/interventions -- list the intervention catalogue.

Used by the frontend to populate the ScenarioBuilder dynamically instead of
hardcoding intervention names. Returns all rows from the interventions table
(seeded by app/db/init_db.py with costs and lifetimes from plan.md §7.3).
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.models.intervention import Intervention

router = APIRouter(prefix="/interventions", tags=["interventions"])


class InterventionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    intervention_id: int
    name: str
    surface_type: str
    unit_cost: float
    unit_label: str
    annual_maintenance: float | None = None
    lifetime_years: float | None = None
    cooling_coefficient: float | None = None
    source: str | None = None


class FeasibilityCapOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cell_id: int
    intervention_name: str
    surface_type: str
    cap_sqm: float
    unit_cost: float
    assumptions: dict | None = None


@router.get("", response_model=list[InterventionOut])
def list_interventions(db: Session = Depends(get_db)):
    """Return all intervention catalogue rows.

    These are seeded at startup by app/db/init_db.py with indicative unit
    costs from Takacs et al. (2025) / NRDC 2018 India. Cooling coefficients
    are set to None until the ML pipeline has been run.
    """
    interventions = db.execute(select(Intervention).order_by(Intervention.intervention_id)).scalars().all()
    return [InterventionOut.model_validate(iv) for iv in interventions]


@router.get("/caps", response_model=list[FeasibilityCapOut])
def get_cell_caps(
    cell_id: int = Query(..., description="Grid cell ID to retrieve feasibility bounds for"),
    db: Session = Depends(get_db),
):
    """Return the feasibility caps for all interventions in a given cell.

    Checks the feasibility_caps table first; if not populated yet, derives
    caps dynamically from the grid cell surface areas (building rooftop, road area,
    open ground) per plan.md section 7.2.
    """
    from app.models.feasibility import FeasibilityCap
    from app.models.geography import GridCell

    # 1. Check if feasibility_caps rows exist in DB
    caps_stmt = (
        select(FeasibilityCap, Intervention)
        .join(Intervention, FeasibilityCap.intervention_id == Intervention.intervention_id)
        .where(FeasibilityCap.cell_id == cell_id)
    )
    cap_rows = db.execute(caps_stmt).all()
    if cap_rows:
        return [
            FeasibilityCapOut(
                cell_id=cell_id,
                intervention_name=iv.name,
                surface_type=iv.surface_type,
                cap_sqm=cap.cap_sqm,
                unit_cost=iv.unit_cost,
                assumptions=cap.assumptions,
            )
            for cap, iv in cap_rows
        ]

    # 2. Fallback: compute from GridCell geometry properties
    cell = db.execute(select(GridCell).where(GridCell.cell_id == cell_id)).scalar_one_or_none()
    interventions = db.execute(select(Intervention).order_by(Intervention.intervention_id)).scalars().all()

    bldg = cell.building_rooftop_sqm if (cell and cell.building_rooftop_sqm is not None) else 1500.0
    road = cell.road_area_sqm if (cell and cell.road_area_sqm is not None) else 800.0
    ground = cell.open_ground_sqm if (cell and cell.open_ground_sqm is not None) else 2000.0

    defaults_map = {
        "cool_roof_limewash": (bldg * 0.6, "rooftop", {"usable_fraction": 0.6}),
        "cool_roof_hdpe": (bldg * 0.6, "rooftop", {"usable_fraction": 0.6}),
        "green_roof": (bldg * 0.4, "rooftop", {"usable_fraction": 0.4}),
        "tree_planting": (ground * 0.8, "open_ground", {"plantable_fraction": 0.8}),
        "cool_pavement": (road * 0.7, "road", {"smc_control_fraction": 0.7}),
    }

    results = []
    for iv in interventions:
        cap_val, s_type, assumptions = defaults_map.get(
            iv.name, (1000.0, iv.surface_type, {"default": True})
        )
        results.append(
            FeasibilityCapOut(
                cell_id=cell_id,
                intervention_name=iv.name,
                surface_type=iv.surface_type or s_type,
                cap_sqm=round(cap_val, 1),
                unit_cost=iv.unit_cost,
                assumptions=assumptions,
            )
        )
    return results
