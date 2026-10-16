"""Feasibility caps: how much of each intervention physically fits in a cell.

See plan.md section 7.2. Real cap computation needs the same feature layers
as the model (building footprints, road network, land cover fractions) --
this module currently exposes the lookup + clipping logic; the caps
themselves are computed offline by a not-yet-written
ml/src/data/feasibility_caps.py and loaded into the feasibility_caps table.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.feasibility import FeasibilityCap
from app.models.intervention import Intervention


def get_cap(db: Session, cell_id: int, intervention_name: str) -> float:
    """Return the feasibility cap (m^2) for one cell + intervention, or 0.0
    if no cap row exists yet (treated as "not feasible here")."""
    stmt = (
        select(FeasibilityCap.cap_sqm)
        .join(Intervention, Intervention.intervention_id == FeasibilityCap.intervention_id)
        .where(FeasibilityCap.cell_id == cell_id, Intervention.name == intervention_name)
    )
    result = db.execute(stmt).scalar_one_or_none()
    return result or 0.0


def clip_to_caps(db: Session, cell_id: int, intervention_mix: dict[str, float]) -> dict[str, float]:
    """Clip a requested {intervention_name: area_sqm} mix down to what's
    physically feasible in this cell. Returns the clipped mix."""
    clipped: dict[str, float] = {}
    for name, requested_area in intervention_mix.items():
        cap = get_cap(db, cell_id, name)
        clipped[name] = min(requested_area, cap)
    return clipped
