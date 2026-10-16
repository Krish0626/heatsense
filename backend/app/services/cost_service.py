"""Cost calculations for interventions: unit cost, maintenance, lifetime-
annualized cost. See plan.md section 7.3."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.intervention import Intervention


def get_intervention(db: Session, name: str) -> Intervention | None:
    stmt = select(Intervention).where(Intervention.name == name)
    return db.execute(stmt).scalar_one_or_none()


def annualized_cost(intervention: Intervention, area_sqm: float, horizon_years: float = 10.0) -> float:
    """Total cost of running one intervention over `horizon_years`, including
    reapplication where lifetime < horizon (e.g. limewash reapplied ~5x over
    10 years) plus yearly maintenance. See plan.md section 7.3.
    """
    if intervention.lifetime_years and intervention.lifetime_years > 0:
        reapplications = max(1, round(horizon_years / intervention.lifetime_years))
    else:
        reapplications = 1
    install_cost = intervention.unit_cost * area_sqm * reapplications
    maintenance_cost = intervention.annual_maintenance * area_sqm * horizon_years
    return install_cost + maintenance_cost
