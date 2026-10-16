"""POST /api/v1/scenarios/simulate -- what-if intervention simulation
(FR-06, FR-07)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.schemas.scenario import ScenarioRequest, ScenarioResultOut
from app.services import scenario_service

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.post("/simulate", response_model=ScenarioResultOut)
def simulate_scenario(request: ScenarioRequest, db: Session = Depends(get_db)):
    return scenario_service.run_scenario(db, request)
