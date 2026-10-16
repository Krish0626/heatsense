"""POST /api/v1/optimization/pareto -- NSGA-II cooling-vs-cost trade-off
(deck: "Pareto Optimization (NSGA-II)", currently partial/planned)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.schemas.optimization import ParetoRequest, ParetoRunOut
from app.services import optimization_service

router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.post("/pareto", response_model=ParetoRunOut)
def run_pareto(request: ParetoRequest, db: Session = Depends(get_db)):
    return optimization_service.run_pareto_optimization(db, request)
