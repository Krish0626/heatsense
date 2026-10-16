"""Request/response schemas for POST /api/v1/optimization/pareto.

Kept deliberately small: the heavy lifting (surrogate cooling coefficients,
NSGA-II run itself) lives in app/services/optimization_service.py and
ml/src/optimization/nsga2_optimizer.py.
"""
from pydantic import BaseModel, Field

from app.schemas.common import Season


class ParetoRequest(BaseModel):
    cell_ids: list[int] = Field(
        ..., min_length=1, description="Candidate cells, e.g. the top-N hotspots for the season"
    )
    season: Season
    objectives: list[str] = Field(default=["cost", "cooling"])
    budget_cap: float | None = None
    population_size: int = 100
    generations: int = 200


class ParetoSolutionOut(BaseModel):
    solution_id: int
    total_cost: float
    total_cooling: float
    equity_score: float | None = None
    is_named_solution: str | None = None
    allocation: dict = Field(
        ..., description="cell_id -> {intervention_name: area_sqm}"
    )


class ParetoRunOut(BaseModel):
    run_id: int
    season: Season
    objectives: list[str]
    solutions: list[ParetoSolutionOut]
