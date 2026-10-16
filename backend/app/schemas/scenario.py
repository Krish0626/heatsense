"""Request/response schemas for POST /api/v1/scenarios/simulate.

intervention_mix keys are Intervention.name values (e.g. "cool_roof_limewash"),
values are area in square metres the planner wants to allocate; the service
layer clips each to the cell's feasibility cap before simulating.
"""
from pydantic import BaseModel, Field

from app.schemas.common import Season


class ScenarioRequest(BaseModel):
    cell_ids: list[int] = Field(..., min_length=1)
    season: Season
    intervention_mix: dict[str, float] = Field(
        ..., description="intervention name -> requested area in sqm"
    )
    name: str | None = None


class ScenarioResultOut(BaseModel):
    scenario_id: int
    estimated_cooling_c: float
    cooling_uncertainty_c: float | None = None
    total_cost: float
    annualized_cost: float | None = None
    cost_per_degree: float | None = None
    binding_constraints: dict | None = None
    allocation_used: dict[str, float] = Field(
        ..., description="intervention name -> area actually used, after clipping to caps"
    )
