"""What-if scenario runs: a chosen intervention mix on a set of cells, and
the resulting cooling/cost estimate. See plan.md section 7.5."""
from datetime import datetime

from sqlalchemy import ARRAY, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.cell_features import SeasonType


class Scenario(Base):
    __tablename__ = "scenarios"

    scenario_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    cell_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    season: Mapped[str] = mapped_column(SeasonType)
    intervention_mix: Mapped[dict] = mapped_column(JSONB)  # {"cool_roof_limewash": 1200, ...}
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScenarioResult(Base):
    __tablename__ = "scenario_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.scenario_id"))
    estimated_cooling_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    cooling_uncertainty_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    annualized_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_degree: Mapped[float | None] = mapped_column(Float, nullable=True)
    binding_constraints: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
