"""NSGA-II Pareto optimization runs and their non-dominated solutions.
See plan.md section 8."""
from datetime import datetime

from sqlalchemy import ARRAY, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.cell_features import SeasonType


class ParetoRun(Base):
    __tablename__ = "pareto_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cell_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))  # candidate cells, e.g. top-N hotspots
    season: Mapped[str] = mapped_column(SeasonType)
    objectives: Mapped[list] = mapped_column(JSONB)  # ["cost", "cooling", "equity"]
    algorithm: Mapped[str] = mapped_column(String(30), default="nsga2")
    population_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    budget_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ParetoSolution(Base):
    __tablename__ = "pareto_solutions"

    solution_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("pareto_runs.run_id"))
    allocation: Mapped[dict] = mapped_column(JSONB)  # {cell_id: {intervention_id: area_sqm}}
    total_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cooling: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_named_solution: Mapped[str | None] = mapped_column(String(30), nullable=True)
