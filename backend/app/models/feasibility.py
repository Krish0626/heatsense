"""Feasibility caps (how much of each intervention physically fits in a
cell) and recommendations (ranked system suggestions). Kept as two tables,
not one combined box like the deck's ER diagram -- caps are inputs,
recommendations are derived outputs. See plan.md section 2.2 and 7.2.
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.cell_features import SeasonType


class FeasibilityCap(Base):
    __tablename__ = "feasibility_caps"

    cap_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.cell_id"))
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.intervention_id"))
    version_id: Mapped[int] = mapped_column(ForeignKey("data_versions.version_id"))
    cap_sqm: Mapped[float] = mapped_column(Float)
    assumptions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class Recommendation(Base):
    __tablename__ = "recommendations"

    recommendation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.cell_id"))
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.intervention_id"))
    season: Mapped[str] = mapped_column(SeasonType)
    recommended_area_sqm: Mapped[float] = mapped_column(Float)
    estimated_cooling_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
