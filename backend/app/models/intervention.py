"""Cooling intervention catalogue: cool roof, tree planting, cool pavement,
green roof. Unit costs are starting values from Takacs et al. (2025),
standardized to 2020 USD -- see plan.md section 7.3. Replace with SMC /
CPWD schedule-of-rates figures once available.
"""
from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Intervention(Base):
    __tablename__ = "interventions"

    intervention_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    surface_type: Mapped[str] = mapped_column(String(20))  # rooftop | road | open_ground
    unit_cost: Mapped[float] = mapped_column(Float)
    unit_label: Mapped[str] = mapped_column(String(20), default="per_sqm")
    annual_maintenance: Mapped[float] = mapped_column(Float, default=0.0)
    lifetime_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    cooling_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
