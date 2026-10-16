"""Model output per cell and season: observed/predicted LST, anomaly,
hotspot flag, and the SHAP attribution for that prediction.

is_hotspot and shap_values were moved here from planning_areas in the
original deck design -- both are per-cell-per-season facts, not
per-planning-area facts. See plan.md section 2.2.
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.cell_features import SeasonType


class HeatPrediction(Base):
    __tablename__ = "heat_predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.cell_id"))
    version_id: Mapped[int] = mapped_column(ForeignKey("data_versions.version_id"))
    season: Mapped[str] = mapped_column(SeasonType)

    observed_lst_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_lst_c: Mapped[float] = mapped_column(Float)
    city_anomaly_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    rural_anomaly_c: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_hotspot: Mapped[bool] = mapped_column(Boolean, default=False)
    hotspot_method: Mapped[str | None] = mapped_column(String(30), nullable=True)
    hotspot_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    shap_values: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
