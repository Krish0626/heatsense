"""Tracks which imagery window + model version produced a given batch of
predictions/features, so re-training doesn't silently overwrite history."""
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DataVersion(Base):
    __tablename__ = "data_versions"

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(50))
    imagery_start: Mapped[date] = mapped_column(Date)
    imagery_end: Mapped[date] = mapped_column(Date)
    model_name: Mapped[str] = mapped_column(String(50))
    model_version: Mapped[str] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
