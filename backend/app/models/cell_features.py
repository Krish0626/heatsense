"""Per-cell, per-season feature values used as model inputs.

Feature list follows plan.md section 6.3. valid_obs_count records how many
cloud-free scenes contributed to this cell/season, which matters most for
the monsoon season (see plan.md 5.3).
"""
from sqlalchemy import Enum, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

SEASONS = ("winter", "pre_monsoon", "monsoon", "post_monsoon")
SeasonType = Enum(*SEASONS, name="season_type")


class CellFeatures(Base):
    __tablename__ = "cell_features"

    feature_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.cell_id"))
    version_id: Mapped[int] = mapped_column(ForeignKey("data_versions.version_id"))
    season: Mapped[str] = mapped_column(SeasonType)

    ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndbi: Mapped[float | None] = mapped_column(Float, nullable=True)
    mndwi: Mapped[float | None] = mapped_column(Float, nullable=True)
    albedo: Mapped[float | None] = mapped_column(Float, nullable=True)
    built_up_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    impervious_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    bare_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    vegetation_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    water_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    road_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_to_water_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_to_coast_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    population_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    night_lights: Mapped[float | None] = mapped_column(Float, nullable=True)
    valid_obs_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Keep this list in sync with ml/src/data/feature_engineering.py::FEATURE_COLUMNS
