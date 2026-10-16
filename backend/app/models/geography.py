"""Shared administrative-hierarchy models: City, Zone, PlanningArea, GridCell.

Mirrors database/schema.sql. Geometry columns use GeoAlchemy2's Geometry
type; SRID 4326 for anything served to the frontend, built internally in
EPSG:32643 (UTM 43N) by ml/src/data/grid_builder.py so the 100m cells are
true squares on the ground.
"""
from geoalchemy2 import Geometry
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class City(Base):
    __tablename__ = "cities"

    city_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100))
    boundary = mapped_column(Geometry("MULTIPOLYGON", srid=4326))

    zones: Mapped[list["Zone"]] = relationship(back_populates="city")


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.city_id"))
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    geom = mapped_column(Geometry("MULTIPOLYGON", srid=4326))

    city: Mapped["City"] = relationship(back_populates="zones")
    planning_areas: Mapped[list["PlanningArea"]] = relationship(back_populates="zone")


class PlanningArea(Base):
    __tablename__ = "planning_areas"

    area_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.zone_id"))
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    geom = mapped_column(Geometry("MULTIPOLYGON", srid=4326))

    zone: Mapped["Zone"] = relationship(back_populates="planning_areas")
    grid_cells: Mapped[list["GridCell"]] = relationship(back_populates="planning_area")


class GridCell(Base):
    """A single 100m x 100m analytical cell.

    Note: is_hotspot and shap_values live on HeatPrediction (per cell *and*
    season), not here -- a cell itself isn't a hotspot, it's a hotspot for a
    given season. See plan.md section 2.2.
    """

    __tablename__ = "grid_cells"

    cell_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    area_id: Mapped[int] = mapped_column(ForeignKey("planning_areas.area_id"))
    geom = mapped_column(Geometry("POLYGON", srid=4326))
    centroid = mapped_column(Geometry("POINT", srid=4326))
    building_rooftop_sqm: Mapped[float | None] = mapped_column(Float, nullable=True)
    road_area_sqm: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_ground_sqm: Mapped[float | None] = mapped_column(Float, nullable=True)

    planning_area: Mapped["PlanningArea"] = relationship(back_populates="grid_cells")
