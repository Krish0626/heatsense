"""Database initialisation helper for development and CI.

Creates all tables defined in backend/app/models/ via SQLAlchemy
Base.metadata.create_all(), then seeds the interventions catalogue
(unit costs, lifetimes, and cooling coefficients from Takacs et al. 2025
and the literature review in plan.md section 7.3).

Usage:
    # From the backend/ directory:
    python -m app.db.init_db

    # Or with a custom DB URL:
    DATABASE_URL=postgresql+psycopg://... python -m app.db.init_db
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.base import Base
from app.db.database import SessionLocal, engine
import app.models  # noqa: F401 -- registers all ORM models on Base.metadata


# ---------------------------------------------------------------------------
# Seed data: interventions catalogue
# Values from Takacs et al. (2025), npj Urban Sustainability, and the
# supporting literature review in plan.md section 7.3.
# Unit cost is USD per m² (2020 USD, from the NRDC 2018 India study).
# Convert to INR at a stated rate where needed; these are indicative.
# ---------------------------------------------------------------------------

INTERVENTIONS_SEED = [
    {
        "name": "cool_roof_limewash",
        "surface_type": "rooftop",
        "unit_cost": 0.75,           # USD/m², re-applied ~5x over 10 years
        "unit_label": "per_sqm",
        "annual_maintenance": 0.0,
        "lifetime_years": 2.0,       # effective life before reapplication
        "cooling_coefficient": None,  # set after model is trained
        "source": "Takacs et al. (2025) npj Urban Sustainability; NRDC 2018 India",
    },
    {
        "name": "cool_roof_hdpe",
        "surface_type": "rooftop",
        "unit_cost": 2.10,           # USD/m², 5-year lifetime
        "unit_label": "per_sqm",
        "annual_maintenance": 0.0,
        "lifetime_years": 5.0,
        "cooling_coefficient": None,
        "source": "Takacs et al. (2025) npj Urban Sustainability",
    },
    {
        "name": "tree_planting",
        "surface_type": "open_ground",
        "unit_cost": 2.50,           # USD per tree; 2 trees / 100 m² assumed
        "unit_label": "per_sqm",     # converted: 2 trees/100m² × 2.50 = 0.05 USD/m²
        "annual_maintenance": 5.0 * 2 / 100,  # 5 USD/tree/year × 2/100m²
        "lifetime_years": None,      # trees are semi-permanent; use 10-yr horizon
        "cooling_coefficient": None,
        "source": (
            "Takacs et al. (2025): ~0.43°C per 10% extra tree canopy; "
            "tree meta-analysis: up to 12°C pedestrian-level (Comm. Earth & Env., 2024)"
        ),
    },
    {
        "name": "cool_pavement",
        "surface_type": "road",
        "unit_cost": 3.50,           # TODO: replace with schedule-of-rates figure;
        "unit_label": "per_sqm",     # placeholder from US studies (plan.md section 7.3)
        "annual_maintenance": 0.10,
        "lifetime_years": 10.0,
        "cooling_coefficient": None,
        "source": (
            "Placeholder (no India unit cost found in literature sweep, plan.md TODO). "
            "Cooling: up to 12°C surface temp drop at full albedo increase "
            "(cool pavements meta-analysis, Sci. of Total Env., 2023)."
        ),
    },
    {
        "name": "green_roof",
        "surface_type": "rooftop",
        "unit_cost": 15.0,           # USD/m², installed; wide range in literature
        "unit_label": "per_sqm",
        "annual_maintenance": 1.0,
        "lifetime_years": 40.0,
        "cooling_coefficient": None,
        "source": (
            "Stretch goal (plan.md section 7.1). "
            "Cooling: ~4°C daytime surface UHI at full coverage (ERL, 2014). "
            "Cost is a broad placeholder."
        ),
    },
]


def init_db(drop_first: bool = False) -> None:
    """Create tables and seed reference data.

    Args:
        drop_first: if True, drop all tables before recreating (DANGEROUS in
                    production; use only in dev/test environments).
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        if drop_first:
            Base.metadata.drop_all(bind=conn)
            print("[init_db] Dropped all tables.")
        Base.metadata.create_all(bind=conn)
        print("[init_db] Tables created (or already exist).")

    _seed_interventions()


def _seed_interventions() -> None:
    """Insert the interventions catalogue if the table is empty."""
    from app.models.intervention import Intervention
    from sqlalchemy import select

    with SessionLocal() as db:
        existing = db.execute(select(Intervention)).scalars().first()
        if existing is not None:
            print("[init_db] Interventions already seeded, skipping.")
            return

        for row in INTERVENTIONS_SEED:
            db.add(Intervention(**row))
        db.commit()
        print(f"[init_db] Seeded {len(INTERVENTIONS_SEED)} intervention rows.")


if __name__ == "__main__":
    init_db()
    print("[init_db] Done.")
