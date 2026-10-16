"""Shared FastAPI dependencies for the v1 API (currently just re-exports
get_db; auth dependency can be added here once login is wired up)."""
from app.db.database import get_db  # noqa: F401
