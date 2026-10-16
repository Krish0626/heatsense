"""Aggregates every v1 endpoint router under /api/v1."""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, grid, hotspots, interventions, optimization, reports, scenarios

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(grid.router)
api_router.include_router(hotspots.router)
api_router.include_router(interventions.router)
api_router.include_router(scenarios.router)
api_router.include_router(optimization.router)
api_router.include_router(reports.router)

