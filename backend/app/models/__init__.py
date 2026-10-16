"""Import every model so app.db.base.Base.metadata is complete (needed for
Alembic autogeneration and for create_all() in dev/test setups)."""
from app.models.geography import City, GridCell, PlanningArea, Zone  # noqa: F401
from app.models.data_version import DataVersion  # noqa: F401
from app.models.cell_features import CellFeatures  # noqa: F401
from app.models.heat_prediction import HeatPrediction  # noqa: F401
from app.models.intervention import Intervention  # noqa: F401
from app.models.feasibility import FeasibilityCap, Recommendation  # noqa: F401
from app.models.scenario import Scenario, ScenarioResult  # noqa: F401
from app.models.optimization import ParetoRun, ParetoSolution  # noqa: F401
from app.models.user import User  # noqa: F401
