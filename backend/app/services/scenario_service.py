"""What-if scenario simulation: given a chosen intervention mix on a set of
cells, estimate cooling and cost. See plan.md section 7.4 and 7.5.

The counterfactual cooling estimate (re-running the trained XGBoost model
with perturbed features) lives in ml_service.py / the ml package, since it
needs the loaded model + feature pipeline. This module wires that estimate
together with feasibility clipping and cost, and persists the run.
"""
from sqlalchemy.orm import Session

from app.models.scenario import Scenario, ScenarioResult
from app.schemas.scenario import ScenarioRequest, ScenarioResultOut
from app.services import cost_service, feasibility_service, ml_service


def run_scenario(db: Session, request: ScenarioRequest) -> ScenarioResultOut:
    binding_constraints: dict[str, list[int]] = {}
    total_cost = 0.0
    used_mix: dict[str, float] = {name: 0.0 for name in request.intervention_mix}

    for cell_id in request.cell_ids:
        clipped = feasibility_service.clip_to_caps(db, cell_id, request.intervention_mix)
        for name, area in clipped.items():
            used_mix[name] += area
            if area < request.intervention_mix.get(name, 0.0):
                binding_constraints.setdefault(name, []).append(cell_id)

            intervention = cost_service.get_intervention(db, name)
            if intervention is not None:
                total_cost += cost_service.annualized_cost(intervention, area)

    # Counterfactual cooling estimate via the trained model (see plan.md 7.4).
    # Pass `db` so ml_service can load cell_features rows; it falls back to
    # (0.0, None) safely when no model artifact exists yet.
    estimated_cooling, uncertainty = ml_service.estimate_cooling(
        cell_ids=request.cell_ids,
        season=request.season.value,
        intervention_mix=used_mix,
        db=db,
    )

    scenario = Scenario(
        name=request.name,
        cell_ids=request.cell_ids,
        season=request.season.value,
        intervention_mix=request.intervention_mix,
    )
    db.add(scenario)
    db.flush()  # get scenario.scenario_id without committing yet
    scenario_id = scenario.scenario_id if scenario.scenario_id is not None else 1

    result = ScenarioResult(
        scenario_id=scenario_id,
        estimated_cooling_c=estimated_cooling,
        cooling_uncertainty_c=uncertainty,
        total_cost=total_cost,
        annualized_cost=total_cost,
        cost_per_degree=(total_cost / estimated_cooling) if estimated_cooling else None,
        binding_constraints=binding_constraints or None,
    )
    db.add(result)
    db.commit()

    return ScenarioResultOut(
        scenario_id=scenario_id,
        estimated_cooling_c=estimated_cooling,
        cooling_uncertainty_c=uncertainty,
        total_cost=total_cost,
        annualized_cost=total_cost,
        cost_per_degree=result.cost_per_degree,
        binding_constraints=result.binding_constraints,
        allocation_used=used_mix,
    )
