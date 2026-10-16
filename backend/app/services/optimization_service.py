"""Wires the NSGA-II optimizer (ml/src/optimization/nsga2_optimizer.py) to
the API: builds the candidate set, cooling-coefficient surrogate, and cost
model, runs the optimization, and persists the Pareto front.

See plan.md section 8. Actual NSGA-II logic (via pymoo) lives in ml/ so it
can be developed/tested independently of the API; this module calls into it
with the data assembled from the DB.

The optimizer works at a surrogate level (precomputed cooling coefficients
per cell/intervention) so XGBoost is NOT re-run inside every fitness
evaluation -- only once per cell/intervention pair when building the
CandidateCell list. See plan.md section 8.2.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.feasibility import FeasibilityCap
from app.models.intervention import Intervention
from app.models.geography import GridCell
from app.models.heat_prediction import HeatPrediction
from app.models.optimization import ParetoRun, ParetoSolution
from app.schemas.common import Season
from app.schemas.optimization import ParetoRequest, ParetoRunOut, ParetoSolutionOut
from app.services import ml_service


def _build_candidate_cells(
    db: Session,
    cell_ids: list[int],
    season: Season,
    intervention_names: list[str],
) -> list:
    """Build the CandidateCell list consumed by nsga2_optimizer.optimize().

    For each (cell, intervention) pair:
      - Feasibility cap comes from feasibility_caps table.
      - Unit cost comes from interventions table.
      - Cooling coefficient is estimated via ml_service.estimate_cooling
        on a unit (1 sqm) perturbation, then scaled.

    Cells or interventions missing feasibility cap rows are skipped.
    """
    # Add ml/ to PYTHONPATH once so the nsga2_optimizer import works.
    ml_src = str(Path(__file__).resolve().parents[4] / "ml")
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    from src.optimization.nsga2_optimizer import CandidateCell  # noqa: PLC0415

    # Prefetch interventions.
    interventions: dict[str, Intervention] = {
        iv.name: iv
        for iv in db.execute(
            select(Intervention).where(Intervention.name.in_(intervention_names))
        ).scalars().all()
    }

    candidates = []
    for cell_id in cell_ids:
        # Population weight: use population_density from latest cell_features row.
        from app.models.cell_features import CellFeatures  # noqa: PLC0415
        cf_row = db.execute(
            select(CellFeatures)
            .where(CellFeatures.cell_id == cell_id, CellFeatures.season == season.value)
            .order_by(CellFeatures.version_id.desc())
            .limit(1)
        ).scalar_one_or_none()
        pop_weight = float(cf_row.population_density or 1.0) if cf_row else 1.0

        cell_interventions: list[str] = []
        caps: list[float] = []
        coeffs: list[float] = []
        costs: list[float] = []

        for iv_name in intervention_names:
            iv = interventions.get(iv_name)
            if iv is None:
                continue

            cap_row = db.execute(
                select(FeasibilityCap.cap_sqm)
                .join(Intervention, Intervention.intervention_id == FeasibilityCap.intervention_id)
                .where(
                    FeasibilityCap.cell_id == cell_id,
                    Intervention.name == iv_name,
                )
            ).scalar_one_or_none()

            if cap_row is None or cap_row <= 0:
                continue

            # Cooling coefficient: deg C cooling per sqm treated.
            # Use a 10-sqm probe via estimate_cooling; divide by probe area.
            probe_sqm = 10.0
            cooling_at_probe, _ = ml_service.estimate_cooling(
                cell_ids=[cell_id],
                season=season.value,
                intervention_mix={iv_name: probe_sqm},
                db=db,
            )
            coeff = cooling_at_probe / probe_sqm if probe_sqm > 0 else 0.0

            cell_interventions.append(iv_name)
            caps.append(float(cap_row))
            coeffs.append(coeff)
            costs.append(float(iv.unit_cost))

        if not cell_interventions:
            continue  # no feasible interventions for this cell

        candidates.append(
            CandidateCell(
                cell_id=cell_id,
                interventions=cell_interventions,
                caps_sqm=caps,
                cooling_coeffs=coeffs,
                unit_costs=costs,
                population_weight=pop_weight,
            )
        )
    return candidates


def _pareto_result_to_solutions(
    db: Session,
    run: ParetoRun,
    result,
    problem,
) -> list[ParetoSolution]:
    """Convert pymoo Result + CoolingCostProblem into ParetoSolution ORM rows."""
    if result is None or result.X is None:
        return []

    solutions: list[ParetoSolution] = []
    F = result.F  # objective values [total_cost, -total_cooling, (-equity)?]

    # Identify named solutions: cheapest_useful (min cost among cooling>0),
    # best_value (min cost/cooling ratio), max_cooling (max cooling).
    cooling_vals = -F[:, 1]
    cost_vals = F[:, 0]

    best_value_idx = None
    min_ratio = float("inf")
    max_cooling_idx = int(cooling_vals.argmax())
    cheapest_idx = None

    for i, (cost, cooling) in enumerate(zip(cost_vals, cooling_vals)):
        if cooling > 0:
            if cheapest_idx is None or cost < cost_vals[cheapest_idx]:
                cheapest_idx = i
            ratio = cost / cooling
            if ratio < min_ratio:
                min_ratio = ratio
                best_value_idx = i

    named: dict[int, str] = {}
    if cheapest_idx is not None:
        named[cheapest_idx] = "cheapest_useful"
    if best_value_idx is not None and best_value_idx != cheapest_idx:
        named[best_value_idx] = "best_value"
    if max_cooling_idx not in named:
        named[max_cooling_idx] = "max_cooling"

    for i, (x_row, f_row) in enumerate(zip(result.X, F)):
        # Decode decision vector into allocation dict.
        allocation: dict[str, dict[str, float]] = {}
        for var_idx, (ci, ii) in enumerate(problem.var_meta):
            cell = problem.candidates[ci]
            area = float(x_row[var_idx])
            if area < 0.5:  # treat sub-0.5 sqm as zero
                continue
            cell_key = str(cell.cell_id)
            iv_name = cell.interventions[ii]
            allocation.setdefault(cell_key, {})[iv_name] = round(area, 1)

        sol = ParetoSolution(
            run_id=run.run_id,
            allocation=allocation,
            total_cost=float(f_row[0]),
            total_cooling=float(-f_row[1]),
            equity_score=float(-f_row[2]) if len(f_row) > 2 else None,
            is_named_solution=named.get(i),
        )
        db.add(sol)
        solutions.append(sol)

    db.flush()
    return solutions


def run_pareto_optimization(db: Session, request: ParetoRequest) -> ParetoRunOut:
    """Build candidate set, run NSGA-II, persist Pareto front, return results.

    Falls back gracefully (empty solutions list) if:
      - No trained model is available yet.
      - No feasibility cap rows exist for the requested cells.
      - pymoo is not installed.
    """
    # Add ml/ to sys.path so nsga2_optimizer is importable.
    ml_src = str(Path(__file__).resolve().parents[3] / "ml")
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    use_equity = "equity" in request.objectives
    default_interventions = ["cool_roof_limewash", "tree_planting", "cool_pavement"]

    run = ParetoRun(
        cell_ids=request.cell_ids,
        season=request.season.value,
        objectives=request.objectives,
        population_size=request.population_size,
        generations=request.generations,
        budget_cap=request.budget_cap,
    )
    db.add(run)
    db.flush()
    run_id = run.run_id if run.run_id is not None else 1

    try:
        from src.optimization.nsga2_optimizer import optimize  # noqa: PLC0415

        candidates = _build_candidate_cells(
            db, request.cell_ids, request.season, default_interventions
        )

        if not candidates:
            db.commit()
            return ParetoRunOut(
                run_id=run_id,
                season=request.season,
                objectives=request.objectives,
                solutions=[],
            )

        nsga_result, problem = optimize(
            candidates=candidates,
            budget_cap=request.budget_cap,
            use_equity=use_equity,
            population_size=request.population_size,
            generations=request.generations,
        )

        solutions = _pareto_result_to_solutions(db, run, nsga_result, problem)
        db.commit()

        solution_outs = [
            ParetoSolutionOut(
                solution_id=sol.solution_id if sol.solution_id is not None else (idx + 1),
                total_cost=sol.total_cost,
                total_cooling=sol.total_cooling,
                equity_score=sol.equity_score,
                is_named_solution=sol.is_named_solution,
                allocation=sol.allocation,
            )
            for idx, sol in enumerate(solutions)
        ]

    except Exception as exc:
        # Don't crash the API; return empty solutions and log the error.
        db.rollback()
        db.add(run)
        db.commit()
        print(f"[optimization_service] NSGA-II failed: {exc!r}")
        solution_outs = []

    return ParetoRunOut(
        run_id=run_id,
        season=request.season,
        objectives=request.objectives,
        solutions=solution_outs,
    )
