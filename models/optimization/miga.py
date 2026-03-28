"""Multi-Island Genetic Algorithm (MIGA).

This module implements a continuous multi-island genetic algorithm for
box-constrained, constrained, single-objective, and multi-objective
optimization problems.

Each island evolves an elitist real-coded genetic population locally, and
periodic migration exchanges high-quality individuals across islands to
balance exploration and exploitation.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)


EPS: float = 1e-12
MIN_ISLAND_SIZE: int = 4


def _parse_bounds(
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse box bounds into lower and upper vectors."""
    if isinstance(bounds, Bounds):
        lower = np.asarray(bounds.lb, dtype=float)
        upper = np.asarray(bounds.ub, dtype=float)
    else:
        arr = np.asarray(bounds, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dim, 2)")
        lower = arr[:, 0]
        upper = arr[:, 1]

    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have the same shape")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("all bounds must be finite for multi_island_genetic_optimize")
    if np.any(upper <= lower):
        raise ValueError("each upper bound must be strictly greater than lower bound")
    return lower, upper


def _reflect_bounds(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Reflect out-of-bounds positions back into the feasible box."""
    y = x.copy()
    y = np.where(y < lower, 2.0 * lower - y, y)
    y = np.where(y > upper, 2.0 * upper - y, y)
    return np.clip(y, lower, upper)


def _constraint_violation(
    x: np.ndarray,
    constraints: Sequence[Any],
    args: Tuple[Any, ...],
) -> float:
    """Compute the aggregated violation of linear, nonlinear, or dict constraints."""
    if not constraints:
        return 0.0

    violation = 0.0
    for constraint in constraints:
        if isinstance(constraint, LinearConstraint):
            values = np.atleast_1d(constraint.A @ x)
            lb = np.atleast_1d(constraint.lb).astype(float)
            ub = np.atleast_1d(constraint.ub).astype(float)
            low_v = np.maximum(lb - values, 0.0)
            up_v = np.maximum(values - ub, 0.0)
            low_v[~np.isfinite(lb)] = 0.0
            up_v[~np.isfinite(ub)] = 0.0
            violation += float(np.sum(low_v + up_v))
            continue

        if isinstance(constraint, NonlinearConstraint):
            values = np.atleast_1d(constraint.fun(x))
            lb = np.atleast_1d(constraint.lb).astype(float)
            ub = np.atleast_1d(constraint.ub).astype(float)
            low_v = np.maximum(lb - values, 0.0)
            up_v = np.maximum(values - ub, 0.0)
            low_v[~np.isfinite(lb)] = 0.0
            up_v[~np.isfinite(ub)] = 0.0
            violation += float(np.sum(low_v + up_v))
            continue

        if isinstance(constraint, dict):
            constraint_type = constraint.get("type", "").lower()
            constraint_fun = constraint.get("fun")
            constraint_args = constraint.get("args", args)
            if constraint_fun is None:
                raise ValueError("constraint dict must contain key 'fun'")
            values = np.atleast_1d(constraint_fun(x, *constraint_args))
            if constraint_type == "ineq":
                violation += float(np.sum(np.maximum(-values, 0.0)))
            elif constraint_type == "eq":
                violation += float(np.sum(np.abs(values)))
            else:
                raise ValueError(
                    f"unsupported constraint dict type: '{constraint_type}'"
                )
            continue

        raise TypeError(
            "each constraint must be a LinearConstraint, NonlinearConstraint, "
            "or an SLSQP-style dict with keys 'type' and 'fun'"
        )

    return violation


def _nondominated_indices(objective_matrix: np.ndarray) -> np.ndarray:
    """Return the indices of Pareto non-dominated rows for a minimization problem."""
    num_points = objective_matrix.shape[0]
    is_dominated = np.zeros(num_points, dtype=bool)
    for index in range(num_points):
        if is_dominated[index]:
            continue
        better_or_equal = np.all(objective_matrix <= objective_matrix[index], axis=1)
        strictly_better = np.any(objective_matrix < objective_matrix[index], axis=1)
        dominates = better_or_equal & strictly_better
        dominates[index] = False
        if np.any(dominates):
            is_dominated[index] = True
    return np.where(~is_dominated)[0]


def _normalize_weights(
    num_objectives: int,
    objective_weights: Optional[ArrayLike],
) -> np.ndarray:
    """Normalize multi-objective weights."""
    if objective_weights is None:
        return np.full(num_objectives, 1.0 / num_objectives)

    weights = np.asarray(objective_weights, dtype=float).reshape(-1)
    if weights.size != num_objectives:
        raise ValueError(
            "objective_weights length must match number of objectives"
        )
    if np.any(weights < 0.0):
        raise ValueError("objective_weights must be non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("objective_weights sum must be positive")
    return weights / weight_sum


def _scalarize_objectives(
    objective_values: np.ndarray,
    multi_objective: bool,
    weights: Optional[np.ndarray],
    scalarization: str,
    reference_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert raw objective vectors into scalar fitness values."""
    if not multi_objective:
        if objective_values.shape[1] != 1:
            raise ValueError(
                "objective returned multiple values; set multi_objective=True"
            )
        return objective_values[:, 0]

    if objective_values.shape[1] < 2:
        raise ValueError(
            "multi_objective=True requires objective vector length >= 2"
        )
    if scalarization not in {"weighted_sum", "tchebycheff"}:
        raise ValueError(
            "scalarization must be 'weighted_sum' or 'tchebycheff'"
        )
    if weights is None:
        raise ValueError("weights must be provided for multi-objective scalarization")

    if scalarization == "weighted_sum":
        return objective_values @ weights

    if reference_vectors is None or reference_vectors.size == 0:
        ideal = np.min(objective_values, axis=0)
    else:
        ideal = np.min(reference_vectors, axis=0)
    return np.max(weights[None, :] * np.abs(objective_values - ideal), axis=1)


def _evaluate_population(
    population: np.ndarray,
    func: Callable[..., ArrayLike],
    args: Tuple[Any, ...],
    vectorized: bool,
    multi_objective: bool,
) -> Tuple[np.ndarray, int]:
    """Evaluate the full population and return objective vectors."""
    num_points = population.shape[0]

    if vectorized:
        objective_values = np.asarray(func(population, *args), dtype=float)
        if objective_values.ndim == 1:
            objective_values = objective_values[:, None]
        if objective_values.shape[0] != num_points:
            raise ValueError(
                "vectorized objective must return shape (n_pop,) or (n_pop, n_obj)"
            )
        if not multi_objective and objective_values.shape[1] != 1:
            raise ValueError(
                "objective returned multiple values; set multi_objective=True"
            )
        if multi_objective and objective_values.shape[1] < 2:
            raise ValueError(
                "multi_objective=True requires objective vector length >= 2"
            )
        return objective_values, num_points

    objective_values: List[np.ndarray] = []
    for individual in population:
        raw = np.asarray(func(individual, *args), dtype=float)
        vector = np.atleast_1d(raw).reshape(-1)
        objective_values.append(vector)

    objective_matrix = np.vstack(objective_values)
    if not multi_objective and objective_matrix.shape[1] != 1:
        raise ValueError(
            "objective returned multiple values; set multi_objective=True"
        )
    if multi_objective and objective_matrix.shape[1] < 2:
        raise ValueError(
            "multi_objective=True requires objective vector length >= 2"
        )
    return objective_matrix, num_points


def _build_islands(num_points: int, num_islands: int) -> List[np.ndarray]:
    """Partition a population into contiguous island index blocks."""
    base_size = num_points // num_islands
    remainder = num_points % num_islands
    sizes = np.full(num_islands, base_size, dtype=int)
    sizes[:remainder] += 1

    if np.any(sizes < MIN_ISLAND_SIZE):
        raise ValueError(
            f"population must provide at least {MIN_ISLAND_SIZE} individuals per island"
        )

    islands: List[np.ndarray] = []
    start = 0
    for size in sizes:
        islands.append(np.arange(start, start + int(size)))
        start += int(size)
    return islands


def _tournament_select(
    island_indices: np.ndarray,
    energies: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    """Select one parent index by tournament selection."""
    sample_size = min(tournament_size, island_indices.size)
    candidates = rng.choice(island_indices, size=sample_size, replace=False)
    return int(candidates[np.argmin(energies[candidates])])


def _blend_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
    blend_alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two offspring with BLX-alpha crossover."""
    if rng.random() > crossover_rate:
        return parent_a.copy(), parent_b.copy()

    diff = np.abs(parent_a - parent_b)
    child_lower = np.minimum(parent_a, parent_b) - blend_alpha * diff
    child_upper = np.maximum(parent_a, parent_b) + blend_alpha * diff
    return (
        rng.uniform(child_lower, child_upper),
        rng.uniform(child_lower, child_upper),
    )


def _gaussian_mutation(
    child: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    span: np.ndarray,
    mutation_rate: float,
    mutation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply independent Gaussian mutation to a real-coded chromosome."""
    mutated = child.copy()
    mask = rng.random(mutated.size) < mutation_rate
    if np.any(mask):
        sigma = np.maximum(mutation_scale * span[mask], EPS)
        mutated[mask] += rng.normal(0.0, sigma, size=np.sum(mask))
    return _reflect_bounds(mutated, lower, upper)


def _update_archive(
    archive_points: List[np.ndarray],
    archive_vectors: List[np.ndarray],
    archive_violations: List[float],
    population: np.ndarray,
    objective_vectors: np.ndarray,
    violations: np.ndarray,
) -> None:
    """Append evaluated population data to the Pareto archive."""
    archive_points.extend([row.copy() for row in population])
    archive_vectors.extend([row.copy() for row in objective_vectors])
    archive_violations.extend([float(value) for value in violations])


def multi_island_genetic_optimize(
    func: Callable[..., ArrayLike],
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
    args: Tuple[Any, ...] = (),
    maxiter: int = 200,
    popsize: int = 15,
    tol: float = 1e-6,
    seed: Optional[Union[int, np.random.Generator]] = None,
    callback: Optional[Callable[[np.ndarray, float], bool]] = None,
    disp: bool = False,
    polish: bool = False,
    init: Union[str, np.ndarray] = "random",
    atol: float = 0.0,
    constraints: Union[Sequence[Any], Any] = (),
    x0: Optional[ArrayLike] = None,
    vectorized: bool = False,
    *,
    multi_objective: bool = False,
    objective_weights: Optional[ArrayLike] = None,
    scalarization: str = "weighted_sum",
    return_pareto: bool = False,
    penalty_start: float = 10.0,
    penalty_growth: float = 1.05,
    num_islands: int = 4,
    migration_interval: int = 10,
    migration_size: int = 2,
    crossover_rate: float = 0.9,
    mutation_rate: Optional[float] = None,
    mutation_scale: float = 0.15,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    blend_alpha: float = 0.5,
) -> OptimizeResult:
    """Minimize an objective function with a multi-island genetic algorithm.

    Args:
        func: Objective function ``f(x, *args) -> scalar | array``.
            Input shape is ``(n_dim,)``. Return a scalar for single-objective
            optimization or a 1-D array for multi-objective optimization.
        bounds: Box constraints specified as a ``Bounds`` object or an
            array-like of shape ``(n_dim, 2)``.
        args: Extra positional arguments forwarded to ``func``.
        maxiter: Maximum number of evolutionary generations.
        popsize: Population size multiplier. The actual population size is
            ``max(num_islands * 4, popsize * n_dim)``.
        tol: Relative convergence tolerance applied to the standard deviation
            of penalized energies.
        seed: Integer seed or ``np.random.Generator`` for reproducibility.
        callback: Optional callable ``callback(xk, convergence) -> bool``.
            Returning ``True`` stops the run early.
        disp: If ``True``, print per-generation progress.
        polish: If ``True``, perform local refinement from the best point found.
        init: Either ``"random"`` or a custom initial population array with
            shape ``(n_pop, n_dim)``.
        atol: Absolute convergence tolerance term.
        constraints: Optional sequence of linear, nonlinear, or dict-style
            constraints. Violations are penalized with an exterior penalty.
        x0: Optional initial point inserted at ``population[0]``.
        vectorized: If ``True``, ``func`` receives the full population with
            shape ``(n_pop, n_dim)`` and must return ``(n_pop,)`` or
            ``(n_pop, n_obj)``.
        multi_objective: Whether ``func`` returns multiple objectives.
        objective_weights: Optional non-negative weights used for scalarization.
        scalarization: Multi-objective scalarization method. Supported values
            are ``"weighted_sum"`` and ``"tchebycheff"``.
        return_pareto: If ``True`` and ``multi_objective=True``, attach a
            Pareto archive to the result object.
        penalty_start: Initial constraint penalty multiplier.
        penalty_growth: Multiplicative penalty growth factor per generation.
        num_islands: Number of islands in the population topology.
        migration_interval: Number of generations between migration steps.
        migration_size: Number of elites migrated from each island.
        crossover_rate: Probability of applying crossover to a parent pair.
        mutation_rate: Per-gene mutation probability. Defaults to ``1 / n_dim``.
        mutation_scale: Relative Gaussian mutation scale with respect to the
            bound span.
        elite_fraction: Fraction of the best individuals preserved per island.
        tournament_size: Number of candidates sampled in tournament selection.
        blend_alpha: BLX-alpha crossover expansion coefficient.

    Returns:
        ``OptimizeResult`` containing the best solution, scalar objective,
        population diagnostics, and optional Pareto archive.

    Raises:
        ValueError: If any configuration argument is invalid.
        TypeError: If an unsupported constraint type is provided.

    Complexity:
        Time: ``O(maxiter * n_pop * n_dim)`` plus objective evaluations.
        Space: ``O(n_pop * n_dim)``.
    """
    lower, upper = _parse_bounds(bounds)
    n_dim = lower.size

    if maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {maxiter}")
    if popsize < 1:
        raise ValueError(f"popsize must be >= 1, got {popsize}")
    if num_islands < 1:
        raise ValueError(f"num_islands must be >= 1, got {num_islands}")
    if migration_interval < 1:
        raise ValueError(
            f"migration_interval must be >= 1, got {migration_interval}"
        )
    if migration_size < 1:
        raise ValueError(f"migration_size must be >= 1, got {migration_size}")
    if not (0.0 < crossover_rate <= 1.0):
        raise ValueError(
            f"crossover_rate must be in (0, 1], got {crossover_rate}"
        )
    if mutation_scale <= 0.0:
        raise ValueError(
            f"mutation_scale must be > 0, got {mutation_scale}"
        )
    if not (0.0 < elite_fraction < 1.0):
        raise ValueError(
            f"elite_fraction must be in (0, 1), got {elite_fraction}"
        )
    if tournament_size < 2:
        raise ValueError(
            f"tournament_size must be >= 2, got {tournament_size}"
        )
    if penalty_start <= 0.0:
        raise ValueError(
            f"penalty_start must be > 0, got {penalty_start}"
        )
    if penalty_growth < 1.0:
        raise ValueError(
            f"penalty_growth must be >= 1, got {penalty_growth}"
        )
    if blend_alpha < 0.0:
        raise ValueError(f"blend_alpha must be >= 0, got {blend_alpha}")

    rng: np.random.Generator = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )
    span = upper - lower

    constraints_seq: Sequence[Any]
    if constraints is None:
        constraints_seq = ()
    elif isinstance(constraints, (list, tuple)):
        constraints_seq = constraints
    else:
        constraints_seq = (constraints,)

    if isinstance(init, str):
        if init.lower() != "random":
            raise ValueError(
                f"init must be 'random' or a custom array, got '{init}'"
            )
        population_size = max(num_islands * MIN_ISLAND_SIZE, int(popsize) * n_dim)
        population = rng.uniform(lower, upper, size=(population_size, n_dim))
    else:
        population = np.asarray(init, dtype=float)
        if population.ndim != 2 or population.shape[1] != n_dim:
            raise ValueError(
                f"custom init must have shape (n_pop, {n_dim}), got {population.shape}"
            )
        population_size = population.shape[0]

    islands = _build_islands(population_size, num_islands)

    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x0_arr.size != n_dim:
            raise ValueError(f"x0 must have {n_dim} elements, got {x0_arr.size}")
        population[0] = np.clip(x0_arr, lower, upper)

    if mutation_rate is None:
        mutation_rate = 1.0 / max(n_dim, 1)
    if not (0.0 < mutation_rate <= 1.0):
        raise ValueError(
            f"mutation_rate must be in (0, 1], got {mutation_rate}"
        )

    objective_vectors_arr, nfev = _evaluate_population(
        population=population,
        func=func,
        args=args,
        vectorized=vectorized,
        multi_objective=multi_objective,
    )
    weights = (
        _normalize_weights(objective_vectors_arr.shape[1], objective_weights)
        if multi_objective
        else None
    )
    objective_scalars = _scalarize_objectives(
        objective_values=objective_vectors_arr,
        multi_objective=multi_objective,
        weights=weights,
        scalarization=scalarization,
        reference_vectors=objective_vectors_arr,
    )
    penalties = np.array(
        [
            _constraint_violation(population[index], constraints_seq, args)
            for index in range(population_size)
        ],
        dtype=float,
    )

    archive_points: List[np.ndarray] = []
    archive_vectors: List[np.ndarray] = []
    archive_violations: List[float] = []
    if return_pareto and multi_objective:
        _update_archive(
            archive_points=archive_points,
            archive_vectors=archive_vectors,
            archive_violations=archive_violations,
            population=population,
            objective_vectors=objective_vectors_arr,
            violations=penalties,
        )

    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * penalties

    best_idx = int(np.argmin(energies))
    best_x = population[best_idx].copy()
    best_energy = float(energies[best_idx])
    best_obj_vec = objective_vectors_arr[best_idx].copy()
    best_scalar = float(objective_scalars[best_idx])

    success = False
    message = "Maximum number of generations reached."

    for iteration in range(1, maxiter + 1):
        progress = (iteration - 1) / max(maxiter - 1, 1)
        current_mutation_scale = mutation_scale * (1.0 - 0.75 * progress)
        offspring = np.empty_like(population)

        for island_indices in islands:
            island_size = island_indices.size
            island_order = island_indices[np.argsort(energies[island_indices])]
            elite_count = max(1, int(np.ceil(elite_fraction * island_size)))
            elite_count = min(elite_count, island_size - 2)
            elites = population[island_order[:elite_count]].copy()

            children: List[np.ndarray] = []
            num_children = island_size - elite_count
            while len(children) < num_children:
                parent_a_idx = _tournament_select(
                    island_indices=island_indices,
                    energies=energies,
                    tournament_size=tournament_size,
                    rng=rng,
                )
                parent_b_idx = _tournament_select(
                    island_indices=island_indices,
                    energies=energies,
                    tournament_size=tournament_size,
                    rng=rng,
                )
                parent_a = population[parent_a_idx]
                parent_b = population[parent_b_idx]
                child_a, child_b = _blend_crossover(
                    parent_a=parent_a,
                    parent_b=parent_b,
                    crossover_rate=crossover_rate,
                    blend_alpha=blend_alpha,
                    rng=rng,
                )
                child_a = _gaussian_mutation(
                    child=child_a,
                    lower=lower,
                    upper=upper,
                    span=span,
                    mutation_rate=mutation_rate,
                    mutation_scale=current_mutation_scale,
                    rng=rng,
                )
                child_b = _gaussian_mutation(
                    child=child_b,
                    lower=lower,
                    upper=upper,
                    span=span,
                    mutation_rate=mutation_rate,
                    mutation_scale=current_mutation_scale,
                    rng=rng,
                )
                children.append(child_a)
                if len(children) < num_children:
                    children.append(child_b)

            island_population = np.vstack([elites, np.vstack(children[:num_children])])
            if island_population.shape[0] != island_size:
                raise RuntimeError("island reconstruction produced an invalid size")
            offspring[island_indices] = island_population

        offspring_vectors, eval_count = _evaluate_population(
            population=offspring,
            func=func,
            args=args,
            vectorized=vectorized,
            multi_objective=multi_objective,
        )
        nfev += eval_count

        reference_vectors = (
            np.vstack([objective_vectors_arr, offspring_vectors])
            if multi_objective
            else offspring_vectors
        )
        offspring_scalars = _scalarize_objectives(
            objective_values=offspring_vectors,
            multi_objective=multi_objective,
            weights=weights,
            scalarization=scalarization,
            reference_vectors=reference_vectors,
        )
        offspring_penalties = np.array(
            [
                _constraint_violation(offspring[index], constraints_seq, args)
                for index in range(population_size)
            ],
            dtype=float,
        )

        penalty_factor *= penalty_growth
        offspring_energies = offspring_scalars + penalty_factor * offspring_penalties

        population = offspring
        objective_vectors_arr = offspring_vectors
        objective_scalars = offspring_scalars
        penalties = offspring_penalties
        energies = offspring_energies

        if return_pareto and multi_objective:
            _update_archive(
                archive_points=archive_points,
                archive_vectors=archive_vectors,
                archive_violations=archive_violations,
                population=population,
                objective_vectors=objective_vectors_arr,
                violations=penalties,
            )

        if num_islands > 1 and iteration % migration_interval == 0:
            migrants = []
            migrant_vectors = []
            migrant_scalars = []
            migrant_penalties = []
            migrant_energies = []

            for island_indices in islands:
                island_order = island_indices[np.argsort(energies[island_indices])]
                num_migrants = min(migration_size, island_indices.size)
                selected = island_order[:num_migrants]
                migrants.append(population[selected].copy())
                migrant_vectors.append(objective_vectors_arr[selected].copy())
                migrant_scalars.append(objective_scalars[selected].copy())
                migrant_penalties.append(penalties[selected].copy())
                migrant_energies.append(energies[selected].copy())

            for island_id, island_indices in enumerate(islands):
                source_id = (island_id - 1) % num_islands
                incoming_x = migrants[source_id]
                incoming_vectors = migrant_vectors[source_id]
                incoming_scalars = migrant_scalars[source_id]
                incoming_penalties = migrant_penalties[source_id]
                incoming_energies = migrant_energies[source_id]

                replace_count = min(incoming_x.shape[0], island_indices.size)
                target_order = island_indices[np.argsort(energies[island_indices])[::-1]]
                target_indices = target_order[:replace_count]
                population[target_indices] = incoming_x[:replace_count]
                objective_vectors_arr[target_indices] = incoming_vectors[:replace_count]
                objective_scalars[target_indices] = incoming_scalars[:replace_count]
                penalties[target_indices] = incoming_penalties[:replace_count]
                energies[target_indices] = incoming_energies[:replace_count]

        current_best_idx = int(np.argmin(energies))
        if energies[current_best_idx] < best_energy:
            best_idx = current_best_idx
            best_x = population[current_best_idx].copy()
            best_energy = float(energies[current_best_idx])
            best_obj_vec = objective_vectors_arr[current_best_idx].copy()
            best_scalar = float(objective_scalars[current_best_idx])

        convergence = float(np.std(energies))
        threshold = float(atol + tol * abs(np.mean(energies)))

        if disp:
            print(
                f"[MIGA] iter={iteration:4d}  best={best_energy:.8e}  "
                f"mean={np.mean(energies):.8e}  std={convergence:.8e}"
            )

        if callback is not None and callback(best_x.copy(), convergence):
            message = "Stopped by callback."
            break

        if convergence <= threshold:
            success = True
            message = "Optimization converged."
            break

    if polish:
        def _local_objective(x_local: np.ndarray) -> float:
            raw = np.atleast_1d(np.asarray(func(x_local, *args), dtype=float))
            if raw.size == 1:
                objective_value = float(raw[0])
            else:
                if weights is None:
                    local_weights = _normalize_weights(raw.size, objective_weights)
                else:
                    local_weights = weights
                if scalarization == "weighted_sum":
                    objective_value = float(np.dot(local_weights, raw))
                else:
                    reference = np.min(objective_vectors_arr, axis=0)
                    objective_value = float(
                        np.max(local_weights * np.abs(raw - reference))
                    )
            violation = _constraint_violation(x_local, constraints_seq, args)
            return objective_value + penalty_factor * violation

        local_method = "L-BFGS-B" if not constraints_seq else "SLSQP"
        local_res = minimize(
            _local_objective,
            x0=best_x,
            method=local_method,
            bounds=Bounds(lower, upper),
            constraints=constraints_seq if constraints_seq else (),
        )
        nfev += int(getattr(local_res, "nfev", 0))
        if local_res.fun < best_energy:
            best_x = np.asarray(local_res.x, dtype=float)
            best_obj_vec = np.atleast_1d(np.asarray(func(best_x, *args), dtype=float))
            nfev += 1
            best_energy = float(local_res.fun)
            if best_obj_vec.size == 1:
                best_scalar = float(best_obj_vec[0])
            else:
                local_weights = weights
                if local_weights is None:
                    local_weights = _normalize_weights(
                        best_obj_vec.size, objective_weights
                    )
                if scalarization == "weighted_sum":
                    best_scalar = float(np.dot(local_weights, best_obj_vec))
                else:
                    reference = np.min(objective_vectors_arr, axis=0)
                    best_scalar = float(
                        np.max(local_weights * np.abs(best_obj_vec - reference))
                    )

    result = OptimizeResult()
    result.x = best_x
    result.fun = best_scalar
    result.success = bool(success)
    result.message = message
    result.nit = int(iteration if "iteration" in locals() else 0)
    result.nfev = int(nfev)
    result.population = population.copy()
    result.population_energies = energies.copy()
    result.constraint_violation = float(
        _constraint_violation(best_x, constraints_seq, args)
    )
    result.objective_vector = best_obj_vec.copy()
    result.penalized_fun = float(best_energy)
    result.optimizer = "MIGA"

    if multi_objective:
        result.fun_vector = best_obj_vec.copy()
        if return_pareto and archive_vectors:
            archive_f = np.asarray(archive_vectors, dtype=float)
            archive_x = np.asarray(archive_points, dtype=float)
            archive_v = np.asarray(archive_violations, dtype=float)
            feasible_mask = archive_v <= 1.0e-8
            if np.any(feasible_mask):
                candidate_f = archive_f[feasible_mask]
                candidate_x = archive_x[feasible_mask]
                candidate_v = archive_v[feasible_mask]
            else:
                min_violation = float(np.min(archive_v))
                candidate_mask = archive_v <= min_violation + 1.0e-8
                candidate_f = archive_f[candidate_mask]
                candidate_x = archive_x[candidate_mask]
                candidate_v = archive_v[candidate_mask]
            nd_idx = _nondominated_indices(candidate_f)
            result.pareto_f = candidate_f[nd_idx]
            result.pareto_x = candidate_x[nd_idx]
            result.pareto_constraint_violation = candidate_v[nd_idx]

    return result
