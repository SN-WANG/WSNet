# Multi-Island Genetic Algorithm (MIGA) Optimizer
# Code author: Shengning Wang

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, minimize
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


def _parse_bounds(bounds: Union[Bounds, Sequence[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert box bounds to lower and upper vectors.

    Args:
        bounds (Union[Bounds, Sequence[Tuple[float, float]]]): Variable bounds.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            lower (np.ndarray): Lower bounds, shape (input_dim,), dtype: float64.
            upper (np.ndarray): Upper bounds, shape (input_dim,), dtype: float64.
    """
    if isinstance(bounds, Bounds):
        lower = np.asarray(bounds.lb, dtype=np.float64).reshape(-1)
        upper = np.asarray(bounds.ub, dtype=np.float64).reshape(-1)
    else:
        bounds_arr = np.asarray(bounds, dtype=np.float64)
        if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
            raise ValueError("Bounds must have shape (num_vars, 2).")
        lower = bounds_arr[:, 0]
        upper = bounds_arr[:, 1]

    if lower.shape != upper.shape:
        raise ValueError("Lower and upper bounds must have the same shape.")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Bounds must be finite.")
    if np.any(upper <= lower):
        raise ValueError("Each upper bound must be greater than the lower bound.")

    return lower, upper


def _normalize_constraints(constraints: Union[Sequence[Any], Any]) -> Tuple[Any, ...]:
    """
    Normalize constraints to a tuple.
    """
    if constraints is None:
        return ()
    if isinstance(constraints, (list, tuple)):
        return tuple(constraints)
    return (constraints,)


def _repair_to_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Reflect points back into the search box.
    """
    x = np.where(x < lower, 2.0 * lower - x, x)
    x = np.where(x > upper, 2.0 * upper - x, x)
    return np.clip(x, lower, upper)


def _constraint_violation(x: np.ndarray, constraints: Sequence[Any], args: Tuple[Any, ...]) -> float:
    """
    Calculate the total constraint violation at one point.

    Args:
        x (np.ndarray): Candidate point, shape (input_dim,), dtype: float64.
        constraints (Sequence[Any]): Constraint objects.
        args (Tuple[Any, ...]): Extra objective arguments.

    Returns:
        float: Aggregated non-negative violation value.
    """
    if not constraints:
        return 0.0

    violation = 0.0
    for con in constraints:
        if isinstance(con, LinearConstraint):
            values = np.atleast_1d(con.A @ x)
            lower = np.atleast_1d(con.lb).astype(np.float64)
            upper = np.atleast_1d(con.ub).astype(np.float64)
            low_violation = np.maximum(lower - values, 0.0)
            up_violation = np.maximum(values - upper, 0.0)
            low_violation[~np.isfinite(lower)] = 0.0
            up_violation[~np.isfinite(upper)] = 0.0
            violation += float(np.sum(low_violation + up_violation))
            continue

        if isinstance(con, NonlinearConstraint):
            values = np.atleast_1d(con.fun(x))
            lower = np.atleast_1d(con.lb).astype(np.float64)
            upper = np.atleast_1d(con.ub).astype(np.float64)
            low_violation = np.maximum(lower - values, 0.0)
            up_violation = np.maximum(values - upper, 0.0)
            low_violation[~np.isfinite(lower)] = 0.0
            up_violation[~np.isfinite(upper)] = 0.0
            violation += float(np.sum(low_violation + up_violation))
            continue

        if isinstance(con, dict):
            con_type = con.get("type", "").lower()
            con_fun = con.get("fun")
            con_args = con.get("args", args)
            if con_fun is None:
                raise ValueError("Constraint dict must contain key 'fun'.")

            values = np.atleast_1d(con_fun(x, *con_args))
            if con_type == "ineq":
                violation += float(np.sum(np.maximum(-values, 0.0)))
            elif con_type == "eq":
                violation += float(np.sum(np.abs(values)))
            else:
                raise ValueError(f"Unsupported constraint type: {con_type}.")
            continue

        raise TypeError("Unsupported constraint format.")

    return violation


def _normalize_weights(num_objectives: int, objective_weights: Optional[np.ndarray]) -> np.ndarray:
    """
    Normalize multi-objective weights.
    """
    if objective_weights is None:
        return np.ones(num_objectives, dtype=np.float64) / num_objectives

    weights = np.asarray(objective_weights, dtype=np.float64).reshape(-1)
    if weights.size != num_objectives:
        raise ValueError("Objective weights must match the objective dimension.")
    if np.any(weights < 0.0):
        raise ValueError("Objective weights must be non-negative.")

    weight_sum = np.sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("Objective weights must have a positive sum.")

    return weights / weight_sum


def _nondominated_indices(y: np.ndarray) -> np.ndarray:
    """
    Return indices of the Pareto non-dominated points.
    """
    num_points = y.shape[0]
    dominated = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        if dominated[i]:
            continue
        better_or_equal = np.all(y <= y[i], axis=1)
        strictly_better = np.any(y < y[i], axis=1)
        dominates_i = better_or_equal & strictly_better
        dominates_i[i] = False
        if np.any(dominates_i):
            dominated[i] = True

    return np.where(~dominated)[0]


def _evaluate_population(
    population: np.ndarray,
    func: Callable,
    args: Tuple[Any, ...],
    multi_objective: bool,
    weights: Optional[np.ndarray],
    scalarization: str,
    reference_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the full population and return objective vectors and scalar fitness.
    """
    objective_vectors = []
    for i in range(population.shape[0]):
        values = np.atleast_1d(np.asarray(func(population[i], *args), dtype=np.float64)).reshape(-1)
        objective_vectors.append(values)

    objective_vectors = np.vstack(objective_vectors)

    if not multi_objective:
        if objective_vectors.shape[1] != 1:
            raise ValueError("Objective returned multiple values; set multi_objective=True.")
        return objective_vectors, objective_vectors[:, 0]

    if objective_vectors.shape[1] < 2:
        raise ValueError("Multi-objective mode requires at least two objectives.")

    if weights is None:
        weights = np.ones(objective_vectors.shape[1], dtype=np.float64) / objective_vectors.shape[1]

    if scalarization == "weighted_sum":
        objective_scalars = objective_vectors @ weights
    elif scalarization == "tchebycheff":
        if reference_values is None or reference_values.size == 0:
            ideal = np.min(objective_vectors, axis=0)
        else:
            ideal = np.min(reference_values, axis=0)
        objective_scalars = np.max(weights[None, :] * np.abs(objective_vectors - ideal), axis=1)
    else:
        raise ValueError("Unsupported scalarization method.")

    return objective_vectors, objective_scalars


def _split_islands(num_points: int, num_islands: int) -> List[np.ndarray]:
    """
    Split the population indices into several islands.
    """
    base_size = num_points // num_islands
    remainder = num_points % num_islands
    island_sizes = np.full(num_islands, base_size, dtype=int)
    island_sizes[:remainder] += 1

    if np.any(island_sizes < 4):
        raise ValueError("Each island must contain at least 4 individuals.")

    islands = []
    start = 0
    for size in island_sizes:
        islands.append(np.arange(start, start + size))
        start += size

    return islands


def _select_parent(indices: np.ndarray, energies: np.ndarray, tournament_size: int, rng: np.random.Generator) -> int:
    """
    Tournament selection.
    """
    candidate_size = min(tournament_size, indices.size)
    candidate_indices = rng.choice(indices, size=candidate_size, replace=False)
    return int(candidate_indices[np.argmin(energies[candidate_indices])])


def multi_island_genetic_optimize(
    func: Callable,
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
    args: Tuple[Any, ...] = (),
    maxiter: int = 200,
    popsize: int = 15,
    tol: float = 1e-6,
    seed: Optional[Union[int, np.random.Generator]] = None,
    polish: bool = False,
    constraints: Union[Sequence[Any], Any] = (),
    x0: Optional[np.ndarray] = None,
    multi_objective: bool = False,
    objective_weights: Optional[np.ndarray] = None,
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
    """
    Multi-island genetic algorithm for continuous optimization.

    Args:
        func (Callable): Objective function.
        bounds (Union[Bounds, Sequence[Tuple[float, float]]]): Variable bounds.
        args (Tuple[Any, ...]): Extra objective arguments.
        maxiter (int): Maximum iterations.
        popsize (int): Population size multiplier.
        tol (float): Convergence tolerance.
        seed (Optional[Union[int, np.random.Generator]]): Random seed.
        polish (bool): Whether to run local refinement at the end.
        constraints (Union[Sequence[Any], Any]): Constraints.
        x0 (Optional[np.ndarray]): Initial guess.
        multi_objective (bool): Whether the objective is multi-objective.
        objective_weights (Optional[np.ndarray]): Objective weights.
        scalarization (str): Scalarization method.
        return_pareto (bool): Whether to return Pareto points.
        penalty_start (float): Initial penalty factor.
        penalty_growth (float): Penalty growth factor.
        num_islands (int): Number of islands.
        migration_interval (int): Migration interval.
        migration_size (int): Number of migrants.
        crossover_rate (float): BLX-alpha crossover probability.
        mutation_rate (Optional[float]): Per-gene mutation probability.
        mutation_scale (float): Gaussian mutation scale.
        elite_fraction (float): Elite retention ratio.
        tournament_size (int): Tournament size.
        blend_alpha (float): BLX-alpha expansion coefficient.

    Returns:
        OptimizeResult: Optimization result.
    """
    lower, upper = _parse_bounds(bounds)
    num_vars = lower.size
    span = upper - lower
    constraints = _normalize_constraints(constraints)

    if maxiter < 1:
        raise ValueError("maxiter must be >= 1.")
    if popsize < 1:
        raise ValueError("popsize must be >= 1.")
    if num_islands < 1:
        raise ValueError("num_islands must be >= 1.")
    if migration_interval < 1:
        raise ValueError("migration_interval must be >= 1.")
    if migration_size < 1:
        raise ValueError("migration_size must be >= 1.")
    if not (0.0 < crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be in (0, 1].")
    if mutation_scale <= 0.0:
        raise ValueError("mutation_scale must be > 0.")
    if not (0.0 < elite_fraction < 1.0):
        raise ValueError("elite_fraction must be in (0, 1).")
    if tournament_size < 2:
        raise ValueError("tournament_size must be >= 2.")
    if penalty_start <= 0.0:
        raise ValueError("penalty_start must be > 0.")
    if penalty_growth < 1.0:
        raise ValueError("penalty_growth must be >= 1.")
    if blend_alpha < 0.0:
        raise ValueError("blend_alpha must be >= 0.")

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    num_pop = max(num_islands * 4, int(popsize) * num_vars)
    population = rng.uniform(lower, upper, size=(num_pop, num_vars))
    islands = _split_islands(num_pop, num_islands)

    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.size != num_vars:
            raise ValueError("x0 dimension does not match bounds.")
        population[0] = np.clip(x0, lower, upper)

    if mutation_rate is None:
        mutation_rate = 1.0 / max(num_vars, 1)
    if not (0.0 < mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in (0, 1].")

    weights = None

    objective_vectors, objective_scalars = _evaluate_population(
        population=population,
        func=func,
        args=args,
        multi_objective=multi_objective,
        weights=weights,
        scalarization=scalarization,
    )

    nfev = num_pop

    if multi_objective:
        weights = _normalize_weights(objective_vectors.shape[1], objective_weights)
        if scalarization == "weighted_sum":
            objective_scalars = objective_vectors @ weights
        else:
            ideal = np.min(objective_vectors, axis=0)
            objective_scalars = np.max(weights[None, :] * np.abs(objective_vectors - ideal), axis=1)

    violations = np.array(
        [_constraint_violation(population[i], constraints, args) for i in range(num_pop)],
        dtype=np.float64,
    )
    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * violations

    archive_x: List[np.ndarray] = []
    archive_f: List[np.ndarray] = []
    archive_v: List[float] = []
    if return_pareto and multi_objective:
        archive_x.extend([row.copy() for row in population])
        archive_f.extend([row.copy() for row in objective_vectors])
        archive_v.extend([float(value) for value in violations])

    best_index = int(np.argmin(energies))
    best_x = population[best_index].copy()
    best_f = objective_vectors[best_index].copy()
    best_fun = float(objective_scalars[best_index])
    best_energy = float(energies[best_index])

    message = "Maximum number of iterations reached."
    success = False

    for iteration in range(maxiter):
        ratio = iteration / max(maxiter - 1, 1)
        curr_mutation_scale = mutation_scale * (1.0 - 0.75 * ratio)
        new_population = np.empty_like(population)

        for island in islands:
            island_size = island.size
            island_order = island[np.argsort(energies[island])]
            elite_count = max(1, int(np.ceil(elite_fraction * island_size)))
            elite_count = min(elite_count, island_size - 2)

            elites = population[island_order[:elite_count]].copy()
            children = []

            while len(children) < island_size - elite_count:
                idx_a = _select_parent(island, energies, tournament_size, rng)
                idx_b = _select_parent(island, energies, tournament_size, rng)
                parent_a = population[idx_a]
                parent_b = population[idx_b]

                if rng.random() < crossover_rate:
                    diff = np.abs(parent_a - parent_b)
                    child_lower = np.minimum(parent_a, parent_b) - blend_alpha * diff
                    child_upper = np.maximum(parent_a, parent_b) + blend_alpha * diff
                    child_a = rng.uniform(child_lower, child_upper)
                    child_b = rng.uniform(child_lower, child_upper)
                else:
                    child_a = parent_a.copy()
                    child_b = parent_b.copy()

                for child in [child_a, child_b]:
                    mask = rng.random(num_vars) < mutation_rate
                    if np.any(mask):
                        sigma = np.maximum(curr_mutation_scale * span[mask], 1e-12)
                        child[mask] += rng.normal(0.0, sigma, size=np.sum(mask))
                    child = _repair_to_bounds(child, lower, upper)
                    children.append(child)
                    if len(children) >= island_size - elite_count:
                        break

            new_population[island] = np.vstack([elites, np.asarray(children[:island_size - elite_count])])

        reference_values = objective_vectors if multi_objective else None
        new_objective_vectors, new_objective_scalars = _evaluate_population(
            population=new_population,
            func=func,
            args=args,
            multi_objective=multi_objective,
            weights=weights,
            scalarization=scalarization,
            reference_values=reference_values,
        )
        nfev += num_pop
        new_violations = np.array(
            [_constraint_violation(new_population[i], constraints, args) for i in range(num_pop)],
            dtype=np.float64,
        )

        penalty_factor *= penalty_growth
        new_energies = new_objective_scalars + penalty_factor * new_violations

        population = new_population
        objective_vectors = new_objective_vectors
        objective_scalars = new_objective_scalars
        violations = new_violations
        energies = new_energies

        if return_pareto and multi_objective:
            archive_x.extend([row.copy() for row in population])
            archive_f.extend([row.copy() for row in objective_vectors])
            archive_v.extend([float(value) for value in violations])

        if num_islands > 1 and (iteration + 1) % migration_interval == 0:
            migrants_x = []
            migrants_f = []
            migrants_s = []
            migrants_v = []
            migrants_e = []

            for island in islands:
                island_order = island[np.argsort(energies[island])]
                count = min(migration_size, island.size)
                selected = island_order[:count]
                migrants_x.append(population[selected].copy())
                migrants_f.append(objective_vectors[selected].copy())
                migrants_s.append(objective_scalars[selected].copy())
                migrants_v.append(violations[selected].copy())
                migrants_e.append(energies[selected].copy())

            for island_id, island in enumerate(islands):
                source_id = (island_id - 1) % num_islands
                incoming_x = migrants_x[source_id]
                incoming_f = migrants_f[source_id]
                incoming_s = migrants_s[source_id]
                incoming_v = migrants_v[source_id]
                incoming_e = migrants_e[source_id]

                replace_count = min(incoming_x.shape[0], island.size)
                island_order = island[np.argsort(energies[island])[::-1]]
                replace_idx = island_order[:replace_count]

                population[replace_idx] = incoming_x[:replace_count]
                objective_vectors[replace_idx] = incoming_f[:replace_count]
                objective_scalars[replace_idx] = incoming_s[:replace_count]
                violations[replace_idx] = incoming_v[:replace_count]
                energies[replace_idx] = incoming_e[:replace_count]

        curr_best_idx = int(np.argmin(energies))
        if energies[curr_best_idx] < best_energy:
            best_x = population[curr_best_idx].copy()
            best_f = objective_vectors[curr_best_idx].copy()
            best_fun = float(objective_scalars[curr_best_idx])
            best_energy = float(energies[curr_best_idx])

        if np.std(energies) <= tol * max(np.abs(np.mean(energies)), 1.0):
            success = True
            message = "Optimization converged."
            break

    if polish:
        def local_objective(x_local: np.ndarray) -> float:
            values = np.atleast_1d(np.asarray(func(x_local, *args), dtype=np.float64)).reshape(-1)
            if values.size == 1:
                value = float(values[0])
            else:
                local_weights = _normalize_weights(values.size, objective_weights)
                if scalarization == "weighted_sum":
                    value = float(np.dot(local_weights, values))
                else:
                    value = float(np.max(local_weights * np.abs(values - best_f)))
            return value + penalty_factor * _constraint_violation(x_local, constraints, args)

        polish_method = "L-BFGS-B" if not constraints else "SLSQP"
        polish_result = minimize(
            local_objective,
            x0=best_x,
            method=polish_method,
            bounds=Bounds(lower, upper),
            constraints=constraints if constraints else (),
        )
        nfev += int(getattr(polish_result, "nfev", 0))
        if polish_result.fun < best_energy:
            best_x = np.asarray(polish_result.x, dtype=np.float64)
            best_f = np.atleast_1d(np.asarray(func(best_x, *args), dtype=np.float64)).reshape(-1)
            nfev += 1
            if best_f.size == 1:
                best_fun = float(best_f[0])
            else:
                local_weights = _normalize_weights(best_f.size, objective_weights)
                if scalarization == "weighted_sum":
                    best_fun = float(np.dot(local_weights, best_f))
                else:
                    reference = np.min(objective_vectors, axis=0)
                    best_fun = float(np.max(local_weights * np.abs(best_f - reference)))
            best_energy = float(polish_result.fun)

    result = OptimizeResult()
    result.x = best_x
    result.fun = best_fun
    result.success = success
    result.message = message
    result.nit = iteration + 1 if "iteration" in locals() else 0
    result.nfev = nfev
    result.population = population.copy()
    result.population_energies = energies.copy()
    result.objective_vector = best_f.copy()
    result.constraint_violation = float(_constraint_violation(best_x, constraints, args))
    result.penalized_fun = best_energy
    result.optimizer = "MIGA"

    if multi_objective:
        result.fun_vector = best_f.copy()
        if return_pareto and len(archive_f) > 0:
            archive_y = np.asarray(archive_f, dtype=np.float64)
            archive_x = np.asarray(archive_x, dtype=np.float64)
            archive_v = np.asarray(archive_v, dtype=np.float64)

            feasible_mask = archive_v <= 1e-8
            if np.any(feasible_mask):
                archive_y = archive_y[feasible_mask]
                archive_x = archive_x[feasible_mask]
            else:
                best_violation = np.min(archive_v)
                near_mask = archive_v <= best_violation + 1e-8
                archive_y = archive_y[near_mask]
                archive_x = archive_x[near_mask]

            nd_idx = _nondominated_indices(archive_y)
            result.pareto_f = archive_y[nd_idx]
            result.pareto_x = archive_x[nd_idx]

    return result
