# CFSSDA Dragonfly Optimizer
# Code author: Shengning Wang

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, minimize
from scipy.special import gamma
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


def _parse_bounds(bounds: Union[Bounds, Sequence[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert box bounds to lower and upper vectors.
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


def _levy_flight(num_vars: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one Levy-flight step.
    """
    if not (0.0 < beta <= 2.0):
        raise ValueError("levy_beta must be in (0, 2].")

    sigma_u = (
        gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
        / (gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)

    u = rng.normal(0.0, sigma_u, size=num_vars)
    v = rng.normal(0.0, 1.0, size=num_vars)
    return 0.01 * u / (np.abs(v) ** (1.0 / beta) + 1e-12)


def dragonfly_optimize(
    func: Callable,
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
    args: Tuple[Any, ...] = (),
    maxiter: int = 200,
    popsize: int = 30,
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
    c_max: float = 2.0,
    c_min: float = 0.2,
    inertia_start: float = 0.9,
    inertia_end: float = 0.2,
    neighbor_radius_start: Optional[float] = None,
    neighbor_radius_end: float = 0.0,
    coulomb_alpha_mean: float = 2.0,
    coulomb_alpha_std: float = 0.25,
    k0: float = 1.0,
    levy_beta: float = 1.5,
) -> OptimizeResult:
    """
    Coulomb-force-search dragonfly optimizer.

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
        c_max (float): Initial behavior weight.
        c_min (float): Final behavior weight.
        inertia_start (float): Initial inertia weight.
        inertia_end (float): Final inertia weight.
        neighbor_radius_start (Optional[float]): Initial neighborhood radius.
        neighbor_radius_end (float): Final neighborhood radius.
        coulomb_alpha_mean (float): Mean decay factor for Coulomb search.
        coulomb_alpha_std (float): Std decay factor for Coulomb search.
        k0 (float): Initial Coulomb coefficient.
        levy_beta (float): Levy-flight beta.

    Returns:
        OptimizeResult: Optimization result.
    """
    lower, upper = _parse_bounds(bounds)
    num_vars = lower.size
    num_pop = max(20, int(popsize) * num_vars)
    span = upper - lower
    constraints = _normalize_constraints(constraints)

    if maxiter < 1:
        raise ValueError("maxiter must be >= 1.")
    if num_pop < 2:
        raise ValueError("Population size must be >= 2.")
    if penalty_start <= 0.0:
        raise ValueError("penalty_start must be > 0.")
    if penalty_growth < 1.0:
        raise ValueError("penalty_growth must be >= 1.")
    if scalarization not in ["weighted_sum", "tchebycheff"]:
        raise ValueError("Unsupported scalarization method.")

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    if neighbor_radius_start is None:
        neighbor_radius_start = 0.25 * float(np.linalg.norm(span))
    neighbor_radius_start = max(neighbor_radius_start, 1e-12)
    neighbor_radius_end = max(neighbor_radius_end, 0.0)

    population = rng.uniform(lower, upper, size=(num_pop, num_vars))
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.size != num_vars:
            raise ValueError("x0 dimension does not match bounds.")
        population[0] = np.clip(x0, lower, upper)

    delta_x = rng.uniform(-0.1, 0.1, size=(num_pop, num_vars)) * span

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

    best_idx = int(np.argmin(energies))
    best_x = population[best_idx].copy()
    best_f = objective_vectors[best_idx].copy()
    best_fun = float(objective_scalars[best_idx])
    best_energy = float(energies[best_idx])

    message = "Maximum number of iterations reached."
    success = False

    for iteration in range(maxiter):
        ratio = iteration / max(maxiter - 1, 1)
        inertia = inertia_start + (inertia_end - inertia_start) * ratio
        behavior = c_max + (c_min - c_max) * ratio
        neighbor_radius = neighbor_radius_start + (neighbor_radius_end - neighbor_radius_start) * ratio

        curr_best = float(np.min(energies))
        curr_worst = float(np.max(energies))
        mass_raw = (energies - curr_worst) / (curr_best - curr_worst + 1e-12)
        mass_raw = np.maximum(mass_raw, 1e-12)
        mass = mass_raw / (np.sum(mass_raw) + 1e-12)
        gamma_w = c_min + (c_max - c_min) * mass_raw
        fit_g = gamma_w * mass

        order = np.argsort(mass)[::-1]
        kbest_count = max(1, int(np.ceil(num_pop - (num_pop - 1) * ratio)))
        kbest = order[:kbest_count]

        food_idx = int(np.argmin(energies))
        enemy_idx = int(np.argmax(energies))
        food_pos = population[food_idx]
        enemy_pos = population[enemy_idx]

        alpha_hat = abs(rng.normal(coulomb_alpha_mean, coulomb_alpha_std))
        k_t = k0 * np.exp(-alpha_hat * (iteration + 1) / maxiter)
        max_step = 0.2 * span

        new_population = population.copy()
        new_delta_x = delta_x.copy()

        for i in range(num_pop):
            distances = np.linalg.norm(population - population[i], axis=1)
            neighbors = np.where((distances > 0.0) & (distances <= neighbor_radius))[0]

            if neighbors.size == 0:
                levy_step = _levy_flight(num_vars, levy_beta, rng) * np.maximum(np.abs(population[i]), 1.0)
                new_delta_x[i] = levy_step
                new_population[i] = population[i] + levy_step
                continue

            separation = -np.sum(population[i] - population[neighbors], axis=0)
            alignment = np.mean(delta_x[neighbors], axis=0)
            cohesion = np.mean(population[neighbors], axis=0) - population[i]
            food_attraction = food_pos - population[i]
            enemy_avoidance = population[i] - enemy_pos

            force = np.zeros(num_vars, dtype=np.float64)
            for j in kbest:
                if j == i:
                    continue
                diff = population[j] - population[i]
                dist = np.linalg.norm(diff) + 1e-12
                force += rng.random() * k_t * mass[i] * mass[j] * diff / dist

            enemy_diff = enemy_pos - population[i]
            enemy_dist = np.linalg.norm(enemy_diff) + 1e-12
            force -= rng.random() * k_t * mass[i] * mass[enemy_idx] * enemy_diff / enemy_dist
            acceleration = force / (fit_g[i] + 1e-12)

            step = (
                inertia * delta_x[i]
                + behavior * rng.random() * separation
                + behavior * rng.random() * alignment
                + behavior * rng.random() * cohesion
                + 2.0 * rng.random() * food_attraction
                + behavior * rng.random() * enemy_avoidance
                + acceleration
            )
            step = np.clip(step, -max_step, max_step)
            new_delta_x[i] = step
            new_population[i] = population[i] + step

        new_population = _repair_to_bounds(new_population, lower, upper)
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

        improved = new_energies < energies
        population[improved] = new_population[improved]
        delta_x[improved] = new_delta_x[improved]
        objective_vectors[improved] = new_objective_vectors[improved]
        objective_scalars[improved] = new_objective_scalars[improved]
        violations[improved] = new_violations[improved]
        energies[improved] = new_energies[improved]

        if return_pareto and multi_objective:
            archive_x.extend([row.copy() for row in population])
            archive_f.extend([row.copy() for row in objective_vectors])
            archive_v.extend([float(value) for value in violations])

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
                    reference = np.min(objective_vectors, axis=0)
                    value = float(np.max(local_weights * np.abs(values - reference)))
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
    result.optimizer = "CFSSDA"

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
