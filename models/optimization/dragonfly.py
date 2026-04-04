# CFSSDA Dragonfly Optimizer
# Author: Shengning Wang

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize
from scipy.special import gamma

from models.optimization._shared import (
    _append_archive,
    _apply_initial_guess,
    _constraint_violation,
    _evaluate_constraint_violations,
    _evaluate_population,
    _finalize_pareto_archive,
    _initialize_objective_values,
    _make_rng,
    _normalize_constraints,
    _normalize_weights,
    _parse_bounds,
    _repair_to_bounds,
)


def _levy_flight(num_vars: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one Levy-flight step.

    Args:
        num_vars (int): Number of variables.
        beta (float): Levy-flight exponent.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Levy-flight step with shape (num_vars,) and dtype float64.
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


def _compute_coulomb_force(
    population: np.ndarray,
    index: int,
    kbest: np.ndarray,
    masses: np.ndarray,
    enemy_pos: np.ndarray,
    enemy_mass: float,
    k_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute the Coulomb interaction force for one individual.

    Args:
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        index (int): Current individual index.
        kbest (np.ndarray): Indices of the current k-best set with shape (num_kbest,) and dtype int64.
        masses (np.ndarray): Mass values with shape (num_points,) and dtype float64.
        enemy_pos (np.ndarray): Worst individual position with shape (num_vars,) and dtype float64.
        enemy_mass (float): Mass of the worst individual.
        k_t (float): Iteration-dependent Coulomb coefficient.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Interaction force with shape (num_vars,) and dtype float64.
    """
    force = np.zeros(population.shape[1], dtype=np.float64)

    for other_index in kbest:
        if other_index == index:
            continue
        diff = population[other_index] - population[index]
        dist = np.linalg.norm(diff) + 1e-12
        force += rng.random() * k_t * masses[index] * masses[other_index] * diff / dist

    enemy_diff = enemy_pos - population[index]
    enemy_dist = np.linalg.norm(enemy_diff) + 1e-12
    force -= rng.random() * k_t * masses[index] * enemy_mass * enemy_diff / enemy_dist
    return force


def _update_population(
    population: np.ndarray,
    delta_x: np.ndarray,
    masses: np.ndarray,
    fit_g: np.ndarray,
    kbest: np.ndarray,
    food_pos: np.ndarray,
    enemy_pos: np.ndarray,
    enemy_mass: float,
    neighbor_radius: float,
    inertia: float,
    behavior: float,
    k_t: float,
    levy_beta: float,
    span: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one dragonfly population update.

    Args:
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        delta_x (np.ndarray): Velocity-like state with shape (num_points, num_vars) and dtype float64.
        masses (np.ndarray): Mass values with shape (num_points,) and dtype float64.
        fit_g (np.ndarray): Scaled mass values with shape (num_points,) and dtype float64.
        kbest (np.ndarray): Indices of the current k-best set with shape (num_kbest,) and dtype int64.
        food_pos (np.ndarray): Best position with shape (num_vars,) and dtype float64.
        enemy_pos (np.ndarray): Worst position with shape (num_vars,) and dtype float64.
        enemy_mass (float): Mass of the worst individual.
        neighbor_radius (float): Neighborhood radius.
        inertia (float): Inertia coefficient.
        behavior (float): Social behavior coefficient.
        k_t (float): Iteration-dependent Coulomb coefficient.
        levy_beta (float): Levy-flight exponent.
        span (np.ndarray): Variable ranges with shape (num_vars,) and dtype float64.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Updated population and updated velocity-like state, both with shape
            (num_points, num_vars) and dtype float64.
    """
    num_points, num_vars = population.shape
    max_step = 0.2 * span
    new_population = population.copy()
    new_delta_x = delta_x.copy()

    for i in range(num_points):
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

        force = _compute_coulomb_force(
            population=population,
            index=i,
            kbest=kbest,
            masses=masses,
            enemy_pos=enemy_pos,
            enemy_mass=enemy_mass,
            k_t=k_t,
            rng=rng,
        )
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

    return new_population, new_delta_x


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
    Optimize a continuous objective with the Coulomb-force-search dragonfly algorithm.

    Args:
        func (Callable): Objective function that maps a candidate with shape (num_vars,) and dtype float64
            to either a scalar or an objective vector with shape (num_objectives,) and dtype float64.
        bounds (Union[Bounds, Sequence[Tuple[float, float]]]): Variable bounds for each dimension.
        args (Tuple[Any, ...]): Extra objective arguments.
        maxiter (int): Maximum iterations.
        popsize (int): Population size multiplier.
        tol (float): Convergence tolerance.
        seed (Optional[Union[int, np.random.Generator]]): Random seed or generator.
        polish (bool): Whether to run local refinement at the end.
        constraints (Union[Sequence[Any], Any]): Constraints accepted by SciPy.
        x0 (Optional[np.ndarray]): Initial guess with shape (num_vars,) and dtype float64.
        multi_objective (bool): Whether the objective is multi-objective.
        objective_weights (Optional[np.ndarray]): Objective weights with shape (num_objectives,) and dtype float64.
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
        coulomb_alpha_std (float): Standard deviation of the decay factor.
        k0 (float): Initial Coulomb coefficient.
        levy_beta (float): Levy-flight exponent.

    Returns:
        OptimizeResult: SciPy-style optimization result with population state and optional Pareto front.
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

    rng = _make_rng(seed)

    if neighbor_radius_start is None:
        neighbor_radius_start = 0.25 * float(np.linalg.norm(span))
    neighbor_radius_start = max(neighbor_radius_start, 1e-12)
    neighbor_radius_end = max(neighbor_radius_end, 0.0)

    population = rng.uniform(lower, upper, size=(num_pop, num_vars))
    _apply_initial_guess(population, x0, lower, upper)
    delta_x = rng.uniform(-0.1, 0.1, size=(num_pop, num_vars)) * span

    objective_vectors, objective_scalars, weights = _initialize_objective_values(
        population=population,
        func=func,
        args=args,
        multi_objective=multi_objective,
        objective_weights=objective_weights,
        scalarization=scalarization,
    )
    nfev = num_pop

    violations = _evaluate_constraint_violations(population, constraints, args)
    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * violations

    archive_x: List[np.ndarray] = []
    archive_f: List[np.ndarray] = []
    archive_v: List[float] = []
    if return_pareto and multi_objective:
        _append_archive(archive_x, archive_f, archive_v, population, objective_vectors, violations)

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
        enemy_mass = float(mass[enemy_idx])

        alpha_hat = abs(rng.normal(coulomb_alpha_mean, coulomb_alpha_std))
        k_t = k0 * np.exp(-alpha_hat * (iteration + 1) / maxiter)

        new_population, new_delta_x = _update_population(
            population=population,
            delta_x=delta_x,
            masses=mass,
            fit_g=fit_g,
            kbest=kbest,
            food_pos=food_pos,
            enemy_pos=enemy_pos,
            enemy_mass=enemy_mass,
            neighbor_radius=neighbor_radius,
            inertia=inertia,
            behavior=behavior,
            k_t=k_t,
            levy_beta=levy_beta,
            span=span,
            rng=rng,
        )

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

        new_violations = _evaluate_constraint_violations(new_population, constraints, args)
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
            _append_archive(archive_x, archive_f, archive_v, population, objective_vectors, violations)

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
        if return_pareto and archive_f:
            pareto_x, pareto_f = _finalize_pareto_archive(archive_x, archive_f, archive_v)
            result.pareto_f = pareto_f
            result.pareto_x = pareto_x

    return result
