# CFARSSDA Dragonfly Optimizer
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


# ============================================================
# Levy Flight And Physics Proxies
# ============================================================

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

    # Mantegna's Levy-flight scale:
    # sigma_u = [Gamma(1 + beta) sin(pi beta / 2)
    #           / (Gamma((1 + beta) / 2) beta 2^((beta - 1) / 2))]^(1 / beta).
    sigma_u = (
        gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
        / (gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)

    # Draw u ~ N(0, sigma_u^2) and v ~ N(0, 1), then form
    # L = u / |v|^(1 / beta) to obtain the heavy-tailed flight increment.
    u = rng.normal(0.0, sigma_u, size=num_vars)
    v = rng.normal(0.0, 1.0, size=num_vars)

    # The 0.01 factor keeps the Levy jump in the same scale as the bounded design space.
    return 0.01 * u / (np.abs(v) ** (1.0 / beta) + 1e-12)


def _compute_air_density(
    candidate: np.ndarray,
    lower: np.ndarray,
    span: np.ndarray,
    sea_level_density: float,
    temperature_ratio_floor: float,
) -> float:
    """
    Estimate the local air density proxy from a candidate position.

    Args:
        candidate (np.ndarray): Candidate position with shape (num_vars,).
        lower (np.ndarray): Lower bounds with shape (num_vars,).
        span (np.ndarray): Variable span with shape (num_vars,).
        sea_level_density (float): Reference sea-level air density.
        temperature_ratio_floor (float): Lower bound for the temperature ratio proxy.

    Returns:
        float: Air density proxy.
    """
    # Normalize the candidate into an altitude-like coordinate:
    # h_hat = mean((x - l) / (u - l)) in [0, 1].
    normalized_altitude = np.clip(np.mean((candidate - lower) / (span + 1e-12)), 0.0, 1.0)

    # Use a linear temperature-ratio proxy
    # tau = 1 - (1 - tau_min) h_hat,
    # then map density as rho = rho_0 * tau.
    temperature_ratio = 1.0 - (1.0 - temperature_ratio_floor) * normalized_altitude
    return float(sea_level_density * max(temperature_ratio, 1e-6))


def _compute_block_area(
    mass_value: float,
    mass_scale: float,
    area_min: float,
    area_max: float,
    area_shape: float,
) -> float:
    """
    Map a mass value to an adaptive blocking area.

    Args:
        mass_value (float): Current mass-like value.
        mass_scale (float): Population scale reference.
        area_min (float): Lower asymptote of the blocking area.
        area_max (float): Upper asymptote of the blocking area.
        area_shape (float): Logistic shape factor.

    Returns:
        float: Adaptive blocking area.
    """
    # Convert the current effective mass into a normalized ratio
    # r_m = m / m_max in [0, 1].
    ratio = np.clip(mass_value / max(mass_scale, 1e-12), 0.0, 1.0)

    # The adaptive area follows a logistic law
    # A(r_m) = A_min + (A_max - A_min) / (1 + exp(-k (r_m - 0.5))),
    # so larger masses induce a wider aerodynamic blocking area.
    logistic = 1.0 / (1.0 + np.exp(-area_shape * (ratio - 0.5)))
    return float(area_min + (area_max - area_min) * logistic)


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
    # Coulomb-like interaction model:
    # F_i = sum_{j in K_best, j != i} r_ij k_t m_i m_j (x_j - x_i) / ||x_j - x_i||^2
    #       - r_ie k_t m_i m_e (x_e - x_i) / ||x_e - x_i||^2.
    force = np.zeros(population.shape[1], dtype=np.float64)

    for other_index in kbest:
        if other_index == index:
            continue

        # The attractive component is accumulated from the current elite set K_best.
        diff = population[other_index] - population[index]
        dist = np.linalg.norm(diff) + 1e-12
        force += rng.random() * k_t * masses[index] * masses[other_index] * diff / (dist**2)

    # The enemy term enters with the opposite sign and pushes the individual away
    # from the current worst location x_e.
    enemy_diff = enemy_pos - population[index]
    enemy_dist = np.linalg.norm(enemy_diff) + 1e-12
    force -= rng.random() * k_t * masses[index] * enemy_mass * enemy_diff / (enemy_dist**2)
    return force


# ============================================================
# Dragonfly State Update
# ============================================================

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
    lower: np.ndarray,
    span: np.ndarray,
    drag_coefficient: float,
    sea_level_density: float,
    temperature_ratio_floor: float,
    area_min: float,
    area_max: float,
    area_shape: float,
    stamina: float,
    cohesion_floor: float,
    food_weight: float,
    enemy_weight: float,
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
        lower (np.ndarray): Lower bounds with shape (num_vars,).
        span (np.ndarray): Variable ranges with shape (num_vars,) and dtype float64.
        drag_coefficient (float): Air drag coefficient.
        sea_level_density (float): Reference sea-level air density.
        temperature_ratio_floor (float): Lower bound for the temperature ratio proxy.
        area_min (float): Lower asymptote of the blocking area.
        area_max (float): Upper asymptote of the blocking area.
        area_shape (float): Logistic shape factor.
        stamina (float): Flight stamina.
        cohesion_floor (float): Lower bound for the cohesion scaling.
        food_weight (float): Food attraction weight.
        enemy_weight (float): Enemy repulsion weight.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Updated population and updated velocity-like state, both with shape
            (num_points, num_vars) and dtype float64.
    """
    num_points, num_vars = population.shape

    # Each coordinate step is clipped by
    # Delta_max = 0.2 * (u - l),
    # which keeps the flight increment inside a controlled fraction of the box span.
    max_step = 0.2 * span
    new_population = population.copy()
    new_delta_x = delta_x.copy()
    max_fit_g = float(np.max(fit_g))

    for i in range(num_points):
        # Neighborhood definition:
        # N_i = {j != i | ||x_j - x_i||_2 <= R_t}.
        distances = np.linalg.norm(population - population[i], axis=1)
        neighbors = np.where((distances > 0.0) & (distances <= neighbor_radius))[0]

        # Aerodynamic proxies are recomputed from the current position and effective mass.
        air_density = _compute_air_density(
            candidate=population[i],
            lower=lower,
            span=span,
            sea_level_density=sea_level_density,
            temperature_ratio_floor=temperature_ratio_floor,
        )
        block_area = _compute_block_area(
            mass_value=fit_g[i],
            mass_scale=max_fit_g,
            area_min=area_min,
            area_max=area_max,
            area_shape=area_shape,
        )
        speed = float(np.linalg.norm(delta_x[i]))

        # Quadratic drag model:
        # F_d = -(1 / 2) C_d rho A ||Delta x_i|| Delta x_i,
        # a_d = F_d / g_i.
        drag_force = -0.5 * drag_coefficient * air_density * block_area * speed * delta_x[i]
        drag_acceleration = drag_force / (fit_g[i] + 1e-12)

        # Cohesion is weakened by a large blocking area:
        # s_coh = max(s_coh,min, 1 - A / A_max).
        cohesion_scale = max(cohesion_floor, 1.0 - block_area / max(area_max, 1e-12))

        if neighbors.size == 0:
            # Isolated individuals switch to Levy exploration:
            # Delta x_i^(t+1) = eta_t L_i + a_d,
            # x_i^(t+1) = x_i^t + clip(Delta x_i^(t+1), -Delta_max, Delta_max).
            levy_step = _levy_flight(num_vars, levy_beta, rng) * np.maximum(np.abs(population[i]), 1.0)
            step = stamina * levy_step + drag_acceleration
            step = np.clip(step, -max_step, max_step)
            new_delta_x[i] = step
            new_population[i] = population[i] + step
            continue

        # Canonical dragonfly social terms:
        # S_i = -sum_{j in N_i} (x_i - x_j)
        # A_i = mean_{j in N_i} Delta x_j
        # C_i = mean_{j in N_i} x_j - x_i.
        separation = -np.sum(population[i] - population[neighbors], axis=0)
        alignment = np.mean(delta_x[neighbors], axis=0)
        cohesion = np.mean(population[neighbors], axis=0) - population[i]

        # Food attraction and enemy avoidance:
        # F_i = x_food - x_i,
        # E_i = x_i - x_enemy.
        food_attraction = food_pos - population[i]
        enemy_avoidance = population[i] - enemy_pos

        # Coulomb acceleration augments the social update with elite interactions:
        # a_i^C = F_i^C / g_i.
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

        # The full CFARSSDA velocity-like update is
        # Delta x_i^(t+1) = eta_t [
        #     w_t Delta x_i^t
        #     + c_t r_1 S_i
        #     + c_t r_2 A_i
        #     + c_t s_coh r_3 C_i
        #     + w_f r_4 F_i
        #     + w_e c_t r_5 E_i
        #     + a_i^C
        #     + a_i^d
        # ].
        step = (
            inertia * delta_x[i]
            + behavior * rng.random() * separation
            + behavior * rng.random() * alignment
            + cohesion_scale * behavior * rng.random() * cohesion
            + food_weight * rng.random() * food_attraction
            + enemy_weight * behavior * rng.random() * enemy_avoidance
            + acceleration
            + drag_acceleration
        )

        # Apply stamina scaling eta_t and then update
        # x_i^(t+1) = x_i^t + clip(Delta x_i^(t+1), -Delta_max, Delta_max).
        step *= stamina
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
    drag_coefficient: float = 0.5,
    sea_level_density: float = 1.225,
    temperature_ratio_floor: float = 0.85,
    area_min: float = 0.2,
    area_max: float = 1.0,
    area_shape: float = 6.0,
    stamina_decay: float = 3.0,
    stamina_floor: float = 0.25,
    cohesion_floor: float = 0.3,
    food_weight: float = 2.0,
    enemy_weight: float = 1.0,
) -> OptimizeResult:
    """
    Optimize a continuous objective with the CFARSSDA dragonfly algorithm.

    Args:
        func (Callable): Objective function mapping a candidate with shape (num_vars,) to a scalar
            or an objective vector with shape (num_objectives,).
        bounds (Union[Bounds, Sequence[Tuple[float, float]]]): Variable bounds for each dimension.
        args (Tuple[Any, ...]): Extra objective arguments.
        maxiter (int): Maximum iterations.
        popsize (int): Population size multiplier.
        tol (float): Convergence tolerance.
        seed (Optional[Union[int, np.random.Generator]]): Random seed or generator.
        polish (bool): Whether to run local refinement at the end.
        constraints (Union[Sequence[Any], Any]): Constraints accepted by SciPy.
        x0 (Optional[np.ndarray]): Initial guess with shape (num_vars,).
        multi_objective (bool): Whether the objective is multi-objective.
        objective_weights (Optional[np.ndarray]): Objective weights with shape (num_objectives,).
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
        drag_coefficient (float): Air drag coefficient.
        sea_level_density (float): Reference sea-level air density.
        temperature_ratio_floor (float): Lower bound for the temperature ratio proxy.
        area_min (float): Lower asymptote of the blocking area.
        area_max (float): Upper asymptote of the blocking area.
        area_shape (float): Logistic shape factor.
        stamina_decay (float): Stamina decay rate.
        stamina_floor (float): Stamina floor.
        cohesion_floor (float): Lower bound for the cohesion scaling.
        food_weight (float): Food attraction weight.
        enemy_weight (float): Enemy repulsion weight.

    Returns:
        OptimizeResult: SciPy-style optimization result with population state and optional Pareto front.
    """
    lower, upper = _parse_bounds(bounds)
    num_vars = lower.size

    # The algorithm uses N = max(20, popsize * D) particles for D design variables.
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
    if drag_coefficient < 0.0:
        raise ValueError("drag_coefficient must be >= 0.")
    if sea_level_density <= 0.0:
        raise ValueError("sea_level_density must be > 0.")
    if not (0.0 < temperature_ratio_floor <= 1.0):
        raise ValueError("temperature_ratio_floor must be in (0, 1].")
    if area_min <= 0.0:
        raise ValueError("area_min must be > 0.")
    if area_max < area_min:
        raise ValueError("area_max must be >= area_min.")
    if area_shape <= 0.0:
        raise ValueError("area_shape must be > 0.")
    if stamina_decay < 0.0:
        raise ValueError("stamina_decay must be >= 0.")
    if not (0.0 < stamina_floor <= 1.0):
        raise ValueError("stamina_floor must be in (0, 1].")
    if not (0.0 < cohesion_floor <= 1.0):
        raise ValueError("cohesion_floor must be in (0, 1].")
    if food_weight <= 0.0:
        raise ValueError("food_weight must be > 0.")
    if enemy_weight <= 0.0:
        raise ValueError("enemy_weight must be > 0.")

    rng = _make_rng(seed)

    if neighbor_radius_start is None:
        # Default initial neighborhood radius:
        # R_0 = 0.25 ||u - l||_2.
        neighbor_radius_start = 0.25 * float(np.linalg.norm(span))
    neighbor_radius_start = max(neighbor_radius_start, 1e-12)
    neighbor_radius_end = max(neighbor_radius_end, 0.0)

    # Initialize positions and velocity-like increments as
    # x_i^0 ~ U(l, u),
    # Delta x_i^0 ~ U(-0.1, 0.1) .* (u - l).
    population = rng.uniform(lower, upper, size=(num_pop, num_vars))
    _apply_initial_guess(population, x0, lower, upper)
    delta_x = rng.uniform(-0.1, 0.1, size=(num_pop, num_vars)) * span

    # The initial scalar objective is either
    # s(x) = w^T f(x) for weighted-sum scalarization,
    # or s(x) = max_k w_k |f_k(x) - z_k^*| for Tchebycheff scalarization.
    objective_vectors, objective_scalars, weights = _initialize_objective_values(
        population=population,
        func=func,
        args=args,
        multi_objective=multi_objective,
        objective_weights=objective_weights,
        scalarization=scalarization,
    )
    nfev = num_pop

    # Penalized merit function:
    # E_i = s_i + lambda_0 v_i,
    # where v_i is the aggregated constraint violation.
    violations = _evaluate_constraint_violations(population, constraints, args)
    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * violations

    archive_x: List[np.ndarray] = []
    archive_f: List[np.ndarray] = []
    archive_v: List[float] = []
    if return_pareto and multi_objective:
        # Store the current nondominated candidates in the external Pareto archive.
        _append_archive(archive_x, archive_f, archive_v, population, objective_vectors, violations)

    # The food source is tracked through the best penalized energy found so far.
    best_idx = int(np.argmin(energies))
    best_x = population[best_idx].copy()
    best_f = objective_vectors[best_idx].copy()
    best_fun = float(objective_scalars[best_idx])
    best_energy = float(energies[best_idx])

    message = "Maximum number of iterations reached."
    success = False

    for iteration in range(maxiter):
        # Use the normalized iteration ratio
        # r_t = t / (T - 1)
        # to drive the linear and exponential schedules below.
        ratio = iteration / max(maxiter - 1, 1)

        # Time-varying coefficients:
        # w_t = w_0 + (w_T - w_0) r_t,
        # c_t = c_max + (c_min - c_max) r_t,
        # R_t = R_0 + (R_T - R_0) r_t,
        # eta_t = eta_min + (1 - eta_min) exp(-k_eta r_t).
        inertia = inertia_start + (inertia_end - inertia_start) * ratio
        behavior = c_max + (c_min - c_max) * ratio
        neighbor_radius = neighbor_radius_start + (neighbor_radius_end - neighbor_radius_start) * ratio
        stamina = stamina_floor + (1.0 - stamina_floor) * np.exp(-stamina_decay * ratio)

        # Convert penalized energies into positive masses via
        # m_i^raw = (E_i - E_worst) / (E_best - E_worst),
        # m_i = m_i^raw / sum_j m_j^raw.
        curr_best = float(np.min(energies))
        curr_worst = float(np.max(energies))
        mass_raw = (energies - curr_worst) / (curr_best - curr_worst + 1e-12)
        mass_raw = np.maximum(mass_raw, 1e-12)
        mass = mass_raw / (np.sum(mass_raw) + 1e-12)

        # The effective inertial mass is scaled again as
        # gamma_i = c_min + (c_max - c_min) m_i^raw,
        # g_i = gamma_i m_i.
        gamma_w = c_min + (c_max - c_min) * mass_raw
        fit_g = gamma_w * mass

        # The elite set size contracts from N to 1 according to
        # K_t = ceil(N - (N - 1) r_t),
        # which gradually turns collective search into exploitation.
        order = np.argsort(mass)[::-1]
        kbest_count = max(1, int(np.ceil(num_pop - (num_pop - 1) * ratio)))
        kbest = order[:kbest_count]

        # Food is the best-energy particle and enemy is the worst-energy particle.
        food_idx = int(np.argmin(energies))
        enemy_idx = int(np.argmax(energies))
        food_pos = population[food_idx]
        enemy_pos = population[enemy_idx]
        enemy_mass = float(mass[enemy_idx])

        # The Coulomb coefficient decays exponentially:
        # alpha_hat ~ |N(mu_alpha, sigma_alpha^2)|,
        # k_t = k_0 exp(-alpha_hat (t + 1) / T).
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
            lower=lower,
            span=span,
            drag_coefficient=drag_coefficient,
            sea_level_density=sea_level_density,
            temperature_ratio_floor=temperature_ratio_floor,
            area_min=area_min,
            area_max=area_max,
            area_shape=area_shape,
            stamina=stamina,
            cohesion_floor=cohesion_floor,
            food_weight=food_weight,
            enemy_weight=enemy_weight,
            rng=rng,
        )

        # Project any out-of-box step back onto the feasible bound set [l, u].
        new_population = _repair_to_bounds(new_population, lower, upper)

        # For Tchebycheff scalarization, the current objective vectors act as
        # the moving reference set from which z^* = min f is derived.
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

        # Increase the penalty factor geometrically:
        # lambda_{t+1} = lambda_t * rho_pen,
        # then recompute E_i^(t+1) = s_i^(t+1) + lambda_{t+1} v_i^(t+1).
        new_violations = _evaluate_constraint_violations(new_population, constraints, args)
        penalty_factor *= penalty_growth
        new_energies = new_objective_scalars + penalty_factor * new_violations

        population = new_population
        delta_x = new_delta_x
        objective_vectors = new_objective_vectors
        objective_scalars = new_objective_scalars
        violations = new_violations
        energies = new_energies

        if return_pareto and multi_objective:
            _append_archive(archive_x, archive_f, archive_v, population, objective_vectors, violations)

        # Keep the incumbent minimizer of the penalized merit E(x).
        curr_best_idx = int(np.argmin(energies))
        if energies[curr_best_idx] < best_energy:
            best_x = population[curr_best_idx].copy()
            best_f = objective_vectors[curr_best_idx].copy()
            best_fun = float(objective_scalars[curr_best_idx])
            best_energy = float(energies[curr_best_idx])

        # Convergence is declared when the population energy spread satisfies
        # std(E) <= tol * max(|mean(E)|, 1),
        # meaning the swarm has largely collapsed onto one energy level.
        if np.std(energies) <= tol * max(np.abs(np.mean(energies)), 1.0):
            success = True
            message = "Optimization converged."
            break

    if polish:

        def local_objective(x_local: np.ndarray) -> float:
            # Local refinement solves the same penalized merit function
            # Phi(x) = s(x) + lambda v(x),
            # so the polish stage stays consistent with the global search target.
            values = np.atleast_1d(np.asarray(func(x_local, *args), dtype=np.float64)).reshape(-1)
            if values.size == 1:
                value = float(values[0])
            else:
                local_weights = _normalize_weights(values.size, objective_weights)
                if scalarization == "weighted_sum":
                    # Weighted-sum scalarization:
                    # s(x) = w^T f(x).
                    value = float(np.dot(local_weights, values))
                else:
                    # Tchebycheff scalarization:
                    # s(x) = max_k w_k |f_k(x) - z_k^*|,
                    # with z_k^* taken from the current population minimum.
                    reference = np.min(objective_vectors, axis=0)
                    value = float(np.max(local_weights * np.abs(values - reference)))
            return value + penalty_factor * _constraint_violation(x_local, constraints, args)

        # Use box-only L-BFGS-B when unconstrained, otherwise switch to SLSQP
        # to minimize Phi(x) subject to the provided constraints.
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
            # Accept the polished candidate only if it improves the penalized merit value.
            best_x = np.asarray(polish_result.x, dtype=np.float64)
            best_f = np.atleast_1d(np.asarray(func(best_x, *args), dtype=np.float64)).reshape(-1)
            nfev += 1
            if best_f.size == 1:
                best_fun = float(best_f[0])
            else:
                local_weights = _normalize_weights(best_f.size, objective_weights)
                if scalarization == "weighted_sum":
                    # Reconstruct the reported scalar value with s(x) = w^T f(x).
                    best_fun = float(np.dot(local_weights, best_f))
                else:
                    # Reconstruct the reported scalar value with
                    # s(x) = max_k w_k |f_k(x) - z_k^*|.
                    reference = np.min(objective_vectors, axis=0)
                    best_fun = float(np.max(local_weights * np.abs(best_f - reference)))
            best_energy = float(polish_result.fun)

    # Package both the physical swarm state and the optimization summary
    # so downstream code can reuse the final population and Pareto archive.
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
    result.optimizer = "CFARSSDA"

    if multi_objective:
        result.fun_vector = best_f.copy()
        if return_pareto and archive_f:
            pareto_x, pareto_f = _finalize_pareto_archive(archive_x, archive_f, archive_v)
            result.pareto_f = pareto_f
            result.pareto_x = pareto_x

    return result
