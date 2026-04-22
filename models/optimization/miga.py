# Multi-Island Genetic Algorithm (MIGA) Optimizer
# Author: Shengning Wang

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

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
# Island Partition
# ============================================================
def _split_islands(num_points: int, num_islands: int) -> List[np.ndarray]:
    """
    Split the population indices into several islands.

    Args:
        num_points (int): Population size.
        num_islands (int): Number of islands.

    Returns:
        List[np.ndarray]: Island index arrays, each with dtype int64.
    """
    # Partition the global population into K islands with
    #     N_k = floor(N / K) + 1_{k < N mod K},
    # so the first `remainder` islands carry one extra individual and
    #     sum_{k=1}^K N_k = N
    # holds exactly.
    base_size = num_points // num_islands
    remainder = num_points % num_islands
    island_sizes = np.full(num_islands, base_size, dtype=int)
    island_sizes[:remainder] += 1

    # Each island must retain at least four members so that the genetic loop can
    # keep elites and still generate at least one pair of offspring without the
    # breeding pool collapsing into a degenerate two-point search.
    if np.any(island_sizes < 4):
        raise ValueError("Each island must contain at least 4 individuals.")

    islands = []
    start = 0
    for size in island_sizes:
        # A contiguous index block [start, start + size) is assigned to each island.
        # This realizes a disjoint partition
        #     I_p cap I_q = empty,   union_k I_k = {0, ..., N - 1}.
        islands.append(np.arange(start, start + size))
        start += size

    return islands


# ============================================================
# Parent Selection
# ============================================================
def _select_parent(indices: np.ndarray, energies: np.ndarray, tournament_size: int, rng: np.random.Generator) -> int:
    """
    Select one parent with tournament selection.

    Args:
        indices (np.ndarray): Candidate indices with shape (num_candidates,) and dtype int64.
        energies (np.ndarray): Penalized objective values with shape (num_points,) and dtype float64.
        tournament_size (int): Tournament size.
        rng (np.random.Generator): Random number generator.

    Returns:
        int: Selected parent index.
    """
    # Tournament selection samples a subset T and returns
    #     i* = arg min_{i in T} E_i,
    # where E_i is the penalized energy. This keeps selection pressure local
    # while preserving enough randomness to avoid premature takeover.
    candidate_size = min(tournament_size, indices.size)
    candidate_indices = rng.choice(indices, size=candidate_size, replace=False)
    return int(candidate_indices[np.argmin(energies[candidate_indices])])


# ============================================================
# Island Breeding
# ============================================================
def _breed_island(
    island: np.ndarray,
    population: np.ndarray,
    energies: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    span: np.ndarray,
    crossover_rate: float,
    mutation_rate: float,
    mutation_scale: float,
    elite_fraction: float,
    tournament_size: int,
    blend_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one island population by elitism, crossover, and mutation.

    Args:
        island (np.ndarray): Island indices with shape (island_size,) and dtype int64.
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        energies (np.ndarray): Penalized objective values with shape (num_points,) and dtype float64.
        lower (np.ndarray): Lower bounds with shape (num_vars,) and dtype float64.
        upper (np.ndarray): Upper bounds with shape (num_vars,) and dtype float64.
        span (np.ndarray): Variable ranges with shape (num_vars,) and dtype float64.
        crossover_rate (float): BLX-alpha crossover probability.
        mutation_rate (float): Per-gene mutation probability.
        mutation_scale (float): Gaussian mutation scale.
        elite_fraction (float): Elite retention ratio.
        tournament_size (int): Tournament size.
        blend_alpha (float): BLX-alpha expansion coefficient.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: New island population with shape (island_size, num_vars) and dtype float64.
    """
    island_size = island.size
    num_vars = population.shape[1]

    # Individuals are ranked by penalized energy
    #     E_i = f_i + lambda * v_i,
    # where f_i is the scalarized objective and v_i is the total constraint
    # violation. Lower energy means better fitness inside each island.
    island_order = island[np.argsort(energies[island])]

    # Elitism keeps
    #     n_elite = ceil(r_elite * N_island)
    # top individuals unchanged so high-quality solutions survive stochastic
    # crossover and mutation.
    elite_count = max(1, int(np.ceil(elite_fraction * island_size)))
    elite_count = min(elite_count, island_size - 2)

    elites = population[island_order[:elite_count]].copy()
    children: List[np.ndarray] = []

    # Offspring are generated until the island is refilled, i.e.
    #     N_children = N_island - n_elite.
    while len(children) < island_size - elite_count:
        # Parent indices come from repeated tournament minimization on the same
        # island, so reproduction is driven by intra-island competition rather
        # than global best individuals.
        idx_a = _select_parent(island, energies, tournament_size, rng)
        idx_b = _select_parent(island, energies, tournament_size, rng)
        parent_a = population[idx_a]
        parent_b = population[idx_b]

        if rng.random() < crossover_rate:
            # BLX-alpha crossover samples each gene from the expanded interval
            #     d_j = |x_j^(a) - x_j^(b)|,
            #     l_j = min(x_j^(a), x_j^(b)) - alpha * d_j,
            #     u_j = max(x_j^(a), x_j^(b)) + alpha * d_j,
            #     x'_j ~ U(l_j, u_j).
            # The interval extends beyond the parents when alpha > 0, which
            # encourages extrapolative exploration around the parental segment.
            diff = np.abs(parent_a - parent_b)
            child_lower = np.minimum(parent_a, parent_b) - blend_alpha * diff
            child_upper = np.maximum(parent_a, parent_b) + blend_alpha * diff
            child_a = rng.uniform(child_lower, child_upper)
            child_b = rng.uniform(child_lower, child_upper)
        else:
            # When crossover is skipped, offspring are exact parent copies:
            #     x'_a = x^(a),   x'_b = x^(b).
            child_a = parent_a.copy()
            child_b = parent_b.copy()

        for child in [child_a, child_b]:
            # Mutation uses an independent Bernoulli mask per gene,
            #     m_j ~ Bernoulli(p_mut),
            # so only a subset of coordinates is perturbed in each offspring.
            mask = rng.random(num_vars) < mutation_rate
            if np.any(mask):
                # Gaussian mutation applies
                #     x'_j <- x'_j + epsilon_j,
                #     epsilon_j ~ N(0, sigma_j^2),
                #     sigma_j = eta * (u_j - l_j),
                # where eta is `mutation_scale`. Scaling by the design span keeps
                # the step size dimensionally consistent across variables.
                sigma = np.maximum(mutation_scale * span[mask], 1e-12)
                child[mask] += rng.normal(0.0, sigma, size=np.sum(mask))

            # Projection/repair maps the mutated point back into the box domain
            #     x'_j <- min(max(x'_j, l_j), u_j),
            # ensuring the island population always remains feasible with
            # respect to bound constraints.
            child = _repair_to_bounds(child, lower, upper)
            children.append(child)
            if len(children) >= island_size - elite_count:
                break

    # The next island generation is the elite block stacked with the truncated
    # offspring matrix:
    #     X_next = [X_elite; X_child].
    return np.vstack([elites, np.asarray(children[:island_size - elite_count])])


# ============================================================
# Island Migration
# ============================================================
def _migrate_islands(
    islands: List[np.ndarray],
    population: np.ndarray,
    objective_vectors: np.ndarray,
    objective_scalars: np.ndarray,
    violations: np.ndarray,
    energies: np.ndarray,
    migration_size: int,
) -> None:
    """
    Exchange the best individuals between neighboring islands.

    Args:
        islands (List[np.ndarray]): Island index arrays.
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        objective_vectors (np.ndarray): Objective matrix with shape (num_points, num_objectives) and dtype float64.
        objective_scalars (np.ndarray): Scalar objectives with shape (num_points,) and dtype float64.
        violations (np.ndarray): Constraint violations with shape (num_points,) and dtype float64.
        energies (np.ndarray): Penalized objective values with shape (num_points,) and dtype float64.
        migration_size (int): Number of migrants per island.
    """
    # Migration is performed on a ring topology. Each island exports its best
    # individuals and receives migrants from its left neighbor
    #     source(k) = (k - 1) mod K.
    # This yields steady information flow without collapsing all islands into a
    # single panmictic population.
    migrants_x = []
    migrants_f = []
    migrants_s = []
    migrants_v = []
    migrants_e = []

    for island in islands:
        # The emigrant set is
        #     M_k = top_m arg min_{i in I_k} E_i,
        # i.e. the lowest-energy individuals on island k together with their
        # objective, scalarized objective, violation, and energy states.
        island_order = island[np.argsort(energies[island])]
        count = min(migration_size, island.size)
        selected = island_order[:count]
        migrants_x.append(population[selected].copy())
        migrants_f.append(objective_vectors[selected].copy())
        migrants_s.append(objective_scalars[selected].copy())
        migrants_v.append(violations[selected].copy())
        migrants_e.append(energies[selected].copy())

    num_islands = len(islands)
    for island_id, island in enumerate(islands):
        # Incoming migrants for island k are the elite solutions exported by the
        # preceding island on the ring. This is a deterministic island graph
        #     k <- (k - 1) mod K.
        source_id = (island_id - 1) % num_islands
        incoming_x = migrants_x[source_id]
        incoming_f = migrants_f[source_id]
        incoming_s = migrants_s[source_id]
        incoming_v = migrants_v[source_id]
        incoming_e = migrants_e[source_id]

        # Replacement targets are chosen as the worst individuals in the
        # receiving island:
        #     R_k = top_m arg max_{i in I_k} E_i.
        # Migrants therefore overwrite poor local solutions instead of
        # disrupting the best island members.
        replace_count = min(incoming_x.shape[0], island.size)
        island_order = island[np.argsort(energies[island])[::-1]]
        replace_idx = island_order[:replace_count]

        # State vectors are replaced atomically so that
        #     (x_i, f_i, s_i, v_i, E_i)
        # stays synchronized after migration.
        population[replace_idx] = incoming_x[:replace_count]
        objective_vectors[replace_idx] = incoming_f[:replace_count]
        objective_scalars[replace_idx] = incoming_s[:replace_count]
        violations[replace_idx] = incoming_v[:replace_count]
        energies[replace_idx] = incoming_e[:replace_count]


# ============================================================
# Multi-Island Genetic Optimization
# ============================================================
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
    Optimize a continuous objective with a multi-island genetic algorithm.

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
        num_islands (int): Number of islands.
        migration_interval (int): Migration interval.
        migration_size (int): Number of migrants per migration step.
        crossover_rate (float): BLX-alpha crossover probability.
        mutation_rate (Optional[float]): Per-gene mutation probability.
        mutation_scale (float): Gaussian mutation scale.
        elite_fraction (float): Elite retention ratio.
        tournament_size (int): Tournament size.
        blend_alpha (float): BLX-alpha expansion coefficient.

    Returns:
        OptimizeResult: SciPy-style optimization result with population state and optional Pareto front.
    """
    # The box-constrained search domain is
    #     Omega = {x in R^d | l_j <= x_j <= u_j, j = 1, ..., d}.
    # `lower`, `upper`, and `span` provide the coordinate-wise bounds and span
    # used later by initialization, mutation, and projection.
    lower, upper = _parse_bounds(bounds)
    num_vars = lower.size
    span = upper - lower
    constraints = _normalize_constraints(constraints)

    # These parameter guards preserve the mathematical meaning of the update
    # rules, e.g. p_c in (0, 1], p_mut in (0, 1], alpha >= 0, and gamma >= 1.
    # Without these inequalities, the crossover interval, mutation process, or
    # penalty schedule would lose their intended probabilistic interpretation.
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

    rng = _make_rng(seed)

    # The global population size follows
    #     N = max(4K, popsize * d),
    # where K is the number of islands and d is the design dimension. The 4K
    # lower bound guarantees at least four individuals per island, which is the
    # minimum required by the breeding routine.
    num_pop = max(num_islands * 4, int(popsize) * num_vars)

    # Initial candidates are sampled uniformly from the hyper-rectangle:
    #     x_i^(0) ~ U([l_1, u_1] x ... x [l_d, u_d]).
    population = rng.uniform(lower, upper, size=(num_pop, num_vars))
    islands = _split_islands(num_pop, num_islands)

    # If an initial guess x0 is provided, it is injected into the population so
    # the evolutionary search can exploit domain knowledge from the first
    # generation instead of relying solely on random sampling.
    _apply_initial_guess(population, x0, lower, upper)

    if mutation_rate is None:
        # A standard default is one expected mutated gene per offspring:
        #     p_mut = 1 / d.
        mutation_rate = 1.0 / max(num_vars, 1)
    if not (0.0 < mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in (0, 1].")

    # Denote the population objective matrix by
    #     F = [f(x_1), ..., f(x_N)]^T,
    # and the scalarized fitness vector by
    #     s = [S(f(x_1)), ..., S(f(x_N))]^T.
    # The helpers below build exactly this pair for the initial generation.
    # For multi-objective problems, the helper returns the objective matrix F
    # together with a scalarized score s(x). Typical scalarizations here are
    # weighted sum
    #     s(x) = w^T f(x),
    # or weighted Chebyshev
    #     s(x) = max_m w_m * |f_m(x) - z_m|.
    objective_vectors, objective_scalars, weights = _initialize_objective_values(
        population=population,
        func=func,
        args=args,
        multi_objective=multi_objective,
        objective_weights=objective_weights,
        scalarization=scalarization,
    )
    nfev = num_pop

    # Constraint handling uses a dynamic penalty energy
    #     E(x) = s(x) + lambda * v(x),
    # where v(x) >= 0 aggregates inequality/equality violations and lambda is
    # increased over iterations to shift the search progressively toward
    # feasibility.
    violations = _evaluate_constraint_violations(population, constraints, args)
    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * violations

    archive_x: List[np.ndarray] = []
    archive_f: List[np.ndarray] = []
    archive_v: List[float] = []
    if return_pareto and multi_objective:
        # Archive entries are later filtered by dominance and feasibility, i.e.
        # candidates with smaller violation are preferred first, and among
        # feasible points the nondominated relation defines the Pareto set.
        # The archive stores nondominated candidates so the final Pareto front is
        # not restricted to the last generation only.
        _append_archive(archive_x, archive_f, archive_v, population, objective_vectors, violations)

    # The incumbent best point is initialized from the minimum-energy individual:
    #     x_best = arg min_i E_i.
    best_index = int(np.argmin(energies))
    best_x = population[best_index].copy()
    best_f = objective_vectors[best_index].copy()
    best_fun = float(objective_scalars[best_index])
    best_energy = float(energies[best_index])

    message = "Maximum number of iterations reached."
    success = False

    for iteration in range(maxiter):
        # The mutation scale decays linearly with the normalized iteration ratio:
        #     r_t = t / (T - 1),
        #     eta_t = eta_0 * (1 - 0.75 r_t).
        # Large early mutations explore the design space, while later mutations
        # concentrate the search around promising regions.
        ratio = iteration / max(maxiter - 1, 1)
        current_mutation_scale = mutation_scale * (1.0 - 0.75 * ratio)
        new_population = np.empty_like(population)

        for island in islands:
            # Each island evolves independently for one generation:
            #     X_k^(t+1) = Breed(X_k^(t)).
            # This preserves diverse search trajectories across sub-populations.
            new_population[island] = _breed_island(
                island=island,
                population=population,
                energies=energies,
                lower=lower,
                upper=upper,
                span=span,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                mutation_scale=current_mutation_scale,
                elite_fraction=elite_fraction,
                tournament_size=tournament_size,
                blend_alpha=blend_alpha,
                rng=rng,
            )

        # Multi-objective evaluation can reuse the current objective matrix as a
        # scalarization reference, e.g. for reference-dependent Chebyshev terms.
        reference_values = objective_vectors if multi_objective else None

        # The offspring population is mapped through the objective operator:
        #     F_new = f(X_new),
        #     s_new = S(F_new),
        # where S(.) is the chosen scalarization rule.
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

        # The generational replacement used here is full replacement:
        #     X^(t+1) <- X_new,
        #     F^(t+1) <- F_new,
        #     E^(t+1) <- E_new.
        # Elitism is already enforced inside each island, so no extra global
        # survivor selection is needed at this level.
        # The penalty factor grows geometrically,
        #     lambda_{t+1} = gamma * lambda_t,   gamma >= 1,
        # so the algorithm increasingly prefers feasible points as generations
        # proceed. The corresponding energy update is
        #     E_i^(t+1) = s_i^(t+1) + lambda_{t+1} v_i^(t+1).
        new_violations = _evaluate_constraint_violations(new_population, constraints, args)
        penalty_factor *= penalty_growth
        new_energies = new_objective_scalars + penalty_factor * new_violations

        population = new_population
        objective_vectors = new_objective_vectors
        objective_scalars = new_objective_scalars
        violations = new_violations
        energies = new_energies

        if return_pareto and multi_objective:
            _append_archive(archive_x, archive_f, archive_v, population, objective_vectors, violations)

        # Periodic migration implements inter-island communication every
        # `migration_interval` generations:
        #     if (t + 1) mod tau_mig = 0, migrate().
        if num_islands > 1 and (iteration + 1) % migration_interval == 0:
            _migrate_islands(
                islands=islands,
                population=population,
                objective_vectors=objective_vectors,
                objective_scalars=objective_scalars,
                violations=violations,
                energies=energies,
                migration_size=migration_size,
            )

        # The incumbent is updated by greedy energy comparison:
        #     if min_i E_i^(t+1) < E_best, then x_best <- arg min_i E_i^(t+1).
        curr_best_idx = int(np.argmin(energies))
        if energies[curr_best_idx] < best_energy:
            best_x = population[curr_best_idx].copy()
            best_f = objective_vectors[curr_best_idx].copy()
            best_fun = float(objective_scalars[curr_best_idx])
            best_energy = float(energies[curr_best_idx])

        # Convergence is declared when the population energy spread becomes small:
        #     std(E) <= tol * max(|mean(E)|, 1).
        # This is a relative stationarity test on the scalarized penalized
        # fitness landscape.
        if np.std(energies) <= tol * max(np.abs(np.mean(energies)), 1.0):
            success = True
            message = "Optimization converged."
            break

    if polish:
        # Optional polishing solves the local penalized problem
        #     min_x  s(x) + lambda v(x)
        # subject to the same box/explicit constraints, using the incumbent best
        # evolutionary solution as the local starting point.
        def local_objective(x_local: np.ndarray) -> float:
            values = np.atleast_1d(np.asarray(func(x_local, *args), dtype=np.float64)).reshape(-1)
            if values.size == 1:
                # Single-objective case: s(x) = f(x).
                value = float(values[0])
            else:
                local_weights = _normalize_weights(values.size, objective_weights)
                if scalarization == "weighted_sum":
                    # Weighted-sum scalarization:
                    #     s(x) = w^T f(x).
                    value = float(np.dot(local_weights, values))
                else:
                    # Weighted Chebyshev scalarization:
                    #     s(x) = max_m w_m * |f_m(x) - f_best,m|,
                    # where the current best objective vector plays the role of
                    # the local reference in the polishing stage.
                    value = float(np.max(local_weights * np.abs(values - best_f)))
            # The local objective uses the same penalty form as the evolutionary
            # phase so feasibility pressure remains consistent.
            return value + penalty_factor * _constraint_violation(x_local, constraints, args)

        # L-BFGS-B is used for pure bound constraints, while SLSQP handles the
        # explicit nonlinear/linear SciPy constraints:
        #     method = L-BFGS-B  if C = empty,
        #              SLSQP     otherwise.
        polish_method = "L-BFGS-B" if not constraints else "SLSQP"
        polish_result = minimize(
            local_objective,
            x0=best_x,
            method=polish_method,
            bounds=Bounds(lower, upper),
            constraints=constraints if constraints else (),
        )
        nfev += int(getattr(polish_result, "nfev", 0))

        # The polished point is accepted only when it improves the penalized
        # merit function:
        #     phi(x_polish) < phi(x_best).
        if polish_result.fun < best_energy:
            best_x = np.asarray(polish_result.x, dtype=np.float64)
            best_f = np.atleast_1d(np.asarray(func(best_x, *args), dtype=np.float64)).reshape(-1)
            nfev += 1
            if best_f.size == 1:
                best_fun = float(best_f[0])
            else:
                local_weights = _normalize_weights(best_f.size, objective_weights)
                if scalarization == "weighted_sum":
                    # Keep the final reported scalar objective consistent with
                    # the evolutionary weighted-sum aggregation.
                    best_fun = float(np.dot(local_weights, best_f))
                else:
                    # For Chebyshev aggregation, the final reference is the
                    # componentwise ideal estimate from the last population:
                    #     z_m = min_i f_{i,m}.
                    reference = np.min(objective_vectors, axis=0)
                    # Final scalar score:
                    #     s(x_best) = max_m w_m * |f_m(x_best) - z_m|.
                    best_fun = float(np.max(local_weights * np.abs(best_f - reference)))
            best_energy = float(polish_result.fun)

    # Assemble a SciPy-like result object that exposes both the best incumbent
    # and the final evolutionary population state for downstream analysis.
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
        # `fun_vector` stores the raw objective vector f(x_best), while the scalar
        # field `fun` stores the corresponding aggregation S(f(x_best)).
        result.fun_vector = best_f.copy()
        if return_pareto and archive_f:
            # The final Pareto extraction removes dominated archive members and
            # returns a set approximating
            #     P* = {x in Omega | not exists y: f(y) <= f(x), f(y) != f(x)}.
            # The archive is filtered into the final nondominated set:
            #     P = {x_i | not exists j: f_j <= f_i and f_j != f_i}.
            pareto_x, pareto_f = _finalize_pareto_archive(archive_x, archive_f, archive_v)
            result.pareto_f = pareto_f
            result.pareto_x = pareto_x

    return result
