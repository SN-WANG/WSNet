"""Coulomb Force Search Strategy-based Dragonfly Algorithm (CFSSDA).

This module implements the CFSSDA optimizer proposed in:
    Yuan Yongliang, "Integrated Optimization Design of Bucket-Wheel Reclaimer
    Structure and Operating Parameters", Dalian University of Technology, PhD Thesis.

The Coulomb Force Search Strategy (CFSS) augments the standard Dragonfly Algorithm
(DA, Mirjalili 2016) with gravitational-search-inspired inter-individual Coulomb
forces, enabling faster convergence and higher solution accuracy on high-dimensional
nonlinear problems.

The public entry point ``dragonfly_optimize`` mirrors the SciPy
``differential_evolution`` interface so it can be used as a drop-in replacement
for constrained / unconstrained, single / multi-objective problems.

Code author: Shengning Wang
"""

from __future__ import annotations

from dataclasses import dataclass
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
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

EPS: float = 1e-12  # Numerical epsilon to guard against division by zero.


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------


@dataclass
class _ObjectiveValue:
    """Container pairing a scalarized fitness value with the raw objective vector.

    Attributes:
        scalar: Scalarized fitness used for population ranking and selection.
        vector: Raw objective vector returned by the user's objective function.
            Shape: (n_obj,).
    """

    scalar: float
    vector: np.ndarray


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _parse_bounds(
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a SciPy ``Bounds`` object or a sequence of (lb, ub) pairs.

    Args:
        bounds: Box constraints specified either as a ``scipy.optimize.Bounds``
            instance or as an array-like of shape (n_dim, 2) where each row
            is ``[lower_i, upper_i]``.

    Returns:
        Tuple of two 1-D arrays ``(lower, upper)``, each of shape (n_dim,),
        containing the lower and upper bounds respectively.

    Raises:
        ValueError: If ``bounds`` has the wrong shape, contains non-finite
            values, or any upper bound is not strictly greater than the
            corresponding lower bound.

    Complexity:
        Time:  O(n_dim)
        Space: O(n_dim)
    """
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
        raise ValueError("all bounds must be finite for dragonfly_optimize")
    if np.any(upper <= lower):
        raise ValueError("each upper bound must be strictly greater than lower bound")
    return lower, upper


def _levy_flight(
    rng: np.random.Generator,
    n_dim: int,
    beta: float = 1.5,
) -> np.ndarray:
    """Generate a Levy-flight displacement vector using Mantegna's algorithm.

    Implements the Levy(d) function from the CFSSDA paper (Eq. 3.9–3.10).
    The step length follows a power-law (Levy stable) distribution, enabling
    long-range exploration when a dragonfly has no neighbours.

    Args:
        rng: NumPy random Generator instance for reproducibility.
        n_dim: Dimensionality of the search space. Must be >= 1.
        beta: Levy stability index. Typical value is 1.5. Must satisfy
            0 < beta <= 2.

    Returns:
        A 1-D displacement vector of shape (n_dim,) scaled by 0.01.

    Raises:
        ValueError: If ``n_dim`` < 1 or ``beta`` is outside (0, 2].

    Complexity:
        Time:  O(n_dim)
        Space: O(n_dim)
    """
    if n_dim < 1:
        raise ValueError(f"n_dim must be >= 1, got {n_dim}")
    if not (0.0 < beta <= 2.0):
        raise ValueError(f"beta must be in (0, 2], got {beta}")

    sigma_u = (
        gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
        / (gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)

    u = rng.normal(0.0, sigma_u, size=n_dim)
    v = rng.normal(0.0, 1.0, size=n_dim)
    step = u / ((np.abs(v) + EPS) ** (1.0 / beta))
    return 0.01 * step


def _constraint_violation(
    x: np.ndarray,
    constraints: Sequence[Any],
    args: Tuple[Any, ...],
) -> float:
    """Compute the total aggregated constraint violation at point ``x``.

    Supports three constraint formats: ``scipy.optimize.LinearConstraint``,
    ``scipy.optimize.NonlinearConstraint``, and SLSQP-style dict constraints
    with keys ``"type"`` (``"eq"`` or ``"ineq"``) and ``"fun"``.

    Violation is measured as the sum of positive exceedances:
        - Inequality ``g(x) >= lb``:  violation += max(lb - g(x), 0)
        - Equality   ``h(x)  = 0`` :  violation += |h(x)|

    Args:
        x: Current candidate solution. Shape: (n_dim,).
        constraints: Sequence of constraint objects. An empty sequence
            results in zero violation.
        args: Extra positional arguments forwarded to dict-style constraint
            functions.

    Returns:
        Non-negative scalar representing total constraint violation.
        Returns 0.0 when ``constraints`` is empty.

    Raises:
        ValueError: If a dict constraint is missing the ``"fun"`` key or has
            an unrecognized ``"type"`` value.
        TypeError: If a constraint object is not one of the three supported
            formats.

    Complexity:
        Time:  O(C * n_dim)  where C = len(constraints)
        Space: O(n_dim)
    """
    if not constraints:
        return 0.0

    violation = 0.0
    for c in constraints:
        if isinstance(c, LinearConstraint):
            values = np.atleast_1d(c.A @ x)
            lb = np.atleast_1d(c.lb).astype(float)
            ub = np.atleast_1d(c.ub).astype(float)
            low_v = np.maximum(lb - values, 0.0)
            up_v = np.maximum(values - ub, 0.0)
            low_v[~np.isfinite(lb)] = 0.0
            up_v[~np.isfinite(ub)] = 0.0
            violation += float(np.sum(low_v + up_v))

        elif isinstance(c, NonlinearConstraint):
            values = np.atleast_1d(c.fun(x))
            lb = np.atleast_1d(c.lb).astype(float)
            ub = np.atleast_1d(c.ub).astype(float)
            low_v = np.maximum(lb - values, 0.0)
            up_v = np.maximum(values - ub, 0.0)
            low_v[~np.isfinite(lb)] = 0.0
            up_v[~np.isfinite(ub)] = 0.0
            violation += float(np.sum(low_v + up_v))

        elif isinstance(c, dict):
            c_type = c.get("type", "").lower()
            c_fun = c.get("fun")
            c_args = c.get("args", args)
            if c_fun is None:
                raise ValueError("constraint dict must contain key 'fun'")
            values = np.atleast_1d(c_fun(x, *c_args))
            if c_type == "ineq":
                violation += float(np.sum(np.maximum(-values, 0.0)))
            elif c_type == "eq":
                violation += float(np.sum(np.abs(values)))
            else:
                raise ValueError(f"unsupported constraint dict type: '{c_type}'")

        else:
            raise TypeError(
                "each constraint must be a LinearConstraint, NonlinearConstraint, "
                "or an SLSQP-style dict with keys 'type' and 'fun'"
            )
    return violation


def _nondominated_indices(objective_matrix: np.ndarray) -> np.ndarray:
    """Return the indices of Pareto non-dominated solutions (minimization).

    A solution i is non-dominated if no other solution j satisfies
    ``obj_j <= obj_i`` in all objectives and ``obj_j < obj_i`` in at least
    one objective.

    Args:
        objective_matrix: Matrix of objective values. Shape: (N, n_obj).
            Each row represents one solution's objective vector.

    Returns:
        Integer array of indices corresponding to non-dominated solutions.
        Shape: (n_nd,) where n_nd <= N.

    Complexity:
        Time:  O(N^2 * n_obj)
        Space: O(N)
    """
    n = objective_matrix.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        better_or_equal = np.all(objective_matrix <= objective_matrix[i], axis=1)
        strictly_better = np.any(objective_matrix < objective_matrix[i], axis=1)
        dominates_i = better_or_equal & strictly_better
        dominates_i[i] = False
        if np.any(dominates_i):
            is_dominated[i] = True
    return np.where(~is_dominated)[0]


def _reflect_bounds(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Reflect out-of-bounds positions back into the feasible box, then clip.

    Any coordinate that overshoots a boundary is mirrored back by the amount
    of the exceedance. A final ``np.clip`` guarantees no coordinate escapes
    after double reflection on very large steps.

    Args:
        x: Candidate positions. Shape: (N, n_dim) or (n_dim,).
        lower: Lower bound vector. Shape: (n_dim,).
        upper: Upper bound vector. Shape: (n_dim,).

    Returns:
        Corrected positions array of the same shape as ``x``, fully
        contained within ``[lower, upper]``.

    Complexity:
        Time:  O(N * n_dim)
        Space: O(N * n_dim)
    """
    y = x.copy()
    y = np.where(y < lower, 2.0 * lower - y, y)
    y = np.where(y > upper, 2.0 * upper - y, y)
    return np.clip(y, lower, upper)


# ---------------------------------------------------------------------------
# Public optimiser
# ---------------------------------------------------------------------------


def dragonfly_optimize(
    func: Callable[..., ArrayLike],
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
    args: Tuple[Any, ...] = (),
    maxiter: int = 200,
    popsize: int = 30,
    tol: float = 1e-6,
    mutation: Optional[Union[float, Tuple[float, float]]] = None,
    recombination: Optional[float] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    callback: Optional[Callable[[np.ndarray, float], bool]] = None,
    disp: bool = False,
    polish: bool = False,
    init: Union[str, np.ndarray] = "random",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int = 1,
    constraints: Union[Sequence[Any], Any] = (),
    x0: Optional[ArrayLike] = None,
    integrality: Optional[ArrayLike] = None,
    vectorized: bool = False,
    *,
    multi_objective: bool = False,
    objective_weights: Optional[ArrayLike] = None,
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
    """Minimise an objective function using the CFSSDA algorithm.

    Implements the Coulomb Force Search Strategy-based Dragonfly Algorithm
    (CFSSDA). The step update equation (Eq. 3.32 in the source thesis) is:

        ΔX(t+1) = w·ΔX(t) + s·S_i + a·A_i + c·C_i + f·F_i + e·E_i + a_i(t)

    where ``a_i(t)`` is the Coulomb acceleration derived from inter-individual
    attractive forces (kbest peers) and the repulsive force from the worst
    individual (enemy).  When a dragonfly has no neighbours its position is
    updated via Levy flight (Eq. 3.8): X(t+1) = X(t) + Levy(d) · X(t).

    The interface mirrors ``scipy.optimize.differential_evolution`` so this
    function can be used as a drop-in replacement.  Parameters that are
    accepted purely for API compatibility (``mutation``, ``recombination``,
    ``updating``, ``workers``, ``integrality``) are silently ignored.

    Args:
        func: Objective function ``f(x, *args) -> scalar | array``.
            Input shape (n_dim,); returns a scalar for single-objective or
            a 1-D array of length >= 2 for multi-objective mode.
        bounds: Box constraints for each decision variable, specified as a
            ``scipy.optimize.Bounds`` object or an array-like of shape
            (n_dim, 2) with rows ``[lower_i, upper_i]``.
        args: Extra positional arguments forwarded verbatim to ``func``.
        maxiter: Maximum number of evolutionary iterations. Must be >= 1.
        popsize: Population size multiplier; actual population is
            ``max(20, popsize * n_dim)``.
        tol: Relative convergence tolerance on the standard deviation of
            population energies.  Iteration stops when
            ``std(energies) <= atol + tol * |mean(energies)|``.
        mutation: Accepted for SciPy compatibility; not used by CFSSDA.
        recombination: Accepted for SciPy compatibility; not used by CFSSDA.
        seed: Integer seed or ``np.random.Generator`` for reproducibility.
            ``None`` uses a non-deterministic source.
        callback: Optional callable ``callback(xk, convergence) -> bool``.
            Receives the current best solution and convergence metric each
            iteration.  Return ``True`` to terminate early.
        disp: If ``True``, print per-iteration progress to stdout.
        polish: If ``True``, apply a local gradient-based refinement
            (L-BFGS-B or SLSQP) from the best solution found.
        init: Either ``"random"`` for uniform random initialisation, or a
            custom 2-D array of shape (n_pop, n_dim) supplying the initial
            population.
        atol: Absolute tolerance added to the convergence threshold.
        updating: Accepted for SciPy compatibility; not used by CFSSDA.
        workers: Accepted for SciPy compatibility; not used by CFSSDA.
        constraints: One or more constraints in any of the following forms:
            ``LinearConstraint``, ``NonlinearConstraint``, or an SLSQP-style
            dict with keys ``"type"`` and ``"fun"``.  Violated constraints are
            penalised with a growing exterior-penalty term.
        x0: Optional initial guess inserted at ``population[0]``.
            Shape: (n_dim,).  Clipped to bounds if out of range.
        integrality: Accepted for SciPy compatibility; not enforced.
        vectorized: If ``True``, ``func`` is called with the full population
            matrix of shape (n_pop, n_dim) and must return shape (n_pop,) or
            (n_pop, n_obj).
        multi_objective: Set ``True`` when ``func`` returns a vector of
            objectives.  The vector is scalarized according to
            ``scalarization`` before ranking.
        objective_weights: Non-negative weight vector for scalarization.
            Shape: (n_obj,).  Defaults to uniform weights when ``None``.
        scalarization: Scalarization strategy for multi-objective mode.
            ``"weighted_sum"`` computes ``weights @ f(x)``; ``"tchebycheff"``
            computes ``max(weights * |f(x) - ideal|)``.
        return_pareto: If ``True`` and ``multi_objective=True``, attach the
            Pareto archive (``result.pareto_f``, ``result.pareto_x``) to the
            returned result object.
        penalty_start: Initial penalty multiplier for constraint violations.
            Must be > 0.
        penalty_growth: Multiplicative growth factor applied to the penalty
            each iteration.  Values > 1.0 increase constraint enforcement
            over time.
        c_max: Maximum adaptive behavior-weight coefficient (start of run).
        c_min: Minimum adaptive behavior-weight coefficient (end of run).
        inertia_start: Initial inertia weight ``w`` in the step equation.
        inertia_end: Final inertia weight ``w`` at the last iteration.
        neighbor_radius_start: Initial neighbourhood radius.  When ``None``,
            defaults to 25 % of the diagonal span of the search space.
        neighbor_radius_end: Final neighbourhood radius (typically 0.0 to
            encourage convergence).
        coulomb_alpha_mean: Mean of the Gaussian random variable for the
            exponential decay exponent ``α̂`` in ``k(t) = k0·exp(−α̂·t/T)``.
        coulomb_alpha_std: Standard deviation of the Gaussian for ``α̂``.
        k0: Initial Coulomb search coefficient ``k(t0)``.
        levy_beta: Stability index ``β`` for the Levy flight step.
            Must be in (0, 2].

    Returns:
        A ``scipy.optimize.OptimizeResult`` with the following fields set:

        * **x** *(ndarray, shape (n_dim,))*: Best solution found.
        * **fun** *(float)*: Scalarized objective value at ``x``.
        * **success** *(bool)*: ``True`` if convergence criterion was met.
        * **message** *(str)*: Human-readable termination reason.
        * **nit** *(int)*: Number of iterations completed.
        * **nfev** *(int)*: Total number of objective evaluations.
        * **population** *(ndarray, shape (n_pop, n_dim))*: Final population.
        * **population_energies** *(ndarray, shape (n_pop,))*: Final penalised
          energies for each individual.
        * **constraint_violation** *(float)*: Constraint violation at ``x``.
        * **objective_vector** *(ndarray)*: Raw objective vector at ``x``.
        * **penalized_fun** *(float)*: Penalised energy at ``x``.
        * **optimizer** *(str)*: Always ``"CFSSDA"``.
        * **fun_vector** *(ndarray)*: Present when ``multi_objective=True``.
        * **pareto_f** *(ndarray)*: Present when ``return_pareto=True``.
        * **pareto_x** *(ndarray)*: Present when ``return_pareto=True``.

    Raises:
        ValueError: If ``maxiter`` < 1, population size < 2, ``init`` string
            is not ``"random"``, custom ``init`` has wrong shape, ``x0`` has
            wrong dimension, ``scalarization`` is not a recognised strategy,
            or ``objective_weights`` is incompatible with the objective.
        TypeError: If a constraint object has an unsupported type.

    Complexity:
        Time:  O(maxiter * n_pop^2 * n_dim)  (dominated by neighbour search
               and kbest Coulomb force accumulation)
        Space: O(n_pop * n_dim)
    """
    # Discard compatibility-only parameters.
    del mutation, recombination, updating, workers, integrality

    # ------------------------------------------------------------------
    # 1. Initialisation – bounds, population, step vectors
    # ------------------------------------------------------------------
    lower, upper = _parse_bounds(bounds)
    n_dim = lower.size
    n_pop = max(20, int(popsize) * n_dim)

    if maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {maxiter}")
    if n_pop < 2:
        raise ValueError(f"population size must be >= 2, got {n_pop}")
    if scalarization not in {"weighted_sum", "tchebycheff"}:
        raise ValueError(
            f"scalarization must be 'weighted_sum' or 'tchebycheff', "
            f"got '{scalarization}'"
        )

    rng: np.random.Generator = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )

    # Normalise constraints to a sequence.
    constraints_seq: Sequence[Any]
    if constraints is None:
        constraints_seq = ()
    elif isinstance(constraints, (list, tuple)):
        constraints_seq = constraints
    else:
        constraints_seq = (constraints,)

    span = upper - lower

    # Auto-set neighbourhood radius to 25 % of the search-space diagonal.
    if neighbor_radius_start is None:
        neighbor_radius_start = 0.25 * float(np.linalg.norm(span))
    neighbor_radius_start = max(neighbor_radius_start, EPS)
    neighbor_radius_end = max(neighbor_radius_end, 0.0)

    # Initialise population positions.
    if isinstance(init, str):
        if init.lower() != "random":
            raise ValueError(
                f"init must be 'random' or a custom array, got '{init}'"
            )
        population = rng.uniform(lower, upper, size=(n_pop, n_dim))
    else:
        population = np.asarray(init, dtype=float)
        if population.ndim != 2 or population.shape[1] != n_dim:
            raise ValueError(
                f"custom init must have shape (n_pop, {n_dim}), "
                f"got {population.shape}"
            )
        n_pop = population.shape[0]

    # Optionally seed position 0 with a user-supplied initial guess.
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x0_arr.size != n_dim:
            raise ValueError(
                f"x0 must have {n_dim} elements, got {x0_arr.size}"
            )
        population[0] = np.clip(x0_arr, lower, upper)

    # Initialise step vectors (velocity analogues).
    delta_x = rng.uniform(-0.1, 0.1, size=(n_pop, n_dim)) * span

    # ------------------------------------------------------------------
    # 2. Objective evaluation helpers
    # ------------------------------------------------------------------
    objective_vectors: List[np.ndarray] = []   # Pareto archive (objectives).
    objective_points: List[np.ndarray] = []    # Pareto archive (positions).
    nfev = 0

    def _evaluate_single(x: np.ndarray) -> _ObjectiveValue:
        """Evaluate ``func`` at ``x`` and scalarise the result."""
        nonlocal nfev
        raw = func(x, *args)
        nfev += 1
        vec = np.atleast_1d(np.asarray(raw, dtype=float)).reshape(-1)

        if not multi_objective:
            if vec.size != 1:
                raise ValueError(
                    "objective returned a vector; set multi_objective=True"
                )
            scalar = float(vec[0])
        else:
            if vec.size < 2:
                raise ValueError(
                    "multi_objective=True requires objective vector length >= 2"
                )
            if objective_weights is None:
                w = np.full(vec.size, 1.0 / vec.size)
            else:
                w = np.asarray(objective_weights, dtype=float).reshape(-1)
                if w.size != vec.size:
                    raise ValueError(
                        "objective_weights length must match number of objectives"
                    )
                if np.any(w < 0):
                    raise ValueError("objective_weights must be non-negative")
                w_sum = np.sum(w)
                if w_sum <= 0:
                    raise ValueError("objective_weights sum must be positive")
                w = w / w_sum

            if scalarization == "weighted_sum":
                scalar = float(np.dot(w, vec))
            else:
                obj_matrix = np.asarray(objective_vectors)
                ideal = np.min(obj_matrix, axis=0) if obj_matrix.size else vec
                scalar = float(np.max(w * np.abs(vec - ideal)))

            if return_pareto:
                objective_vectors.append(vec.copy())
                objective_points.append(x.copy())

        return _ObjectiveValue(scalar=scalar, vector=vec)

    # ------------------------------------------------------------------
    # 3. Initial population evaluation
    # ------------------------------------------------------------------
    if vectorized:
        objective_values = np.asarray(func(population, *args), dtype=float)
        if objective_values.ndim == 1:
            objective_values = objective_values[:, None]
        if objective_values.shape[0] != n_pop:
            raise ValueError(
                "vectorized objective must return shape (n_pop,) or (n_pop, n_obj)"
            )
        nfev += n_pop

        if not multi_objective and objective_values.shape[1] != 1:
            raise ValueError(
                "objective returned multiple values; set multi_objective=True"
            )
        if multi_objective and objective_values.shape[1] < 2:
            raise ValueError(
                "multi_objective=True requires objective vector length >= 2"
            )

        if multi_objective:
            if objective_weights is None:
                weights = np.full(
                    objective_values.shape[1], 1.0 / objective_values.shape[1]
                )
            else:
                weights = np.asarray(objective_weights, dtype=float).reshape(-1)
                if weights.size != objective_values.shape[1]:
                    raise ValueError(
                        "objective_weights length must match number of objectives"
                    )
                weights = weights / (np.sum(weights) + EPS)

            if scalarization == "weighted_sum":
                objective_scalars = objective_values @ weights
            else:
                ideal = np.min(objective_values, axis=0)
                objective_scalars = np.max(
                    weights[None, :] * np.abs(objective_values - ideal), axis=1
                )
            if return_pareto:
                objective_vectors.extend([row.copy() for row in objective_values])
                objective_points.extend([row.copy() for row in population])
        else:
            objective_scalars = objective_values[:, 0]

        objective_vectors_arr = objective_values

    else:
        init_vals = [_evaluate_single(population[i]) for i in range(n_pop)]
        objective_scalars = np.asarray([v.scalar for v in init_vals], dtype=float)
        objective_vectors_arr = np.vstack([v.vector for v in init_vals])

    # ------------------------------------------------------------------
    # 4. Penalty-augmented energy initialisation
    # ------------------------------------------------------------------
    penalties = np.array(
        [
            _constraint_violation(population[i], constraints_seq, args)
            for i in range(n_pop)
        ],
        dtype=float,
    )
    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * penalties

    # Track global best.
    best_idx = int(np.argmin(energies))
    best_x = population[best_idx].copy()
    best_energy = float(energies[best_idx])
    best_obj_vec = objective_vectors_arr[best_idx].copy()

    success = False
    message = "Maximum number of iterations reached."

    # ------------------------------------------------------------------
    # 5. Main evolutionary loop
    # ------------------------------------------------------------------
    for it in range(1, maxiter + 1):
        progress = (it - 1) / max(maxiter - 1, 1)

        # Linearly decay inertia, behavior coefficient, and neighbourhood radius.
        inertia = inertia_start + (inertia_end - inertia_start) * progress
        behavior_base = c_max + (c_min - c_max) * progress
        neighborhood_radius = (
            neighbor_radius_start
            + (neighbor_radius_end - neighbor_radius_start) * progress
        )

        # ----- Coulomb mass computation (Eq. 3.22–3.25) -----
        curr_best = float(np.min(energies))
        curr_worst = float(np.max(energies))
        mass_raw = (energies - curr_worst) / (curr_best - curr_worst + EPS)
        mass_raw = np.maximum(mass_raw, EPS)
        mass = mass_raw / (np.sum(mass_raw) + EPS)
        gamma_w = c_min + (c_max - c_min) * mass_raw   # Adaptive weight γ_i.
        fit_g = gamma_w * mass                          # Modified mass fitg_i.

        # kbest: indices of top-k individuals, linearly reduced N → 1.
        order = np.argsort(mass)[::-1]
        kbest_count = max(1, int(np.ceil(n_pop - (n_pop - 1) * progress)))
        kbest = order[:kbest_count]

        # Food (best) and enemy (worst) positions.
        food_idx = int(np.argmin(energies))
        enemy_idx = int(np.argmax(energies))
        food_pos = population[food_idx]
        enemy_pos = population[enemy_idx]

        # Stochastic Coulomb decay coefficient k(t) (Eq. 3.28).
        alpha_hat = abs(rng.normal(coulomb_alpha_mean, coulomb_alpha_std))
        k_t = k0 * np.exp(-alpha_hat * it / maxiter)

        new_population = population.copy()
        new_delta_x = delta_x.copy()
        max_step = 0.2 * span

        # ----- Per-individual position update -----
        for i in range(n_pop):
            distances = np.linalg.norm(population - population[i], axis=1)
            neighbors = np.where(
                (distances > 0.0) & (distances <= neighborhood_radius)
            )[0]

            # No neighbours: update via Levy flight (Eq. 3.8).
            if neighbors.size == 0:
                levy = _levy_flight(rng, n_dim, beta=levy_beta)
                levy_step = levy * population[i]
                new_delta_x[i] = levy_step
                new_population[i] = population[i] + levy_step
                continue

            # Five DA behavioural components (Eq. 3.1–3.5).
            separation = -np.sum(population[i] - population[neighbors], axis=0)
            alignment = np.mean(delta_x[neighbors], axis=0)
            cohesion = np.mean(population[neighbors], axis=0) - population[i]
            food_attr = food_pos - population[i]
            enemy_avoid = enemy_pos + population[i]

            # Coulomb attractive force from kbest peers (Eq. 3.26, 3.30).
            total_force = np.zeros(n_dim, dtype=float)
            for j in kbest:
                if j == i:
                    continue
                diff = population[j] - population[i]
                dist = np.linalg.norm(diff) + EPS
                total_force += (
                    rng.uniform() * k_t * mass[i] * mass[j] * (diff / dist)
                )

            # Coulomb repulsive force from enemy (Eq. 3.27, negated k).
            enemy_diff = enemy_pos - population[i]
            enemy_dist = np.linalg.norm(enemy_diff) + EPS
            total_force += (
                rng.uniform()
                * (-k_t)
                * mass[i]
                * mass[enemy_idx]
                * (enemy_diff / enemy_dist)
            )

            # Acceleration from Newton's second law (Eq. 3.31).
            acceleration = total_force / (fit_g[i] + EPS)

            # Randomised behavior weights.
            s_w = behavior_base * rng.uniform()
            a_w = behavior_base * rng.uniform()
            c_w = behavior_base * rng.uniform()
            f_w = 2.0 * rng.uniform()
            e_w = behavior_base * rng.uniform()

            # Step update (Eq. 3.32).
            candidate_delta = (
                inertia * delta_x[i]
                + s_w * separation
                + a_w * alignment
                + c_w * cohesion
                + f_w * food_attr
                + e_w * enemy_avoid
                + acceleration
            )
            candidate_delta = np.clip(candidate_delta, -max_step, max_step)
            new_delta_x[i] = candidate_delta
            new_population[i] = population[i] + candidate_delta

        # Enforce box constraints via reflection.
        new_population = _reflect_bounds(new_population, lower, upper)

        # ----- Evaluate new candidates -----
        if vectorized:
            new_obj_values = np.asarray(func(new_population, *args), dtype=float)
            if new_obj_values.ndim == 1:
                new_obj_values = new_obj_values[:, None]
            nfev += n_pop

            if multi_objective:
                if objective_weights is None:
                    weights = np.full(
                        new_obj_values.shape[1], 1.0 / new_obj_values.shape[1]
                    )
                else:
                    weights = np.asarray(objective_weights, dtype=float).reshape(-1)
                    weights = weights / (np.sum(weights) + EPS)

                if scalarization == "weighted_sum":
                    new_obj_scalars = new_obj_values @ weights
                else:
                    ideal = np.minimum(
                        np.min(objective_vectors_arr, axis=0),
                        np.min(new_obj_values, axis=0),
                    )
                    new_obj_scalars = np.max(
                        weights[None, :] * np.abs(new_obj_values - ideal), axis=1
                    )
                if return_pareto:
                    objective_vectors.extend(
                        [row.copy() for row in new_obj_values]
                    )
                    objective_points.extend(
                        [row.copy() for row in new_population]
                    )
            else:
                new_obj_scalars = new_obj_values[:, 0]

            new_obj_vectors = new_obj_values

        else:
            new_vals = [_evaluate_single(new_population[i]) for i in range(n_pop)]
            new_obj_scalars = np.asarray([v.scalar for v in new_vals], dtype=float)
            new_obj_vectors = np.vstack([v.vector for v in new_vals])

        # ----- Greedy selection with growing penalty -----
        new_penalties = np.array(
            [
                _constraint_violation(new_population[i], constraints_seq, args)
                for i in range(n_pop)
            ],
            dtype=float,
        )
        penalty_factor *= penalty_growth
        new_energies = new_obj_scalars + penalty_factor * new_penalties

        improved = new_energies < energies
        population[improved] = new_population[improved]
        delta_x[improved] = new_delta_x[improved]
        objective_scalars[improved] = new_obj_scalars[improved]
        objective_vectors_arr[improved] = new_obj_vectors[improved]
        penalties[improved] = new_penalties[improved]
        energies[improved] = new_energies[improved]

        # Update global best.
        curr_idx = int(np.argmin(energies))
        if energies[curr_idx] < best_energy:
            best_idx = curr_idx
            best_x = population[curr_idx].copy()
            best_energy = float(energies[curr_idx])
            best_obj_vec = objective_vectors_arr[curr_idx].copy()

        # ----- Convergence check -----
        convergence = float(np.std(energies))
        threshold = float(atol + tol * abs(np.mean(energies)))

        if disp:
            print(
                f"[CFSSDA] iter={it:4d}  best={best_energy:.8e}  "
                f"mean={np.mean(energies):.8e}  std={convergence:.8e}"
            )

        if callback is not None and callback(best_x.copy(), convergence):
            success = False
            message = "Stopped by callback."
            break

        if convergence <= threshold:
            success = True
            message = "Optimization converged."
            break

    # ------------------------------------------------------------------
    # 6. Optional local polishing step
    # ------------------------------------------------------------------
    if polish:
        def _local_objective(x_local: np.ndarray) -> float:
            """Penalised scalarised objective for local refinement."""
            obj = np.atleast_1d(np.asarray(func(x_local, *args), dtype=float))
            if obj.size == 1:
                obj_scalar = float(obj[0])
            else:
                if objective_weights is None:
                    w_local = np.full(obj.size, 1.0 / obj.size)
                else:
                    w_local = np.asarray(objective_weights, dtype=float).reshape(-1)
                    w_local = w_local / (np.sum(w_local) + EPS)
                if scalarization == "weighted_sum":
                    obj_scalar = float(np.dot(w_local, obj))
                else:
                    obj_scalar = float(
                        np.max(w_local * np.abs(obj - best_obj_vec))
                    )
            pen = _constraint_violation(x_local, constraints_seq, args)
            return obj_scalar + penalty_factor * pen

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
            best_energy = float(local_res.fun)
            best_obj_vec = np.atleast_1d(
                np.asarray(func(best_x, *args), dtype=float)
            )
            nfev += 1

    # ------------------------------------------------------------------
    # 7. Assemble and return result
    # ------------------------------------------------------------------
    if objective_weights is not None:
        w_final = np.asarray(objective_weights, dtype=float)
        w_final = w_final / (np.sum(w_final) + EPS)
    else:
        w_final = np.full(best_obj_vec.size, 1.0 / best_obj_vec.size)

    result = OptimizeResult()
    result.x = best_x
    result.fun = (
        float(best_obj_vec[0])
        if best_obj_vec.size == 1
        else float(np.dot(w_final, best_obj_vec))
    )
    result.success = bool(success)
    result.message = message
    result.nit = int(it if "it" in locals() else 0)
    result.nfev = int(nfev)
    result.population = population.copy()
    result.population_energies = energies.copy()
    result.constraint_violation = float(
        _constraint_violation(best_x, constraints_seq, args)
    )
    result.objective_vector = best_obj_vec.copy()
    result.penalized_fun = float(best_energy)
    result.optimizer = "CFSSDA"

    if multi_objective:
        result.fun_vector = best_obj_vec.copy()
        if return_pareto and len(objective_vectors) > 0:
            archive_f = np.asarray(objective_vectors, dtype=float)
            archive_x = np.asarray(objective_points, dtype=float)
            nd_idx = _nondominated_indices(archive_f)
            result.pareto_f = archive_f[nd_idx]
            result.pareto_x = archive_x[nd_idx]

    return result
