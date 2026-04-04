# Shared helpers for optimization solvers
# Author: Shengning Wang

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint


def _parse_bounds(bounds: Union[Bounds, Sequence[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert box bounds to lower and upper vectors.

    Args:
        bounds (Union[Bounds, Sequence[Tuple[float, float]]]): Variable bounds.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Lower and upper bounds with shape (num_vars,) and dtype float64.
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

    Args:
        constraints (Union[Sequence[Any], Any]): Constraint specification.

    Returns:
        Tuple[Any, ...]: Normalized constraints.
    """
    if constraints is None:
        return ()
    if isinstance(constraints, (list, tuple)):
        return tuple(constraints)
    return (constraints,)


def _repair_to_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Reflect points back into the search box.

    Args:
        x (np.ndarray): Candidate points with shape (..., num_vars) and dtype float64.
        lower (np.ndarray): Lower bounds with shape (num_vars,) and dtype float64.
        upper (np.ndarray): Upper bounds with shape (num_vars,) and dtype float64.

    Returns:
        np.ndarray: Repaired points with shape (..., num_vars) and dtype float64.
    """
    x = np.where(x < lower, 2.0 * lower - x, x)
    x = np.where(x > upper, 2.0 * upper - x, x)
    return np.clip(x, lower, upper)


def _constraint_violation(x: np.ndarray, constraints: Sequence[Any], args: Tuple[Any, ...]) -> float:
    """
    Calculate the total constraint violation at one point.

    Args:
        x (np.ndarray): Candidate point with shape (num_vars,) and dtype float64.
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


def _evaluate_constraint_violations(
    population: np.ndarray,
    constraints: Sequence[Any],
    args: Tuple[Any, ...],
) -> np.ndarray:
    """
    Evaluate total constraint violations for a population.

    Args:
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        constraints (Sequence[Any]): Constraint objects.
        args (Tuple[Any, ...]): Extra objective arguments.

    Returns:
        np.ndarray: Constraint violations with shape (num_points,) and dtype float64.
    """
    return np.array(
        [_constraint_violation(population[i], constraints, args) for i in range(population.shape[0])],
        dtype=np.float64,
    )


def _normalize_weights(num_objectives: int, objective_weights: Optional[np.ndarray]) -> np.ndarray:
    """
    Normalize multi-objective weights.

    Args:
        num_objectives (int): Number of objectives.
        objective_weights (Optional[np.ndarray]): Raw objective weights.

    Returns:
        np.ndarray: Normalized weights with shape (num_objectives,) and dtype float64.
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

    Args:
        y (np.ndarray): Objective matrix with shape (num_points, num_objectives) and dtype float64.

    Returns:
        np.ndarray: Indices of non-dominated points with shape (num_pareto,) and dtype int64.
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


def _scalarize_objectives(
    objective_vectors: np.ndarray,
    multi_objective: bool,
    weights: Optional[np.ndarray],
    scalarization: str,
    reference_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert objective vectors to scalar objective values.

    Args:
        objective_vectors (np.ndarray): Objective matrix with shape (num_points, num_objectives) and dtype float64.
        multi_objective (bool): Whether the objective is multi-objective.
        weights (Optional[np.ndarray]): Objective weights with shape (num_objectives,) and dtype float64.
        scalarization (str): Scalarization method.
        reference_values (Optional[np.ndarray]): Reference objective values for Tchebycheff scalarization.

    Returns:
        np.ndarray: Scalar objective values with shape (num_points,) and dtype float64.
    """
    if not multi_objective:
        if objective_vectors.shape[1] != 1:
            raise ValueError("Objective returned multiple values; set multi_objective=True.")
        return objective_vectors[:, 0]

    if objective_vectors.shape[1] < 2:
        raise ValueError("Multi-objective mode requires at least two objectives.")

    if weights is None:
        weights = np.ones(objective_vectors.shape[1], dtype=np.float64) / objective_vectors.shape[1]

    if scalarization == "weighted_sum":
        return objective_vectors @ weights

    if scalarization == "tchebycheff":
        if reference_values is None or reference_values.size == 0:
            ideal = np.min(objective_vectors, axis=0)
        else:
            ideal = np.min(reference_values, axis=0)
        return np.max(weights[None, :] * np.abs(objective_vectors - ideal), axis=1)

    raise ValueError("Unsupported scalarization method.")


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
    Evaluate a population and return objective vectors and scalar objective values.

    Args:
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        func (Callable): Objective function.
        args (Tuple[Any, ...]): Extra objective arguments.
        multi_objective (bool): Whether the objective is multi-objective.
        weights (Optional[np.ndarray]): Objective weights with shape (num_objectives,) and dtype float64.
        scalarization (str): Scalarization method.
        reference_values (Optional[np.ndarray]): Reference objective values for Tchebycheff scalarization.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Objective vectors with shape (num_points, num_objectives) and scalar objectives
            with shape (num_points,), both in dtype float64.
    """
    objective_vectors = []
    for i in range(population.shape[0]):
        values = np.atleast_1d(np.asarray(func(population[i], *args), dtype=np.float64)).reshape(-1)
        objective_vectors.append(values)

    objective_vectors = np.vstack(objective_vectors)
    objective_scalars = _scalarize_objectives(
        objective_vectors=objective_vectors,
        multi_objective=multi_objective,
        weights=weights,
        scalarization=scalarization,
        reference_values=reference_values,
    )
    return objective_vectors, objective_scalars


def _initialize_objective_values(
    population: np.ndarray,
    func: Callable,
    args: Tuple[Any, ...],
    multi_objective: bool,
    objective_weights: Optional[np.ndarray],
    scalarization: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate the initial population and prepare scalarization weights.

    Args:
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        func (Callable): Objective function.
        args (Tuple[Any, ...]): Extra objective arguments.
        multi_objective (bool): Whether the objective is multi-objective.
        objective_weights (Optional[np.ndarray]): Raw objective weights.
        scalarization (str): Scalarization method.

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            Objective vectors, scalar objectives, and normalized weights.
    """
    objective_vectors, objective_scalars = _evaluate_population(
        population=population,
        func=func,
        args=args,
        multi_objective=multi_objective,
        weights=None,
        scalarization=scalarization,
    )

    weights = None
    if multi_objective:
        weights = _normalize_weights(objective_vectors.shape[1], objective_weights)
        objective_scalars = _scalarize_objectives(
            objective_vectors=objective_vectors,
            multi_objective=True,
            weights=weights,
            scalarization=scalarization,
            reference_values=objective_vectors,
        )

    return objective_vectors, objective_scalars, weights


def _make_rng(seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    """
    Build a NumPy random number generator from an integer seed or an existing generator.

    Args:
        seed (Optional[Union[int, np.random.Generator]]): Seed or generator.

    Returns:
        np.random.Generator: Random number generator.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _apply_initial_guess(
    population: np.ndarray,
    x0: Optional[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """
    Place an initial point into the first population slot.

    Args:
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        x0 (Optional[np.ndarray]): Initial point with shape (num_vars,) and dtype float64.
        lower (np.ndarray): Lower bounds with shape (num_vars,) and dtype float64.
        upper (np.ndarray): Upper bounds with shape (num_vars,) and dtype float64.
    """
    if x0 is None:
        return

    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0.size != lower.size:
        raise ValueError("x0 dimension does not match bounds.")
    population[0] = np.clip(x0, lower, upper)


def _append_archive(
    archive_x: List[np.ndarray],
    archive_f: List[np.ndarray],
    archive_v: List[float],
    population: np.ndarray,
    objective_vectors: np.ndarray,
    violations: np.ndarray,
) -> None:
    """
    Append one population snapshot to the Pareto archive buffers.

    Args:
        archive_x (List[np.ndarray]): Candidate archive.
        archive_f (List[np.ndarray]): Objective archive.
        archive_v (List[float]): Constraint-violation archive.
        population (np.ndarray): Population with shape (num_points, num_vars) and dtype float64.
        objective_vectors (np.ndarray): Objective matrix with shape (num_points, num_objectives) and dtype float64.
        violations (np.ndarray): Constraint violations with shape (num_points,) and dtype float64.
    """
    archive_x.extend([row.copy() for row in population])
    archive_f.extend([row.copy() for row in objective_vectors])
    archive_v.extend([float(value) for value in violations])


def _finalize_pareto_archive(
    archive_x: List[np.ndarray],
    archive_f: List[np.ndarray],
    archive_v: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert archive buffers to final Pareto points and objective values.

    Args:
        archive_x (List[np.ndarray]): Candidate archive.
        archive_f (List[np.ndarray]): Objective archive.
        archive_v (List[float]): Constraint-violation archive.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Pareto points with shape (num_pareto, num_vars) and objective values
            with shape (num_pareto, num_objectives), both in dtype float64.
    """
    archive_y = np.asarray(archive_f, dtype=np.float64)
    archive_points = np.asarray(archive_x, dtype=np.float64)
    archive_violation = np.asarray(archive_v, dtype=np.float64)

    feasible_mask = archive_violation <= 1e-8
    if np.any(feasible_mask):
        archive_y = archive_y[feasible_mask]
        archive_points = archive_points[feasible_mask]
    else:
        best_violation = np.min(archive_violation)
        near_mask = archive_violation <= best_violation + 1e-8
        archive_y = archive_y[near_mask]
        archive_points = archive_points[near_mask]

    nondominated = _nondominated_indices(archive_y)
    return archive_points[nondominated], archive_y[nondominated]
