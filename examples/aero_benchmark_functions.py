"""Benchmark function registry for the aero contract test suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np


ArrayFn = Callable[[np.ndarray], np.ndarray]


def _as_2d_array(x: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    """Validate and reshape user input into a 2-D NumPy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} expects a 2-D array, got shape {arr.shape}.")
    if arr.shape[1] != expected_dim:
        raise ValueError(
            f"{name} expects input_dim={expected_dim}, got {arr.shape[1]}."
        )
    return arr


@dataclass(frozen=True)
class ScalarBenchmark:
    """Container for a scalar-output benchmark function.

    Args:
        name: Canonical benchmark name.
        input_dim: Number of design variables.
        bounds: Input bounds with shape (input_dim, 2).
        output_name: Name of the scalar response.
        description: Short human-readable description.
        evaluator: Vectorized callable that maps ``(N, D)`` to ``(N, 1)``.
        known_optimum: Optional known global minimum value.
        known_minimizers: Optional tuple of known global minimizers.

    Returns:
        ScalarBenchmark: Immutable benchmark specification.

    Raises:
        ValueError: Propagated by the stored evaluator when the input is invalid.

    Shapes:
        ``(N, D) -> (N, 1)``

    Complexity:
        Construction is ``O(1)`` time and ``O(1)`` space.
    """

    name: str
    input_dim: int
    bounds: Tuple[Tuple[float, float], ...]
    output_name: str
    description: str
    evaluator: ArrayFn
    known_optimum: Optional[float] = None
    known_minimizers: Tuple[Tuple[float, ...], ...] = field(default_factory=tuple)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the scalar benchmark.

        Args:
            x: Query points with shape ``(num_samples, input_dim)`` or ``(input_dim,)``.

        Returns:
            np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

        Raises:
            ValueError: If the input array does not match ``input_dim``.

        Shapes:
            ``(N, D) -> (N, 1)``

        Complexity:
            Time is benchmark-dependent; memory is ``O(N)``.
        """

        return self.evaluator(x)

    @property
    def bounds_array(self) -> np.ndarray:
        """Return the bounds as a float64 array."""
        return np.asarray(self.bounds, dtype=np.float64)


@dataclass(frozen=True)
class MultiFidelityBenchmark:
    """Container for a paired low-/high-fidelity benchmark.

    Args:
        name: Canonical benchmark name.
        input_dim: Number of design variables.
        bounds: Input bounds with shape (input_dim, 2).
        output_name: Name of the scalar response.
        description: Short human-readable description.
        high_fidelity: High-fidelity evaluator with shape ``(N, D) -> (N, 1)``.
        low_fidelity: Low-fidelity evaluator with shape ``(N, D) -> (N, 1)``.

    Returns:
        MultiFidelityBenchmark: Immutable paired-function specification.

    Raises:
        ValueError: Propagated by the stored evaluators when the input is invalid.

    Shapes:
        ``(N, D) -> (N, 1)``

    Complexity:
        Construction is ``O(1)`` time and ``O(1)`` space.
    """

    name: str
    input_dim: int
    bounds: Tuple[Tuple[float, float], ...]
    output_name: str
    description: str
    high_fidelity: ArrayFn
    low_fidelity: ArrayFn

    def evaluate_high_fidelity(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the high-fidelity response.

        Args:
            x: Query points with shape ``(num_samples, input_dim)`` or ``(input_dim,)``.

        Returns:
            np.ndarray: High-fidelity responses with shape ``(num_samples, 1)``.

        Raises:
            ValueError: If the input array does not match ``input_dim``.

        Shapes:
            ``(N, D) -> (N, 1)``

        Complexity:
            Time is benchmark-dependent; memory is ``O(N)``.
        """

        return self.high_fidelity(x)

    def evaluate_low_fidelity(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the low-fidelity response.

        Args:
            x: Query points with shape ``(num_samples, input_dim)`` or ``(input_dim,)``.

        Returns:
            np.ndarray: Low-fidelity responses with shape ``(num_samples, 1)``.

        Raises:
            ValueError: If the input array does not match ``input_dim``.

        Shapes:
            ``(N, D) -> (N, 1)``

        Complexity:
            Time is benchmark-dependent; memory is ``O(N)``.
        """

        return self.low_fidelity(x)

    @property
    def bounds_array(self) -> np.ndarray:
        """Return the bounds as a float64 array."""
        return np.asarray(self.bounds, dtype=np.float64)


@dataclass(frozen=True)
class MultiObjectiveBenchmark:
    """Container for a multi-objective benchmark.

    Args:
        name: Canonical benchmark name.
        input_dim: Number of design variables.
        bounds: Input bounds with shape (input_dim, 2).
        output_names: Names of the objectives.
        description: Short human-readable description.
        evaluator: Vectorized callable that maps ``(N, D)`` to ``(N, M)``.

    Returns:
        MultiObjectiveBenchmark: Immutable multi-objective specification.

    Raises:
        ValueError: Propagated by the stored evaluator when the input is invalid.

    Shapes:
        ``(N, D) -> (N, M)``

    Complexity:
        Construction is ``O(1)`` time and ``O(1)`` space.
    """

    name: str
    input_dim: int
    bounds: Tuple[Tuple[float, float], ...]
    output_names: Tuple[str, ...]
    description: str
    evaluator: ArrayFn

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the multi-objective benchmark.

        Args:
            x: Query points with shape ``(num_samples, input_dim)`` or ``(input_dim,)``.

        Returns:
            np.ndarray: Objective values with shape ``(num_samples, num_objectives)``.

        Raises:
            ValueError: If the input array does not match ``input_dim``.

        Shapes:
            ``(N, D) -> (N, M)``

        Complexity:
            Time is benchmark-dependent; memory is ``O(N * M)``.
        """

        return self.evaluator(x)

    @property
    def bounds_array(self) -> np.ndarray:
        """Return the bounds as a float64 array."""
        return np.asarray(self.bounds, dtype=np.float64)


def forrester(x: np.ndarray) -> np.ndarray:
    """Evaluate the Forrester function.

    Args:
        x: Input array with shape ``(num_samples, 1)`` or ``(1,)`` on ``[0, 1]``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly one feature.

    Shapes:
        ``(N, 1) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=1, name="forrester")
    z = x_arr[:, 0]
    y = (6.0 * z - 2.0) ** 2 * np.sin(12.0 * z - 4.0)
    return y.reshape(-1, 1)


def gramacy_lee(x: np.ndarray) -> np.ndarray:
    """Evaluate the Gramacy-Lee function.

    Args:
        x: Input array with shape ``(num_samples, 1)`` or ``(1,)`` on ``[0.5, 2.5]``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly one feature.

    Shapes:
        ``(N, 1) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=1, name="gramacy_lee")
    z = np.clip(x_arr[:, 0], 0.5, 2.5)
    y = np.sin(10.0 * np.pi * z) / (2.0 * z) + (z - 1.0) ** 4
    return y.reshape(-1, 1)


def branin(x: np.ndarray) -> np.ndarray:
    """Evaluate the Branin-Hoo function.

    Args:
        x: Input array with shape ``(num_samples, 2)`` or ``(2,)`` on
            ``[-5, 10] x [0, 15]``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly two features.

    Shapes:
        ``(N, 2) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=2, name="branin")
    x1 = x_arr[:, 0]
    x2 = x_arr[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s
    return y.reshape(-1, 1)


def branin_low_fidelity(x: np.ndarray) -> np.ndarray:
    """Evaluate the low-fidelity Branin variant used for multi-fidelity tests.

    Args:
        x: Input array with shape ``(num_samples, 2)`` or ``(2,)`` on
            ``[-5, 10] x [0, 15]``.

    Returns:
        np.ndarray: Scalar low-fidelity responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly two features.

    Shapes:
        ``(N, 2) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=2, name="branin_low_fidelity")
    hf = branin(x_arr)[:, 0]
    y = 0.9 * hf + 0.4 * np.sin(x_arr[:, 0]) - 0.2 * x_arr[:, 1] + 2.0
    return y.reshape(-1, 1)


def hartman3(x: np.ndarray) -> np.ndarray:
    """Evaluate the three-dimensional Hartman function.

    Args:
        x: Input array with shape ``(num_samples, 3)`` or ``(3,)`` on ``[0, 1]^3``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly three features.

    Shapes:
        ``(N, 3) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=3, name="hartman3")
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float64)
    a_mat = np.array(
        [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ],
        dtype=np.float64,
    )
    p_mat = 1.0e-4 * np.array(
        [
            [3689.0, 1170.0, 2673.0],
            [4699.0, 4387.0, 7470.0],
            [1091.0, 8732.0, 5547.0],
            [381.0, 5743.0, 8828.0],
        ],
        dtype=np.float64,
    )
    total = np.zeros(x_arr.shape[0], dtype=np.float64)
    for idx in range(4):
        total += alpha[idx] * np.exp(
            -np.sum(a_mat[idx] * (x_arr - p_mat[idx]) ** 2, axis=1)
        )
    return (-total).reshape(-1, 1)


def currin_exponential(x: np.ndarray) -> np.ndarray:
    """Evaluate the Currin exponential function.

    Args:
        x: Input array with shape ``(num_samples, 2)`` or ``(2,)`` on ``[0, 1]^2``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly two features.

    Shapes:
        ``(N, 2) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=2, name="currin_exponential")
    x1 = np.clip(x_arr[:, 0], 1.0e-6, 1.0)
    x2 = np.clip(x_arr[:, 1], 1.0e-6, 1.0)
    numerator = 2300.0 * x1 ** 3 + 1900.0 * x1 ** 2 + 2092.0 * x1 + 60.0
    denominator = 100.0 * x1 ** 3 + 500.0 * x1 ** 2 + 4.0 * x1 + 20.0
    y = (1.0 - np.exp(-1.0 / (2.0 * x2))) * (numerator / denominator)
    return y.reshape(-1, 1)


def currin_exponential_low_fidelity(x: np.ndarray) -> np.ndarray:
    """Evaluate the standard low-fidelity Currin exponential approximation.

    Args:
        x: Input array with shape ``(num_samples, 2)`` or ``(2,)`` on ``[0, 1]^2``.

    Returns:
        np.ndarray: Scalar low-fidelity responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly two features.

    Shapes:
        ``(N, 2) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=2, name="currin_exponential_low_fidelity")
    shifts = np.array(
        [[0.05, 0.05], [0.05, -0.05], [-0.05, 0.05], [-0.05, -0.05]],
        dtype=np.float64,
    )
    values = []
    for shift in shifts:
        shifted = x_arr + shift
        shifted[:, 0] = np.clip(shifted[:, 0], 0.0, 1.0)
        shifted[:, 1] = np.clip(shifted[:, 1], 0.0, 1.0)
        values.append(currin_exponential(shifted)[:, 0])
    y = 0.25 * np.sum(values, axis=0)
    return y.reshape(-1, 1)


def park91b(x: np.ndarray) -> np.ndarray:
    """Evaluate the Park91B high-fidelity function.

    Args:
        x: Input array with shape ``(num_samples, 4)`` or ``(4,)`` on ``[0, 1]^4``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly four features.

    Shapes:
        ``(N, 4) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=4, name="park91b")
    y = (
        (2.0 / 3.0) * np.exp(x_arr[:, 0] + x_arr[:, 1])
        - x_arr[:, 3] * np.sin(x_arr[:, 2])
        + x_arr[:, 2]
    )
    return y.reshape(-1, 1)


def park91b_low_fidelity(x: np.ndarray) -> np.ndarray:
    """Evaluate the Park91B low-fidelity approximation.

    Args:
        x: Input array with shape ``(num_samples, 4)`` or ``(4,)`` on ``[0, 1]^4``.

    Returns:
        np.ndarray: Scalar low-fidelity responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly four features.

    Shapes:
        ``(N, 4) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=4, name="park91b_low_fidelity")
    y = 1.2 * park91b(x_arr)[:, 0] - 1.0
    return y.reshape(-1, 1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """Evaluate the two-dimensional Rastrigin function.

    Args:
        x: Input array with shape ``(num_samples, 2)`` or ``(2,)`` on
            ``[-5.12, 5.12]^2``.

    Returns:
        np.ndarray: Scalar responses with shape ``(num_samples, 1)``.

    Raises:
        ValueError: If the input array does not have exactly two features.

    Shapes:
        ``(N, 2) -> (N, 1)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=2, name="rastrigin")
    y = 20.0 + np.sum(x_arr ** 2 - 10.0 * np.cos(2.0 * np.pi * x_arr), axis=1)
    return y.reshape(-1, 1)


def vlmop2(x: np.ndarray) -> np.ndarray:
    """Evaluate the VLMOP2 two-objective benchmark.

    Args:
        x: Input array with shape ``(num_samples, 2)`` or ``(2,)`` on
            ``[-2, 2]^2``.

    Returns:
        np.ndarray: Objective values with shape ``(num_samples, 2)``.

    Raises:
        ValueError: If the input array does not have exactly two features.

    Shapes:
        ``(N, 2) -> (N, 2)``

    Complexity:
        Time ``O(N)`` and space ``O(N)``.
    """

    x_arr = _as_2d_array(x, expected_dim=2, name="vlmop2")
    shift = 1.0 / np.sqrt(2.0)
    f1 = 1.0 - np.exp(-((x_arr[:, 0] - shift) ** 2 + (x_arr[:, 1] - shift) ** 2))
    f2 = 1.0 - np.exp(-((x_arr[:, 0] + shift) ** 2 + (x_arr[:, 1] + shift) ** 2))
    return np.column_stack([f1, f2])


SCALAR_BENCHMARKS: Dict[str, ScalarBenchmark] = {
    "forrester": ScalarBenchmark(
        name="forrester",
        input_dim=1,
        bounds=((0.0, 1.0),),
        output_name="response",
        description="One-dimensional oscillatory Forrester benchmark.",
        evaluator=forrester,
    ),
    "gramacy_lee": ScalarBenchmark(
        name="gramacy_lee",
        input_dim=1,
        bounds=((0.5, 2.5),),
        output_name="response",
        description="One-dimensional Gramacy-Lee interpolation benchmark.",
        evaluator=gramacy_lee,
    ),
    "branin": ScalarBenchmark(
        name="branin",
        input_dim=2,
        bounds=((-5.0, 10.0), (0.0, 15.0)),
        output_name="response",
        description="Two-dimensional Branin-Hoo benchmark.",
        evaluator=branin,
        known_optimum=0.39788735772973816,
        known_minimizers=(
            (-np.pi, 12.275),
            (np.pi, 2.275),
            (3.0 * np.pi, 2.475),
        ),
    ),
    "hartman3": ScalarBenchmark(
        name="hartman3",
        input_dim=3,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Three-dimensional Hartman benchmark.",
        evaluator=hartman3,
        known_optimum=-3.8627797869493365,
        known_minimizers=((0.114614, 0.555649, 0.852547),),
    ),
    "currin_exponential": ScalarBenchmark(
        name="currin_exponential",
        input_dim=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Two-dimensional Currin exponential benchmark.",
        evaluator=currin_exponential,
    ),
    "park91b": ScalarBenchmark(
        name="park91b",
        input_dim=4,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Four-dimensional Park91B benchmark.",
        evaluator=park91b,
    ),
    "rastrigin": ScalarBenchmark(
        name="rastrigin",
        input_dim=2,
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        output_name="response",
        description="Two-dimensional Rastrigin benchmark.",
        evaluator=rastrigin,
        known_optimum=0.0,
        known_minimizers=((0.0, 0.0),),
    ),
}


MULTI_FIDELITY_BENCHMARKS: Dict[str, MultiFidelityBenchmark] = {
    "currin_exponential": MultiFidelityBenchmark(
        name="currin_exponential",
        input_dim=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Currin exponential high-/low-fidelity benchmark pair.",
        high_fidelity=currin_exponential,
        low_fidelity=currin_exponential_low_fidelity,
    ),
    "branin": MultiFidelityBenchmark(
        name="branin",
        input_dim=2,
        bounds=((-5.0, 10.0), (0.0, 15.0)),
        output_name="response",
        description="Branin benchmark with a biased low-fidelity approximation.",
        high_fidelity=branin,
        low_fidelity=branin_low_fidelity,
    ),
    "park91b": MultiFidelityBenchmark(
        name="park91b",
        input_dim=4,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        output_name="response",
        description="Park91B benchmark with a linear-scaled low-fidelity model.",
        high_fidelity=park91b,
        low_fidelity=park91b_low_fidelity,
    ),
}


MULTI_OBJECTIVE_BENCHMARKS: Dict[str, MultiObjectiveBenchmark] = {
    "vlmop2": MultiObjectiveBenchmark(
        name="vlmop2",
        input_dim=2,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        output_names=("f1", "f2"),
        description="Two-objective VLMOP2 benchmark.",
        evaluator=vlmop2,
    ),
}


def get_scalar_benchmark(name: str) -> ScalarBenchmark:
    """Fetch a scalar benchmark by name.

    Args:
        name: Registry key of the requested benchmark.

    Returns:
        ScalarBenchmark: The requested scalar benchmark specification.

    Raises:
        KeyError: If ``name`` is not in the scalar benchmark registry.

    Shapes:
        Not applicable.

    Complexity:
        Average-case lookup is ``O(1)`` time and ``O(1)`` space.
    """

    key = name.lower()
    if key not in SCALAR_BENCHMARKS:
        raise KeyError(f"Unknown scalar benchmark: '{name}'.")
    return SCALAR_BENCHMARKS[key]


def get_multifidelity_benchmark(name: str) -> MultiFidelityBenchmark:
    """Fetch a multi-fidelity benchmark by name.

    Args:
        name: Registry key of the requested benchmark.

    Returns:
        MultiFidelityBenchmark: The requested paired benchmark specification.

    Raises:
        KeyError: If ``name`` is not in the multi-fidelity benchmark registry.

    Shapes:
        Not applicable.

    Complexity:
        Average-case lookup is ``O(1)`` time and ``O(1)`` space.
    """

    key = name.lower()
    if key not in MULTI_FIDELITY_BENCHMARKS:
        raise KeyError(f"Unknown multi-fidelity benchmark: '{name}'.")
    return MULTI_FIDELITY_BENCHMARKS[key]


def get_multiobjective_benchmark(name: str) -> MultiObjectiveBenchmark:
    """Fetch a multi-objective benchmark by name.

    Args:
        name: Registry key of the requested benchmark.

    Returns:
        MultiObjectiveBenchmark: The requested multi-objective specification.

    Raises:
        KeyError: If ``name`` is not in the multi-objective benchmark registry.

    Shapes:
        Not applicable.

    Complexity:
        Average-case lookup is ``O(1)`` time and ``O(1)`` space.
    """

    key = name.lower()
    if key not in MULTI_OBJECTIVE_BENCHMARKS:
        raise KeyError(f"Unknown multi-objective benchmark: '{name}'.")
    return MULTI_OBJECTIVE_BENCHMARKS[key]
