"""Configuration for the aero contract benchmark runner."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Dict, Iterable, List


ALGORITHM_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

DEMO_ALIAS_MAP = {
    "all": ALGORITHM_ORDER,
    "ensemble": ["A", "B"],
    "multifidelity": ["C", "D", "E"],
    "active_learning": ["F", "G", "H"],
    "optimization": ["I", "J"],
}

DEFAULT_ENSEMBLE_CASES = OrderedDict(
    {
        "forrester": {"num_train": 12, "num_test": 200, "lhs_iterations": 30},
        "branin": {"num_train": 18, "num_test": 200, "lhs_iterations": 30},
        "hartman3": {"num_train": 24, "num_test": 200, "lhs_iterations": 30},
    }
)

DEFAULT_MULTIFIDELITY_CASES = OrderedDict(
    {
        "currin_exponential": {
            "num_lf": 60,
            "num_hf": 24,
            "num_test": 300,
            "lhs_iterations": 20,
        },
        "branin": {"num_lf": 60, "num_hf": 24, "num_test": 300, "lhs_iterations": 20},
        "park91b": {"num_lf": 120, "num_hf": 45, "num_test": 300, "lhs_iterations": 20},
    }
)

DEFAULT_SINGLE_OBJECTIVE_ACTIVE_CASE = {
    "name": "branin",
    "num_initial": 12,
    "num_test": 300,
    "num_infill": 6,
    "criterion": "ei",
    "lhs_iterations": 30,
}

DEFAULT_MULTI_FIDELITY_ACTIVE_CASE = {
    "name": "currin_exponential",
    "num_hf_initial": 8,
    "num_lf": 60,
    "num_test": 300,
    "num_infill": 6,
    "ratio": 0.5,
    "lhs_iterations": 30,
}

DEFAULT_MULTI_OBJECTIVE_ACTIVE_CASE = {
    "name": "vlmop2",
    "num_initial": 6,
    "num_test": 500,
    "num_infill": 8,
    "num_samples": 3000,
    "num_candidates": 120,
    "num_restarts": 4,
    "beta": 0.3,
    "lhs_iterations": 30,
}

DEFAULT_OPTIMIZATION_CASES = OrderedDict(
    {
        "branin": {"num_train": 24, "lhs_iterations": 30},
        "hartman3": {"num_train": 36, "lhs_iterations": 30},
        "rastrigin": {"num_train": 30, "lhs_iterations": 30},
    }
)


def _expand_demo_selection(selection: Iterable[str]) -> List[str]:
    """Expand demo aliases such as ``all`` or ``ensemble`` into ``A-J`` labels."""

    expanded: List[str] = []
    for item in selection:
        key = item.lower()
        if key in DEMO_ALIAS_MAP:
            expanded.extend(DEMO_ALIAS_MAP[key])
        else:
            expanded.append(item.upper())
    ordered = []
    for label in ALGORITHM_ORDER:
        if label in expanded:
            ordered.append(label)
    return ordered


def _expand_case_selection(selection: List[str], defaults: Dict[str, dict]) -> List[str]:
    """Expand ``all`` into the ordered default case list."""

    if len(selection) == 1 and selection[0].lower() == "all":
        return list(defaults.keys())

    normalized = [item.lower() for item in selection]
    unknown = [item for item in normalized if item not in defaults]
    if unknown:
        valid = ", ".join(defaults.keys())
        raise ValueError(f"Unknown case(s): {unknown}. Valid choices: {valid}.")
    return normalized


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the aero contract benchmark runner.

    Returns:
        argparse.Namespace: Parsed configuration with expanded demo labels and
            case selections.
    """

    parser = argparse.ArgumentParser(
        description="Aero contract benchmark runner for the 10 WSNet algorithms."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible Latin hypercube sampling and optimizers.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./benchmark_outputs",
        help="Directory used to store the JSON benchmark report.",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        default=["all"],
        help=(
            "Algorithms to run: A B C D E F G H I J or group aliases "
            "(all, ensemble, multifidelity, active_learning, optimization)."
        ),
    )
    parser.add_argument(
        "--ensemble_cases",
        nargs="+",
        default=["all"],
        help="Ensemble test cases to run, or 'all'.",
    )
    parser.add_argument(
        "--multifidelity_cases",
        nargs="+",
        default=["all"],
        help="Multi-fidelity surrogate test cases to run, or 'all'.",
    )
    parser.add_argument(
        "--optimization_cases",
        nargs="+",
        default=["all"],
        help="Optimization test cases to run, or 'all'.",
    )
    parser.add_argument(
        "--lhs_iterations",
        type=int,
        default=30,
        help="Number of maximin LHS candidate designs evaluated per sample batch.",
    )

    parser.add_argument(
        "--ensemble_threshold",
        type=float,
        default=0.5,
        help="Threshold used by TAHS and AES-MSI for model filtering.",
    )
    parser.add_argument(
        "--ensemble_min_relative_gain",
        type=float,
        default=0.10,
        help="Minimum relative R2 gain over the mean single-model baseline.",
    )
    parser.add_argument(
        "--mf_min_r2",
        type=float,
        default=0.90,
        help="Minimum R2 required by each multi-fidelity surrogate case.",
    )
    parser.add_argument(
        "--active_learning_min_relative_gain",
        type=float,
        default=0.20,
        help="Minimum relative R2 improvement required after infill.",
    )

    parser.add_argument(
        "--krg_poly",
        type=str,
        default="constant",
        help="Kriging regression basis (constant, linear, quadratic).",
    )
    parser.add_argument(
        "--krg_kernel",
        type=str,
        default="gaussian",
        help="Kriging correlation kernel.",
    )
    parser.add_argument(
        "--krg_theta0",
        type=float,
        default=1.0,
        help="Initial Kriging theta value.",
    )
    parser.add_argument(
        "--krg_theta_bounds",
        type=float,
        nargs=2,
        default=[1.0e-6, 100.0],
        help="Lower and upper bounds for Kriging theta optimization.",
    )

    parser.add_argument(
        "--de_popsize",
        type=int,
        default=10,
        help="Population size for differential evolution.",
    )
    parser.add_argument(
        "--de_maxiter",
        type=int,
        default=30,
        help="Maximum iterations for differential evolution.",
    )
    parser.add_argument(
        "--df_popsize",
        type=int,
        default=20,
        help="Population size for the dragonfly optimizer.",
    )
    parser.add_argument(
        "--df_maxiter",
        type=int,
        default=50,
        help="Maximum iterations for the dragonfly optimizer.",
    )
    parser.add_argument(
        "--opt_tol",
        type=float,
        default=1.0e-6,
        help="Stopping tolerance for both optimizers.",
    )

    args = parser.parse_args()
    args.demos = _expand_demo_selection(args.demos)
    args.ensemble_cases = _expand_case_selection(
        args.ensemble_cases, DEFAULT_ENSEMBLE_CASES
    )
    args.multifidelity_cases = _expand_case_selection(
        args.multifidelity_cases, DEFAULT_MULTIFIDELITY_CASES
    )
    args.optimization_cases = _expand_case_selection(
        args.optimization_cases, DEFAULT_OPTIMIZATION_CASES
    )
    args.krg_params = {
        "poly": args.krg_poly,
        "kernel": args.krg_kernel,
        "theta0": args.krg_theta0,
        "theta_bounds": tuple(args.krg_theta_bounds),
    }
    return args
