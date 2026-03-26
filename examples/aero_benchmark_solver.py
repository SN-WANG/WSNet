"""Main runner for the aero contract benchmark suite."""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import differential_evolution

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import aero_benchmark_config as benchmark_config
import aero_benchmark_functions as benchmark_functions

from wsnet.models.classical.prs import PRS
from wsnet.models.classical.rbf import RBF
from wsnet.models.classical.krg import KRG
from wsnet.models.classical.svr import SVR
from wsnet.models.ensemble.t_ahs import TAHS
from wsnet.models.ensemble.aes_msi import AESMSI
from wsnet.models.multi_fidelity.mfs_mls import MFSMLS
from wsnet.models.multi_fidelity.mmfs import MMFS
from wsnet.models.multi_fidelity.cca_mfs import CCAMFS
from wsnet.models.optimization.dragonfly import dragonfly_optimize
from wsnet.sampling.doe import lhs_design
from wsnet.sampling.so_infill import SingleObjectiveInfill
from wsnet.sampling.mf_infill import MultiFidelityInfill
from wsnet.sampling.mo_infill import MultiObjectiveInfill
from wsnet.utils.hue_logger import hue, logger
from wsnet.utils.seeder import seed_everything


def scale_to_bounds(x_norm: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale normalized Latin hypercube samples from ``[0, 1]`` to physical bounds.

    Args:
        x_norm: Normalized samples with shape ``(num_samples, input_dim)``.
        bounds: Box bounds with shape ``(input_dim, 2)``.

    Returns:
        np.ndarray: Scaled samples with shape ``(num_samples, input_dim)``.

    Raises:
        ValueError: If the input shapes are incompatible.

    Shapes:
        ``(N, D), (D, 2) -> (N, D)``

    Complexity:
        Time ``O(N * D)`` and space ``O(N * D)``.
    """

    if x_norm.ndim != 2 or bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(
            "scale_to_bounds expects x_norm with shape (N, D) and bounds with shape (D, 2)."
        )
    return bounds[:, 0] + x_norm * (bounds[:, 1] - bounds[:, 0])


def sample_lhs(bounds: np.ndarray, num_samples: int, lhs_iterations: int) -> np.ndarray:
    """Generate Latin hypercube samples inside a bounded design space.

    Args:
        bounds: Box bounds with shape ``(input_dim, 2)``.
        num_samples: Number of samples to draw.
        lhs_iterations: Maximin search iterations passed to ``lhs_design``.

    Returns:
        np.ndarray: Scaled sample matrix with shape ``(num_samples, input_dim)``.

    Raises:
        ValueError: If ``num_samples`` is smaller than one.

    Shapes:
        ``(D, 2) -> (N, D)``

    Complexity:
        Time is dominated by ``lhs_design``; memory is ``O(N * D)``.
    """

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    x_norm = lhs_design(num_samples, bounds.shape[0], iterations=lhs_iterations)
    return scale_to_bounds(x_norm, bounds)


def reset_random_state(seed: int) -> None:
    """Reset Python and NumPy random states without emitting extra log messages.

    Args:
        seed: Integer seed used by Python's ``random`` module and NumPy.

    Returns:
        None.

    Raises:
        ValueError: If ``seed`` is not an integer.

    Shapes:
        Not applicable.

    Complexity:
        Time ``O(1)`` and space ``O(1)``.
    """

    if not isinstance(seed, int):
        raise ValueError(f"reset_random_state expects an int seed, got {type(seed)}.")
    random.seed(seed)
    np.random.seed(seed)


def safe_predict(model: Any, x: np.ndarray) -> np.ndarray:
    """Return only the predictive mean from a WSNet surrogate model.

    Args:
        model: Trained surrogate model with a ``predict`` method.
        x: Query points with shape ``(num_samples, input_dim)``.

    Returns:
        np.ndarray: Predictive mean with shape ``(num_samples, output_dim)``.

    Raises:
        RuntimeError: Propagated if the underlying model is not fitted.

    Shapes:
        ``(N, D) -> (N, M)``

    Complexity:
        Time and space depend on the underlying model.
    """

    prediction = model.predict(x)
    return prediction[0] if isinstance(prediction, tuple) else prediction


def evaluate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination.

    Args:
        y_true: Ground-truth responses with shape ``(num_samples, output_dim)``.
        y_pred: Predicted responses with shape ``(num_samples, output_dim)``.

    Returns:
        float: Scalar R2 score aggregated across all outputs.

    Raises:
        ValueError: If the two arrays do not share the same shape.

    Shapes:
        ``(N, M), (N, M) -> scalar``

    Complexity:
        Time ``O(N * M)`` and space ``O(1)`` besides temporary arrays.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"evaluate_r2 expects identical shapes, got {y_true.shape} and {y_pred.shape}."
        )
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(
        np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    )
    return 1.0 - ss_res / (ss_tot + 1.0e-12)


def evaluate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root mean squared error.

    Args:
        y_true: Ground-truth responses with shape ``(num_samples, output_dim)``.
        y_pred: Predicted responses with shape ``(num_samples, output_dim)``.

    Returns:
        float: Scalar RMSE value.

    Raises:
        ValueError: If the two arrays do not share the same shape.

    Shapes:
        ``(N, M), (N, M) -> scalar``

    Complexity:
        Time ``O(N * M)`` and space ``O(1)`` besides temporary arrays.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"evaluate_rmse expects identical shapes, got {y_true.shape} and {y_pred.shape}."
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_relative_gain(before: float, after: float) -> float:
    """Compute the relative improvement from ``before`` to ``after``.

    Args:
        before: Baseline scalar metric.
        after: Improved scalar metric.

    Returns:
        float: Relative gain ``(after - before) / max(abs(before), eps)``.

    Raises:
        ValueError: If either value is not finite.

    Shapes:
        Not applicable.

    Complexity:
        Time ``O(1)`` and space ``O(1)``.
    """

    if not np.isfinite(before) or not np.isfinite(after):
        raise ValueError(
            f"compute_relative_gain expects finite inputs, got before={before}, after={after}."
        )
    return float((after - before) / max(abs(before), 1.0e-12))


def fit_krg(x_train: np.ndarray, y_train: np.ndarray, args: Any) -> KRG:
    """Fit a Kriging surrogate using the shared CLI hyperparameters.

    Args:
        x_train: Training inputs with shape ``(num_samples, input_dim)``.
        y_train: Training targets with shape ``(num_samples, output_dim)``.
        args: Parsed command-line arguments containing ``krg_params``.

    Returns:
        KRG: Trained Kriging surrogate model.

    Raises:
        RuntimeError: Propagated if model fitting fails.

    Shapes:
        ``(N, D), (N, M) -> model``

    Complexity:
        Time depends on the Kriging optimizer and matrix factorization.
    """

    model = KRG(**args.krg_params)
    model.fit(x_train, y_train)
    return model


def compute_mean_objective_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean R2 across multiple objective columns.

    Args:
        y_true: Ground-truth objectives with shape ``(num_samples, num_objectives)``.
        y_pred: Predicted objectives with shape ``(num_samples, num_objectives)``.

    Returns:
        float: Mean column-wise R2 score.

    Raises:
        ValueError: If the two arrays do not share the same shape.

    Shapes:
        ``(N, M), (N, M) -> scalar``

    Complexity:
        Time ``O(N * M)`` and space ``O(1)`` besides temporary arrays.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "compute_mean_objective_r2 expects y_true and y_pred to share the same shape."
        )
    scores = [
        evaluate_r2(y_true[:, idx : idx + 1], y_pred[:, idx : idx + 1])
        for idx in range(y_true.shape[1])
    ]
    return float(np.mean(scores))


def compute_pareto_size(y_values: np.ndarray) -> int:
    """Compute the number of non-dominated points for a minimization problem.

    Args:
        y_values: Objective matrix with shape ``(num_points, num_objectives)``.

    Returns:
        int: Number of Pareto non-dominated rows.

    Raises:
        ValueError: If ``y_values`` is not two-dimensional.

    Shapes:
        ``(N, M) -> scalar``

    Complexity:
        Time ``O(N^2 * M)`` and space ``O(N^2)``.
    """

    if y_values.ndim != 2:
        raise ValueError(
            f"compute_pareto_size expects a 2-D array, got {y_values.ndim} dimensions."
        )
    y_i = y_values[:, np.newaxis, :]
    y_j = y_values[np.newaxis, :, :]
    diff = y_j - y_i
    dominated = np.all(diff <= 0.0, axis=2) & np.any(diff < 0.0, axis=2)
    np.fill_diagonal(dominated, False)
    return int(np.sum(~np.any(dominated, axis=1)))


def to_serializable(value: Any) -> Any:
    """Recursively convert NumPy-heavy objects into JSON-safe Python types.

    Args:
        value: Arbitrary Python object.

    Returns:
        Any: JSON-serializable representation.

    Raises:
        TypeError: Propagated by ``json.dump`` if an unsupported type remains.

    Shapes:
        Not applicable.

    Complexity:
        Time is linear in the nested container size.
    """

    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def run_ensemble_section(args: Any) -> List[Dict[str, Any]]:
    """Run the ensemble surrogate benchmarks for demos ``A`` and ``B``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List[Dict[str, Any]]: Per-case benchmark records.

    Raises:
        RuntimeError: Propagated if a model fit or prediction fails.

    Shapes:
        Not applicable.

    Complexity:
        Dominated by repeated surrogate fitting for the selected cases.
    """

    single_model_builders = {
        "PRS": lambda: PRS(),
        "RBF": lambda: RBF(),
        "KRG": lambda: KRG(**args.krg_params),
        "SVR": lambda: SVR(),
    }
    ensemble_builders = {
        "A": (
            "TAHS",
            lambda: TAHS(threshold=args.ensemble_threshold, krg_params=args.krg_params),
        ),
        "B": (
            "AESMSI",
            lambda: AESMSI(
                threshold=args.ensemble_threshold, krg_params=args.krg_params
            ),
        ),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}>>> Ensemble Benchmark Cases{hue.q}")

    for case_name in args.ensemble_cases:
        reset_random_state(args.seed)
        spec = benchmark_functions.get_scalar_benchmark(case_name)
        config = benchmark_config.DEFAULT_ENSEMBLE_CASES[case_name]
        bounds = spec.bounds_array
        lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
        x_train = sample_lhs(bounds, config["num_train"], lhs_iterations)
        x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
        y_train = spec.evaluate(x_train)
        y_test = spec.evaluate(x_test)

        single_scores: Dict[str, Dict[str, float]] = {}
        for model_name, builder in single_model_builders.items():
            model = builder()
            model.fit(x_train, y_train)
            y_pred = safe_predict(model, x_test)
            single_scores[model_name] = {
                "r2": evaluate_r2(y_test, y_pred),
                "rmse": evaluate_rmse(y_test, y_pred),
            }

        mean_single_r2 = float(
            np.mean([metrics["r2"] for metrics in single_scores.values()])
        )
        case_result: Dict[str, Any] = {
            "case": case_name,
            "input_dim": spec.input_dim,
            "num_train": config["num_train"],
            "num_test": config["num_test"],
            "single_models": single_scores,
            "mean_single_r2": mean_single_r2,
            "algorithms": {},
        }

        logger.info(
            f"  Case {spec.name}: mean single-model R2 = {hue.c}{mean_single_r2:.4f}{hue.q}"
        )

        for demo_label, (algo_name, builder) in ensemble_builders.items():
            if demo_label not in args.demos:
                continue
            model = builder()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            r2_score = evaluate_r2(y_test, y_pred)
            rmse_score = evaluate_rmse(y_test, y_pred)
            relative_gain = compute_relative_gain(mean_single_r2, r2_score)
            passed = relative_gain >= args.ensemble_min_relative_gain
            case_result["algorithms"][algo_name] = {
                "r2": r2_score,
                "rmse": rmse_score,
                "relative_gain": relative_gain,
                "passed": passed,
            }
            status = "PASS" if passed else "FAIL"
            logger.info(
                f"    {algo_name}: R2={r2_score:.4f}, gain={relative_gain:.4f} -> {status}"
            )

        results.append(case_result)

    return results


def run_multifidelity_section(args: Any) -> List[Dict[str, Any]]:
    """Run the multi-fidelity surrogate benchmarks for demos ``C`` to ``E``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List[Dict[str, Any]]: Per-case benchmark records.

    Raises:
        RuntimeError: Propagated if a model fit or prediction fails.

    Shapes:
        Not applicable.

    Complexity:
        Dominated by the selected multi-fidelity model fits.
    """

    model_builders = {
        "C": ("MFSMLS", lambda: MFSMLS(poly_degree=2)),
        "D": ("MMFS", lambda: MMFS()),
        "E": ("CCAMFS", lambda: CCAMFS()),
    }

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}>>> Multi-Fidelity Benchmark Cases{hue.q}")

    for case_name in args.multifidelity_cases:
        reset_random_state(args.seed)
        spec = benchmark_functions.get_multifidelity_benchmark(case_name)
        config = benchmark_config.DEFAULT_MULTIFIDELITY_CASES[case_name]
        bounds = spec.bounds_array
        lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
        x_lf = sample_lhs(bounds, config["num_lf"], lhs_iterations)
        x_hf = sample_lhs(bounds, config["num_hf"], lhs_iterations)
        x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
        y_lf = spec.evaluate_low_fidelity(x_lf)
        y_hf = spec.evaluate_high_fidelity(x_hf)
        y_test = spec.evaluate_high_fidelity(x_test)

        case_result: Dict[str, Any] = {
            "case": case_name,
            "input_dim": spec.input_dim,
            "num_lf": config["num_lf"],
            "num_hf": config["num_hf"],
            "num_test": config["num_test"],
            "algorithms": {},
        }
        logger.info(f"  Case {spec.name}: threshold R2 >= {args.mf_min_r2:.2f}")

        for demo_label, (algo_name, builder) in model_builders.items():
            if demo_label not in args.demos:
                continue
            model = builder()
            model.fit(x_lf, y_lf, x_hf, y_hf)
            y_pred = model.predict(x_test)
            r2_score = evaluate_r2(y_test, y_pred)
            rmse_score = evaluate_rmse(y_test, y_pred)
            passed = r2_score >= args.mf_min_r2
            case_result["algorithms"][algo_name] = {
                "r2": r2_score,
                "rmse": rmse_score,
                "passed": passed,
            }
            status = "PASS" if passed else "FAIL"
            logger.info(f"    {algo_name}: R2={r2_score:.4f} -> {status}")

        results.append(case_result)

    return results


def run_single_objective_active_case(args: Any) -> Dict[str, Any]:
    """Run the single-objective active learning case for demo ``F``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Benchmark record for the active learning case.

    Raises:
        RuntimeError: Propagated if model training or infill optimization fails.

    Shapes:
        Not applicable.

    Complexity:
        Dominated by repeated Kriging fits across the infill iterations.
    """

    config = benchmark_config.DEFAULT_SINGLE_OBJECTIVE_ACTIVE_CASE
    reset_random_state(args.seed)
    spec = benchmark_functions.get_scalar_benchmark(config["name"])
    bounds = spec.bounds_array
    lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
    x_current = sample_lhs(bounds, config["num_initial"], lhs_iterations)
    x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
    y_current = spec.evaluate(x_current)
    y_test = spec.evaluate(x_test)

    model_before = fit_krg(x_current, y_current, args)
    y_pred_before, _ = model_before.predict(x_test)
    r2_before = evaluate_r2(y_test, y_pred_before)
    rmse_before = evaluate_rmse(y_test, y_pred_before)

    history: List[float] = []
    for _ in range(config["num_infill"]):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = SingleObjectiveInfill(
            model=model_iter,
            bounds=bounds,
            y_train=y_current,
            criterion=config["criterion"],
            target_idx=0,
        )
        x_new = strategy.propose()
        y_new = spec.evaluate(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])
        history.append(float(np.min(y_current[:, 0])))

    model_after = fit_krg(x_current, y_current, args)
    y_pred_after, _ = model_after.predict(x_test)
    r2_after = evaluate_r2(y_test, y_pred_after)
    rmse_after = evaluate_rmse(y_test, y_pred_after)
    relative_gain = compute_relative_gain(r2_before, r2_after)

    passed = relative_gain >= args.active_learning_min_relative_gain
    logger.info(f"{hue.b}>>> Demo F: Single-Objective Active Learning{hue.q}")
    logger.info(
        f"  Case {spec.name}: R2 {r2_before:.4f} -> {r2_after:.4f}, "
        f"gain={relative_gain:.4f} -> {'PASS' if passed else 'FAIL'}"
    )

    return {
        "case": spec.name,
        "num_initial": config["num_initial"],
        "num_test": config["num_test"],
        "num_infill": config["num_infill"],
        "criterion": config["criterion"],
        "r2_before": r2_before,
        "r2_after": r2_after,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "relative_gain": relative_gain,
        "history_best": history,
        "passed": passed,
    }


def run_multi_fidelity_active_case(args: Any) -> Dict[str, Any]:
    """Run the multi-fidelity active learning case for demo ``G``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Benchmark record for the multi-fidelity infill case.

    Raises:
        RuntimeError: Propagated if model training or candidate selection fails.

    Shapes:
        Not applicable.

    Complexity:
        Dominated by repeated Kriging fits plus MICO covariance scoring.
    """

    config = benchmark_config.DEFAULT_MULTI_FIDELITY_ACTIVE_CASE
    reset_random_state(args.seed)
    spec = benchmark_functions.get_multifidelity_benchmark(config["name"])
    bounds = spec.bounds_array
    lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
    x_lf = sample_lhs(bounds, config["num_lf"], lhs_iterations)
    x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
    x_current = sample_lhs(bounds, config["num_hf_initial"], lhs_iterations)
    y_lf = spec.evaluate_low_fidelity(x_lf)
    y_current = spec.evaluate_high_fidelity(x_current)
    y_test = spec.evaluate_high_fidelity(x_test)

    model_before = fit_krg(x_current, y_current, args)
    y_pred_before, _ = model_before.predict(x_test)
    r2_before = evaluate_r2(y_test, y_pred_before)
    rmse_before = evaluate_rmse(y_test, y_pred_before)

    history: List[float] = []
    for _ in range(config["num_infill"]):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = MultiFidelityInfill(
            model=model_iter,
            x_hf=x_current,
            y_hf=y_current,
            x_lf=x_lf,
            y_lf=y_lf,
            target_idx=0,
            ratio=config["ratio"],
        )
        x_new = strategy.propose()
        y_new = spec.evaluate_high_fidelity(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])
        history.append(float(np.min(y_current[:, 0])))

    model_after = fit_krg(x_current, y_current, args)
    y_pred_after, _ = model_after.predict(x_test)
    r2_after = evaluate_r2(y_test, y_pred_after)
    rmse_after = evaluate_rmse(y_test, y_pred_after)
    relative_gain = compute_relative_gain(r2_before, r2_after)

    passed = relative_gain >= args.active_learning_min_relative_gain
    logger.info(f"{hue.b}>>> Demo G: Multi-Fidelity Active Learning{hue.q}")
    logger.info(
        f"  Case {spec.name}: R2 {r2_before:.4f} -> {r2_after:.4f}, "
        f"gain={relative_gain:.4f} -> {'PASS' if passed else 'FAIL'}"
    )

    return {
        "case": spec.name,
        "num_hf_initial": config["num_hf_initial"],
        "num_lf": config["num_lf"],
        "num_test": config["num_test"],
        "num_infill": config["num_infill"],
        "ratio": config["ratio"],
        "r2_before": r2_before,
        "r2_after": r2_after,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "relative_gain": relative_gain,
        "history_best": history,
        "passed": passed,
    }


def run_multi_objective_active_case(args: Any) -> Dict[str, Any]:
    """Run the multi-objective active learning case for demo ``H``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Benchmark record for the multi-objective infill case.

    Raises:
        RuntimeError: Propagated if model training or infill optimization fails.

    Shapes:
        Not applicable.

    Complexity:
        Dominated by repeated Kriging fits and EHVI sampling.
    """

    config = benchmark_config.DEFAULT_MULTI_OBJECTIVE_ACTIVE_CASE
    reset_random_state(args.seed)
    spec = benchmark_functions.get_multiobjective_benchmark(config["name"])
    bounds = spec.bounds_array
    lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
    x_current = sample_lhs(bounds, config["num_initial"], lhs_iterations)
    x_test = sample_lhs(bounds, config["num_test"], lhs_iterations)
    y_current = spec.evaluate(x_current)
    y_test = spec.evaluate(x_test)

    model_before = fit_krg(x_current, y_current, args)
    y_pred_before, _ = model_before.predict(x_test)
    r2_before = compute_mean_objective_r2(y_test, y_pred_before)
    pareto_before = compute_pareto_size(y_current)

    for _ in range(config["num_infill"]):
        model_iter = fit_krg(x_current, y_current, args)
        strategy = MultiObjectiveInfill(
            model=model_iter,
            bounds=bounds,
            y_train=y_current,
            obj_idxs=list(range(y_current.shape[1])),
            constraint_idxs=None,
            constraint_ubs=None,
            num_samples=config["num_samples"],
            num_candidates=config["num_candidates"],
            num_restarts=config["num_restarts"],
            beta=config["beta"],
        )
        x_new = strategy.propose()
        y_new = spec.evaluate(x_new)
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])

    model_after = fit_krg(x_current, y_current, args)
    y_pred_after, _ = model_after.predict(x_test)
    r2_after = compute_mean_objective_r2(y_test, y_pred_after)
    pareto_after = compute_pareto_size(y_current)
    relative_gain = compute_relative_gain(r2_before, r2_after)

    passed = relative_gain >= args.active_learning_min_relative_gain
    logger.info(f"{hue.b}>>> Demo H: Multi-Objective Active Learning{hue.q}")
    logger.info(
        f"  Case {spec.name}: mean-R2 {r2_before:.4f} -> {r2_after:.4f}, "
        f"gain={relative_gain:.4f} -> {'PASS' if passed else 'FAIL'}"
    )

    return {
        "case": spec.name,
        "num_initial": config["num_initial"],
        "num_test": config["num_test"],
        "num_infill": config["num_infill"],
        "num_samples": config["num_samples"],
        "num_candidates": config["num_candidates"],
        "num_restarts": config["num_restarts"],
        "beta": config["beta"],
        "r2_before": r2_before,
        "r2_after": r2_after,
        "relative_gain": relative_gain,
        "pareto_size_before": pareto_before,
        "pareto_size_after": pareto_after,
        "passed": passed,
    }


def run_optimization_section(args: Any) -> List[Dict[str, Any]]:
    """Run the unconstrained single-objective optimization cases for demos ``I`` and ``J``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List[Dict[str, Any]]: Per-case optimization records.

    Raises:
        RuntimeError: Propagated if surrogate fitting or optimization fails.

    Shapes:
        Not applicable.

    Complexity:
        Dominated by the selected global optimizers and surrogate evaluations.
    """

    results: List[Dict[str, Any]] = []
    logger.info(f"{hue.b}>>> Optimization Benchmark Cases{hue.q}")

    for case_name in args.optimization_cases:
        reset_random_state(args.seed)
        spec = benchmark_functions.get_scalar_benchmark(case_name)
        config = benchmark_config.DEFAULT_OPTIMIZATION_CASES[case_name]
        bounds = spec.bounds_array
        lhs_iterations = config.get("lhs_iterations", args.lhs_iterations)
        x_train = sample_lhs(bounds, config["num_train"], lhs_iterations)
        y_train = spec.evaluate(x_train)
        surrogate = fit_krg(x_train, y_train, args)

        def objective(x_vec: np.ndarray) -> float:
            prediction, _ = surrogate.predict(np.asarray(x_vec, dtype=np.float64).reshape(1, -1))
            return float(prediction[0, 0])

        case_result: Dict[str, Any] = {
            "case": case_name,
            "num_train": config["num_train"],
            "known_optimum": spec.known_optimum,
            "algorithms": {},
        }

        if "I" in args.demos:
            result_de = differential_evolution(
                objective,
                bounds=[tuple(bound) for bound in bounds],
                maxiter=args.de_maxiter,
                popsize=args.de_popsize,
                tol=args.opt_tol,
                seed=args.seed,
            )
            verified_de = float(spec.evaluate(result_de.x.reshape(1, -1))[0, 0])
            optimum_gap = None
            if spec.known_optimum is not None:
                optimum_gap = abs(verified_de - spec.known_optimum)
            case_result["algorithms"]["DE"] = {
                "x_best": result_de.x,
                "predicted_value": float(result_de.fun),
                "verified_value": verified_de,
                "optimum_gap": optimum_gap,
            }
            logger.info(
                f"  {spec.name} / DE: predicted={result_de.fun:.4f}, verified={verified_de:.4f}"
            )

        if "J" in args.demos:
            result_df = dragonfly_optimize(
                objective,
                bounds=[tuple(bound) for bound in bounds],
                maxiter=args.df_maxiter,
                popsize=args.df_popsize,
                tol=args.opt_tol,
                seed=args.seed,
                multi_objective=False,
            )
            verified_df = float(spec.evaluate(result_df.x.reshape(1, -1))[0, 0])
            optimum_gap = None
            if spec.known_optimum is not None:
                optimum_gap = abs(verified_df - spec.known_optimum)
            case_result["algorithms"]["CFSSDA"] = {
                "x_best": result_df.x,
                "predicted_value": float(result_df.fun),
                "verified_value": verified_df,
                "optimum_gap": optimum_gap,
            }
            logger.info(
                f"  {spec.name} / CFSSDA: predicted={result_df.fun:.4f}, verified={verified_df:.4f}"
            )

        results.append(case_result)

    return results


def save_results(args: Any, payload: Dict[str, Any]) -> str:
    """Persist the benchmark payload as a JSON file.

    Args:
        args: Parsed command-line arguments containing ``save_dir``.
        payload: Nested benchmark result dictionary.

    Returns:
        str: Absolute path to the saved JSON report.

    Raises:
        OSError: If the output directory or file cannot be created.

    Shapes:
        Not applicable.

    Complexity:
        Time is linear in the payload size.
    """

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.abspath(
        os.path.join(args.save_dir, "aero_benchmark_results.json")
    )
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, indent=2)
    return save_path


def main() -> None:
    """Execute the full aero benchmark workflow."""

    args = benchmark_config.get_args()
    seed_everything(args.seed)

    logger.info(f"{hue.b}Aero Contract Benchmark Suite{hue.q}")
    logger.info(f"  Demos    : {args.demos}")
    logger.info(f"  Save dir : {args.save_dir}")

    payload: Dict[str, Any] = {
        "seed": args.seed,
        "demos": args.demos,
        "thresholds": {
            "ensemble_min_relative_gain": args.ensemble_min_relative_gain,
            "mf_min_r2": args.mf_min_r2,
            "active_learning_min_relative_gain": args.active_learning_min_relative_gain,
        },
    }

    if any(label in args.demos for label in ["A", "B"]):
        payload["ensemble"] = run_ensemble_section(args)

    if any(label in args.demos for label in ["C", "D", "E"]):
        payload["multifidelity"] = run_multifidelity_section(args)

    if "F" in args.demos:
        payload["single_objective_active_learning"] = run_single_objective_active_case(
            args
        )

    if "G" in args.demos:
        payload["multi_fidelity_active_learning"] = run_multi_fidelity_active_case(
            args
        )

    if "H" in args.demos:
        payload["multi_objective_active_learning"] = run_multi_objective_active_case(
            args
        )

    if any(label in args.demos for label in ["I", "J"]):
        payload["optimization"] = run_optimization_section(args)

    save_path = save_results(args, payload)
    logger.info(f"{hue.g}Benchmark report saved to {save_path}{hue.q}")


if __name__ == "__main__":
    main()
