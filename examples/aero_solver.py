# Aero Optimization Solver Benchmark Platform
# Author: Shengning Wang

import os
import sys
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.optimize import OptimizeResult

import aero_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)


# Base Surrogate Model
from wsnet.models.classical.krg import KRG

# Ensemble Surrogate Models
from wsnet.models.ensemble.t_ahs import TAHS
from wsnet.models.ensemble.aes_msi import AESMSI

# Multi-fidelity Surrogate Models
from wsnet.models.multi_fidelity.mfs_mls import MFSMLS
from wsnet.models.multi_fidelity.mmfs import MMFS
from wsnet.models.multi_fidelity.cca_mfs import CCAMFS

# Sequential Sampling Methods
from wsnet.sampling.so_infill import SingleObjectiveInfill
from wsnet.sampling.mf_infill import MultiFidelityInfill
from wsnet.sampling.mo_infill import MultiObjectiveInfill

# Optimization Methods
from wsnet.models.optimization.dragonfly import dragonfly_optimize

# Tools
from wsnet.sampling.doe import lhs_design
from wsnet.utils.seeder import seed_everything
from wsnet.utils.hue_logger import hue, logger


# ======================================================================
# Mock Simulation Model
# ======================================================================

class AbaqusModel:
    """Mock Abaqus FEM solver for benchmark testing.

    Maps 3 skin thickness inputs to 4 structural response outputs
    using modified Branin-like test functions.

    Args:
        fidelity: Simulation fidelity level, "high" or "low".
            Low fidelity adds systematic bias (0.85x + 5.0) and Gaussian noise.
    """

    def __init__(self, fidelity: str = "high") -> None:
        self.fidelity = fidelity
        self.input_vars = ["thick1", "thick2", "thick3"]
        self.output_vars = ["weight", "displacement", "stress_skin", "stress_stiff"]

    def run(self, input_arr: np.ndarray) -> np.ndarray:
        """Execute a single simulation.

        Args:
            input_arr: Design variable values, shape (3,).

        Returns:
            Structural response outputs, shape (4,).

        Raises:
            ValueError: If input does not have exactly 3 elements.
        """
        x = np.array(input_arr).flatten()
        if len(x) != 3:
            raise ValueError(f"Expected 3 inputs, got {len(x)}")

        def _branin(x1: float, x2: float) -> float:
            a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
            r, s, t = 6, 10, 1 / (8 * np.pi)
            return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

        y1 = _branin(x[0], x[1]) + 2.0 * x[2]              # weight
        y2 = 0.5 * x[0]**2 + 1.2 * x[1] + np.sin(x[2])     # displacement
        y3 = _branin(x[1], x[2]) + 0.8 * x[0]               # stress_skin
        y4 = (x[0] - 5)**2 + (x[1] - 5)**2 + (x[2] - 5)**2  # stress_stiff

        res = np.array([y1, y2, y3, y4])

        if self.fidelity == "low":
            res = 0.85 * res + 5.0 + np.random.normal(0, 0.1, size=res.shape)

        return res


# ======================================================================
# Helper Functions
# ======================================================================

def scale_to_bounds(x_norm: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale normalized LHS samples [0, 1] to physical bounds.

    Args:
        x_norm: Normalized samples, shape (n, dim).
        bounds: Physical bounds, shape (dim, 2).

    Returns:
        Scaled samples, shape (n, dim).
    """
    return bounds[:, 0] + x_norm * (bounds[:, 1] - bounds[:, 0])


def run_abaqus_batch(x: np.ndarray, fidelity: str = "high") -> np.ndarray:
    """Run AbaqusModel for a batch of input samples.

    Args:
        x: Input samples, shape (n_samples, num_features).
        fidelity: Simulation fidelity, "high" or "low".

    Returns:
        Collected outputs, shape (n_samples, num_outputs).
    """
    model = AbaqusModel(fidelity=fidelity)
    logger.info(f"Running Abaqus batch (fidelity={fidelity}, n={x.shape[0]})...")
    results = [model.run(x[i]) for i in range(x.shape[0])]
    return np.array(results)


def evaluate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> Dict[str, float]:
    """Compute and log prediction performance metrics.

    Args:
        y_true: Ground truth, shape (n, num_outputs).
        y_pred: Predictions, shape (n, num_outputs).
        label: Model identifier for logging.

    Returns:
        Dict with keys "r2", "mse", "rmse".
    """
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    logger.info(f"--- {label} Performance ---")
    logger.info(f"  R2  : {hue.m}{r2:.6f}{hue.q}")
    logger.info(f"  MSE : {hue.m}{mse:.6f}{hue.q}")
    logger.info(f"  RMSE: {hue.m}{rmse:.6f}{hue.q}")
    return {"r2": r2, "mse": mse, "rmse": rmse}


# ======================================================================
# Two-Stage DoE Data Management
# ======================================================================


def generate_and_cache_doe(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """Generate DoE samples or load from single .npy cache.

    Stage 1 (cache miss): LHS -> scale -> Abaqus -> save .npy dict.
    Stage 2 (cache hit): Load .npy dict directly.

    Args:
        args: Parsed arguments.

    Returns:
        Dict with keys: x_train, y_train, x_test, y_test,
                        x_lf, y_lf, x_hf, y_hf.
    """
    cache_path = os.path.join(args.save_dir, "aero_doe_cache.npy")

    # Check cache
    if os.path.isfile(cache_path):
        logger.info(f"{hue.g}Loading cached DoE data from {cache_path}{hue.q}")
        return np.load(cache_path, allow_pickle=True).item()

    # Generate fresh data
    logger.info(f"{hue.c}Generating DoE samples via LHS...{hue.q}")

    x_train_norm = lhs_design(args.num_train, args.num_features, iterations=args.lhs_iterations)
    x_test_norm = lhs_design(args.num_test, args.num_features, iterations=args.lhs_iterations)
    x_lf_norm = lhs_design(args.num_lf, args.num_features, iterations=args.lhs_iterations)
    x_hf_norm = lhs_design(args.num_hf, args.num_features, iterations=args.lhs_iterations)

    x_train = scale_to_bounds(x_train_norm, args.bounds)
    x_test = scale_to_bounds(x_test_norm, args.bounds)
    x_lf = scale_to_bounds(x_lf_norm, args.bounds)
    x_hf = scale_to_bounds(x_hf_norm, args.bounds)

    y_train = run_abaqus_batch(x_train, fidelity="high")
    y_test = run_abaqus_batch(x_test, fidelity="high")
    y_lf = run_abaqus_batch(x_lf, fidelity="low")
    y_hf = run_abaqus_batch(x_hf, fidelity="high")

    data = {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "x_lf": x_lf, "y_lf": y_lf,
        "x_hf": x_hf, "y_hf": y_hf,
    }

    # Save cache
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(cache_path, data, allow_pickle=True)
    logger.info(f"{hue.g}DoE data cached to {cache_path}{hue.q}")
    logger.info(f"  train: {x_train.shape}, test: {x_test.shape}, "
                f"lf: {x_lf.shape}, hf: {x_hf.shape}")

    return data


# ======================================================================
# Visualization Functions (gated by args.visualize)
# ======================================================================

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
_MARKERS = ["s", "^", "D", "v", "P"]


def _sorted_indices(arr: np.ndarray) -> np.ndarray:
    """Return indices that sort arr in ascending order."""
    return np.argsort(arr)


def plot_ensemble(
    y_test: np.ndarray,
    preds: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    """Compare ensemble model predictions against true values.

    Creates a 2x2 subplot grid (one per output). Each subplot shows
    true values as a sorted black line and model predictions as colored lines.
    Saved to {save_dir}/fig_ensemble.png.

    Args:
        y_test: Ground truth, shape (n, num_outputs).
        preds: Dict mapping model name to y_pred array, shape (n, num_outputs).
        args: Parsed arguments (uses save_dir, output_names).
    """
    import matplotlib.pyplot as plt

    num_out = y_test.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(num_out):
        ax = axes[i]
        idx = _sorted_indices(y_test[:, i])
        ax.plot(range(len(idx)), y_test[idx, i], "k-o", markersize=4,
                label="True", linewidth=1.5, zorder=5)
        for ci, (name, y_pred) in enumerate(preds.items()):
            ax.plot(range(len(idx)), y_pred[idx, i], linestyle="--",
                    marker=_MARKERS[ci % len(_MARKERS)], markersize=4,
                    color=_COLORS[ci % len(_COLORS)], label=name, alpha=0.8)
        name = args.output_names[i] if i < len(args.output_names) else f"output_{i}"
        ax.set_xlabel("Sample Index (sorted)")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ensemble Surrogate Models", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(args.save_dir, "fig_ensemble.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


def plot_multifidelity(
    y_test: np.ndarray,
    y_lf_test: np.ndarray,
    preds: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    """Compare multi-fidelity model predictions with LF baseline and HF truth.

    Creates a 2x2 subplot grid. Each subplot shows HF truth (black),
    LF baseline (gray dashed), and each MF model prediction (colored).
    Saved to {save_dir}/fig_multifidelity.png.

    Args:
        y_test: HF ground truth, shape (n, num_outputs).
        y_lf_test: LF baseline at test points, shape (n, num_outputs).
        preds: Dict mapping model name to y_pred array, shape (n, num_outputs).
        args: Parsed arguments (uses save_dir, output_names).
    """
    import matplotlib.pyplot as plt

    num_out = y_test.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(num_out):
        ax = axes[i]
        idx = _sorted_indices(y_test[:, i])
        ax.plot(range(len(idx)), y_test[idx, i], "k-o", markersize=4,
                label="HF True", linewidth=1.5, zorder=5)
        ax.plot(range(len(idx)), y_lf_test[idx, i], "--", color="gray",
                markersize=3, label="LF Baseline", alpha=0.7)
        for ci, (name, y_pred) in enumerate(preds.items()):
            ax.plot(range(len(idx)), y_pred[idx, i], linestyle="--",
                    marker=_MARKERS[ci % len(_MARKERS)], markersize=4,
                    color=_COLORS[ci % len(_COLORS)], label=name, alpha=0.8)
        name = args.output_names[i] if i < len(args.output_names) else f"output_{i}"
        ax.set_xlabel("Sample Index (sorted)")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-Fidelity Surrogate Models", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(args.save_dir, "fig_multifidelity.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


def plot_sequential(
    y_test: np.ndarray,
    preds: Dict[str, np.ndarray],
    infill_dict: Dict[str, Dict[str, np.ndarray]],
    mo_pareto: Optional[np.ndarray],
    mo_infill_obj: Optional[np.ndarray],
    args: argparse.Namespace,
) -> None:
    """Visualize sequential sampling results.

    Upper 2x2: sorted prediction curves for Demo F/G with infill points in red.
    Lower row: Demo H Pareto front scatter (if available).
    Saved to {save_dir}/fig_sequential.png.

    Args:
        y_test: Ground truth, shape (n, num_outputs).
        preds: Dict mapping method name to y_pred array (Demo F, G only).
        infill_dict: Dict mapping method name to {"y": np.ndarray} of infill outputs.
        mo_pareto: Pareto front points from Demo H, shape (n_pf, 2), or None.
        mo_infill_obj: All infill objective values from Demo H, shape (n_infill, 2), or None.
        args: Parsed arguments.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    has_pareto = mo_pareto is not None and len(mo_pareto) > 0

    if has_pareto:
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])
    else:
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig)

    num_out = y_test.shape[1]
    for i in range(min(num_out, 4)):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        idx = _sorted_indices(y_test[:, i])
        ax.plot(range(len(idx)), y_test[idx, i], "k-o", markersize=4,
                label="True", linewidth=1.5, zorder=5)
        for ci, (name, y_pred) in enumerate(preds.items()):
            ax.plot(range(len(idx)), y_pred[idx, i], linestyle="--",
                    marker=_MARKERS[ci % len(_MARKERS)], markersize=4,
                    color=_COLORS[ci % len(_COLORS)], label=name, alpha=0.8)
        oname = args.output_names[i] if i < len(args.output_names) else f"output_{i}"
        ax.set_xlabel("Sample Index (sorted)")
        ax.set_ylabel(oname)
        ax.set_title(oname)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Lower row: Demo H Pareto front
    if has_pareto:
        ax_pf = fig.add_subplot(gs[2, :])
        ax_pf.scatter(mo_pareto[:, 0], mo_pareto[:, 1], s=50, c="tab:blue",
                      edgecolors="k", linewidths=0.5, label="Pareto Front", zorder=3)
        if mo_infill_obj is not None and len(mo_infill_obj) > 0:
            ax_pf.scatter(mo_infill_obj[:, 0], mo_infill_obj[:, 1], s=40,
                          c="red", marker="x", linewidths=1.5,
                          label="Infill Points", zorder=4)
        obj_names = [args.output_names[i] for i in args.obj_indices[:2]]
        ax_pf.set_xlabel(obj_names[0])
        ax_pf.set_ylabel(obj_names[1])
        ax_pf.set_title("Demo H: Multi-Objective Pareto Front")
        ax_pf.legend(fontsize=9)
        ax_pf.grid(True, alpha=0.3)

    fig.suptitle("Sequential Sampling", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(args.save_dir, "fig_sequential.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


def plot_single_opt(
    opt_results: Dict[str, dict],
    args: argparse.Namespace,
) -> None:
    """Visualize single-objective optimization results.

    Left: bar chart comparing predicted vs verified objective for DE and CFSSDA.
    Right: grouped bar chart of optimal design variables.
    Saved to {save_dir}/fig_single_opt.png.

    Args:
        opt_results: Dict with optimizer names as keys, each containing
            "single_pred", "single_verified", "single_x".
        args: Parsed arguments.
    """
    import matplotlib.pyplot as plt

    names = list(opt_results.keys())
    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: predicted vs verified objective
    x_pos = np.arange(len(names))
    pred_vals = [opt_results[n]["single_pred"] for n in names]
    veri_vals = [opt_results[n]["single_verified"] for n in names]
    width = 0.35
    ax1.bar(x_pos - width / 2, pred_vals, width, label="Predicted", color="tab:blue")
    ax1.bar(x_pos + width / 2, veri_vals, width, label="Verified", color="tab:orange")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names)
    obj_name = args.output_names[args.opt_single_idx]
    ax1.set_ylabel(obj_name)
    ax1.set_title(f"Objective: {obj_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: optimal design variables
    num_vars = args.num_features
    x_pos_v = np.arange(num_vars)
    total_width = 0.7
    bar_width = total_width / len(names)
    for ci, n in enumerate(names):
        x_opt = opt_results[n]["single_x"]
        offset = (ci - (len(names) - 1) / 2) * bar_width
        ax2.bar(x_pos_v + offset, x_opt, bar_width, label=n,
                color=_COLORS[ci % len(_COLORS)])
    ax2.set_xticks(x_pos_v)
    ax2.set_xticklabels(args.input_names[:num_vars])
    ax2.set_ylabel("Design Variable Value")
    ax2.set_title("Optimal Design Variables")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Single-Objective Optimization", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(args.save_dir, "fig_single_opt.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


def plot_multi_opt(
    opt_results: Dict[str, dict],
    weight_ub: float,
    args: argparse.Namespace,
) -> None:
    """Visualize multi-objective optimization results.

    Left: Pareto front scatter (CFSSDA) with DE optimum marked as star.
    Right: constraint bar chart showing weight values vs upper bound.
    Saved to {save_dir}/fig_multi_opt.png.

    Args:
        opt_results: Dict with optimizer names as keys, each containing
            "multi_verified_obj", "multi_verified_weight", and optionally "pareto_f".
        weight_ub: Weight constraint upper bound.
        args: Parsed arguments.
    """
    import matplotlib.pyplot as plt

    names = list(opt_results.keys())
    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    obj_names = [args.output_names[i] for i in args.obj_indices[:2]]

    # Left: Pareto front + DE optimum
    has_pareto = False
    for n in names:
        pf = opt_results[n].get("pareto_f")
        if pf is not None and len(pf) > 0:
            ax1.scatter(pf[:, 0], pf[:, 1], s=50, c="tab:blue",
                        edgecolors="k", linewidths=0.5,
                        label=f"{n} Pareto", zorder=3)
            has_pareto = True
        obj_v = opt_results[n].get("multi_verified_obj")
        if obj_v is not None:
            marker = "*" if not has_pareto else "^"
            ax1.scatter(obj_v[0], obj_v[1], s=150, marker=marker,
                        c=_COLORS[names.index(n) % len(_COLORS)],
                        edgecolors="k", linewidths=1,
                        label=f"{n} Optimum", zorder=4)
    ax1.set_xlabel(obj_names[0])
    ax1.set_ylabel(obj_names[1])
    ax1.set_title("Pareto Front & Optima")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: constraint satisfaction
    x_pos = np.arange(len(names))
    weights = [opt_results[n].get("multi_verified_weight", 0.0) for n in names]
    colors = ["tab:green" if w <= weight_ub else "tab:red" for w in weights]
    ax2.bar(x_pos, weights, 0.5, color=colors, edgecolor="k", linewidth=0.5)
    ax2.axhline(y=weight_ub, color="r", linestyle="--", linewidth=1.5,
                label=f"Constraint UB ({weight_ub:.2f})")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names)
    constraint_name = args.output_names[args.constraint_indices[0]]
    ax2.set_ylabel(constraint_name)
    ax2.set_title(f"Constraint: {constraint_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Multi-Objective Optimization", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(args.save_dir, "fig_multi_opt.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


# ======================================================================
# Pipeline Functions
# ======================================================================

def ensemble_pipeline(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Run ensemble surrogate demos.

    A: TAHS (Two-Stage Adaptive Hybrid Surrogate)
    B: AESMSI (Adaptive Ensemble by Minimum Screening Index)

    Args:
        args: Parsed arguments.
        x_train: Training inputs, shape (n_train, num_features).
        y_train: Training outputs, shape (n_train, num_outputs).
        x_test: Test inputs, shape (n_test, num_features).
        y_test: Test outputs, shape (n_test, num_outputs).

    Returns:
        Dict mapping demo label to metrics dict.
    """
    results = {}
    preds = {}

    # Demo A: TAHS
    if "A" in args.demos:
        logger.info(f"{hue.b}>>> Demo A: T-AHS{hue.q}")
        model = TAHS(threshold=args.ensemble_threshold)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results["TAHS"] = evaluate_metrics(y_test, y_pred, "T-AHS")
        preds["TAHS"] = y_pred

    # Demo B: AESMSI
    if "B" in args.demos:
        logger.info(f"{hue.b}>>> Demo B: AES-MSI{hue.q}")
        model = AESMSI(threshold=args.ensemble_threshold)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results["AESMSI"] = evaluate_metrics(y_test, y_pred, "AES-MSI")
        preds["AESMSI"] = y_pred

    if args.visualize and preds:
        plot_ensemble(y_test, preds, args)

    return results


def multifidelity_pipeline(
    args: argparse.Namespace,
    x_lf: np.ndarray,
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    y_hf: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Run multi-fidelity surrogate demos.

    C: MFS-MLS (Moving Least Squares)
    D: MMFS (Modified Multi-Fidelity Surrogate)
    E: CCA-MFS (Canonical Correlation Analysis)

    Args:
        args: Parsed arguments.
        x_lf: Low-fidelity inputs, shape (n_lf, num_features).
        y_lf: Low-fidelity outputs, shape (n_lf, num_outputs).
        x_hf: High-fidelity inputs, shape (n_hf, num_features).
        y_hf: High-fidelity outputs, shape (n_hf, num_outputs).
        x_test: Test inputs, shape (n_test, num_features).
        y_test: Test outputs, shape (n_test, num_outputs).

    Returns:
        Dict mapping demo label to metrics dict.
    """
    results = {}
    preds = {}

    # Demo C: MFS-MLS
    if "C" in args.demos:
        logger.info(f"{hue.b}>>> Demo C: MFS-MLS{hue.q}")
        model = MFSMLS(poly_degree=args.mf_poly_degree)
        model.fit(x_lf, y_lf, x_hf, y_hf)
        y_pred = model.predict(x_test)
        results["MFSMLS"] = evaluate_metrics(y_test, y_pred, "MFS-MLS")
        preds["MFS-MLS"] = y_pred

    # Demo D: MMFS
    if "D" in args.demos:
        logger.info(f"{hue.b}>>> Demo D: MMFS{hue.q}")
        model = MMFS(sigma_bounds=tuple(args.mf_sigma_bounds))
        model.fit(x_lf, y_lf, x_hf, y_hf)
        y_pred = model.predict(x_test)
        results["MMFS"] = evaluate_metrics(y_test, y_pred, "MMFS")
        preds["MMFS"] = y_pred

    # Demo E: CCA-MFS
    if "E" in args.demos:
        logger.info(f"{hue.b}>>> Demo E: CCA-MFS{hue.q}")
        model = CCAMFS()
        model.fit(x_lf, y_lf, x_hf, y_hf)
        y_pred = model.predict(x_test)
        results["CCAMFS"] = evaluate_metrics(y_test, y_pred, "CCA-MFS")
        preds["CCA-MFS"] = y_pred

    if args.visualize and preds:
        y_lf_test = run_abaqus_batch(x_test, fidelity="low")
        plot_multifidelity(y_test, y_lf_test, preds, args)

    return results


def sequential_pipeline(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_lf: np.ndarray,
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    y_hf: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Optional[KRG]]:
    """Run sequential infill demos.

    F: KRG + EI (Expected Improvement)
    G: KRG + MICO (Multi-Fidelity Infill)
    H: KRG + MOInfill (Constrained IS-EHVI)

    Args:
        args: Parsed arguments.
        x_train: Training inputs, shape (n_train, num_features).
        y_train: Training outputs, shape (n_train, num_outputs).
        x_test: Test inputs, shape (n_test, num_features).
        y_test: Test outputs, shape (n_test, num_outputs).
        x_lf: Low-fidelity inputs, shape (n_lf, num_features).
        y_lf: Low-fidelity outputs, shape (n_lf, num_outputs).
        x_hf: High-fidelity inputs, shape (n_hf, num_features).
        y_hf: High-fidelity outputs, shape (n_hf, num_outputs).

    Returns:
        Tuple of (metrics_dict, trained_krg_model).
        The KRG model from Demo F is returned for reuse by the optimization pipeline.
    """
    results = {}
    model_krg = None
    preds = {}
    infill_dict = {}
    mo_pareto = None
    mo_infill_obj = None

    # Demo F: KRG + Single-Objective Infill (EI)
    if "F" in args.demos:
        logger.info(f"{hue.b}>>> Demo F: KRG + Infill ({args.infill_criterion.upper()}){hue.q}")

        x_current = np.copy(x_train)
        y_current = np.copy(y_train)

        model_krg = KRG(**args.krg_params)
        model_krg.fit(x_current, y_current)

        convergence_history = []

        for i in range(args.num_infill):
            strategy = SingleObjectiveInfill(
                model=model_krg,
                bounds=args.bounds,
                y_train=y_current,
                criterion=args.infill_criterion,
                target_idx=0,
            )
            x_new = strategy.propose()
            y_new_val = AbaqusModel(fidelity="high").run(x_new.flatten())
            y_new = y_new_val.reshape(1, -1)

            if np.isnan(y_new).any():
                logger.info(f"{hue.r}  Iteration {i+1}: NaN detected, skipping.{hue.q}")
                continue

            x_current = np.vstack([x_current, x_new])
            y_current = np.vstack([y_current, y_new])
            model_krg.fit(x_current, y_current)

            best_val = float(y_current[:, 0].min())
            convergence_history.append(best_val)

        y_pred, _ = model_krg.predict(x_test)
        results["KRG_Infill"] = evaluate_metrics(y_test, y_pred, "KRG + Infill")
        preds["KRG+EI"] = y_pred

    # Demo G: KRG + MICO (Multi-Fidelity Infill)
    if "G" in args.demos:
        logger.info(f"{hue.b}>>> Demo G: KRG + MICOInfill{hue.q}")

        x_mico = np.copy(x_hf)
        y_mico = np.copy(y_hf)

        model_krg_mico = KRG(**args.krg_params)
        model_krg_mico.fit(x_mico, y_mico)

        for i in range(args.num_infill):
            mico_strategy = MultiFidelityInfill(
                model=model_krg_mico,
                x_hf=x_mico,
                y_hf=y_mico,
                x_lf=x_lf,
                y_lf=y_lf,
                target_idx=0,
                ratio=args.mico_ratio,
            )
            x_new = mico_strategy.propose()
            y_new_val = AbaqusModel(fidelity="high").run(x_new.flatten())
            y_new = y_new_val.reshape(1, -1)

            if np.isnan(y_new).any():
                logger.info(f"{hue.r}  Iteration {i+1}: NaN detected, skipping.{hue.q}")
                continue

            x_mico = np.vstack([x_mico, x_new])
            y_mico = np.vstack([y_mico, y_new])
            model_krg_mico.fit(x_mico, y_mico)

        y_pred, _ = model_krg_mico.predict(x_test)
        results["KRG_MICO"] = evaluate_metrics(y_test, y_pred, "KRG + MICOInfill")
        preds["KRG+MICO"] = y_pred

    # Demo H: KRG + Multi-Objective Infill (Constrained IS-EHVI)
    if "H" in args.demos:
        logger.info(f"{hue.b}>>> Demo H: KRG + MOInfill (Constrained IS-EHVI){hue.q}")

        constraint_ubs = np.array([
            np.percentile(y_train[:, idx], args.constraint_percentile)
            for idx in args.constraint_indices
        ])

        x_mo = np.copy(x_train)
        y_mo = np.copy(y_train)
        model_krg_mo = KRG(**args.krg_params)
        model_krg_mo.fit(x_mo, y_mo)

        mo_infill_list = []

        for i in range(args.num_infill):
            mo_strategy = MultiObjectiveInfill(
                model=model_krg_mo,
                bounds=args.bounds,
                y_train=y_mo,
                obj_idxs=args.obj_indices,
                constraint_idxs=args.constraint_indices,
                constraint_ubs=constraint_ubs,
            )
            x_new = mo_strategy.propose()
            y_new_val = AbaqusModel(fidelity="high").run(x_new.flatten())
            y_new = y_new_val.reshape(1, -1)

            if np.isnan(y_new).any():
                logger.info(f"{hue.r}  Iteration {i+1}: NaN detected, skipping.{hue.q}")
                continue

            x_mo = np.vstack([x_mo, x_new])
            y_mo = np.vstack([y_mo, y_new])
            model_krg_mo.fit(x_mo, y_mo)
            mo_infill_list.append(y_new[0, args.obj_indices])
            logger.info(
                f"  Iteration {i+1}/{args.num_infill}: "
                f"stress_skin={y_new[0, 2]:.4f}, stress_stiff={y_new[0, 3]:.4f}"
            )

        # Report Pareto front
        y_mo_obj = y_mo[:, args.obj_indices]
        pf_mask = mo_strategy._compute_pareto_mask(y_mo_obj)
        y_pf = y_mo_obj[pf_mask]
        logger.info(f"  Pareto front ({pf_mask.sum()} points):")
        for pt in y_pf:
            logger.info(f"    stress_skin={pt[0]:.4f}, stress_stiff={pt[1]:.4f}")

        mo_pareto = y_pf
        if mo_infill_list:
            mo_infill_obj = np.array(mo_infill_list)

        y_pred, _ = model_krg_mo.predict(x_test)
        results["KRG_MOInfill"] = evaluate_metrics(y_test, y_pred, "KRG + MOInfill")

    if args.visualize and preds:
        plot_sequential(y_test, preds, infill_dict, mo_pareto, mo_infill_obj, args)

    return results, model_krg


def optimization_pipeline(
    args: argparse.Namespace,
    model_krg: KRG,
    y_train: np.ndarray,
) -> Dict[str, dict]:
    """Run optimizer demos on the KRG surrogate surface.

    I: DE (Differential Evolution) — single and multi-objective
    J: CFSSDA (Dragonfly) — single and multi-objective with Pareto front

    Args:
        args: Parsed arguments.
        model_krg: Pre-trained KRG surrogate model.
        y_train: Training outputs for constraint bound computation,
            shape (n_train, num_outputs).

    Returns:
        Dict mapping demo label to optimization results.
    """
    results = {}
    plot_data = {}

    scipy_bounds = [(args.bounds[i, 0], args.bounds[i, 1])
                    for i in range(args.num_features)]
    weight_ub = float(np.percentile(
        y_train[:, args.constraint_indices[0]], args.constraint_percentile
    ))

    def _single_obj(x_vec: np.ndarray) -> float:
        pred, _ = model_krg.predict(x_vec.reshape(1, -1))
        return float(pred[0, args.opt_single_idx])

    def _multi_obj(x_vec: np.ndarray) -> np.ndarray:
        pred, _ = model_krg.predict(x_vec.reshape(1, -1))
        return pred[0, args.obj_indices]

    weight_con = NonlinearConstraint(
        fun=lambda x: float(model_krg.predict(x.reshape(1, -1))[0][0, 0]),
        lb=-np.inf, ub=weight_ub,
    )

    # Demo I: DE (Differential Evolution)
    if "I" in args.demos:
        logger.info(f"{hue.b}>>> Demo I: DE{hue.q}")

        # Single-objective
        ri_s: OptimizeResult = differential_evolution(
            func=_single_obj, bounds=scipy_bounds, constraints=weight_con,
            strategy="best1bin", maxiter=args.de_maxiter, popsize=args.de_popsize,
            tol=args.opt_tol, seed=args.seed,
        )
        true_i_s = AbaqusModel(fidelity="high").run(ri_s.x)
        logger.info(f"  [Single-obj] best x : {ri_s.x}")
        logger.info(f"  Predicted mises     : {hue.c}{ri_s.fun:.6f}{hue.q}  |  "
                     f"Verified: {hue.g}{true_i_s[args.opt_single_idx]:.6f}{hue.q}")
        logger.info(f"  Verified weight     : {true_i_s[0]:.6f}  (ub={weight_ub:.4f})")

        # Multi-objective (scalarized)
        ri_m: OptimizeResult = differential_evolution(
            func=lambda x: float(np.sum(_multi_obj(x))),
            bounds=scipy_bounds, constraints=weight_con,
            strategy="best1bin", maxiter=args.de_maxiter, popsize=args.de_popsize,
            tol=args.opt_tol, seed=args.seed,
        )
        true_i_m = AbaqusModel(fidelity="high").run(ri_m.x)
        logger.info(f"  [Multi-obj]  best x : {ri_m.x}")
        logger.info(f"  Predicted [sk, ss]  : {_multi_obj(ri_m.x)}")
        logger.info(f"  Verified  [sk, ss]  : {hue.g}{true_i_m[args.obj_indices]}{hue.q}")

        results["DE"] = {"single_x": ri_s.x, "single_fun": ri_s.fun,
                         "multi_x": ri_m.x}
        plot_data["DE"] = {
            "single_pred": ri_s.fun,
            "single_verified": float(true_i_s[args.opt_single_idx]),
            "single_x": ri_s.x,
            "multi_verified_obj": true_i_m[args.obj_indices],
            "multi_verified_weight": float(true_i_m[0]),
        }

    # Demo J: CFSSDA (Dragonfly)
    if "J" in args.demos:
        logger.info(f"{hue.b}>>> Demo J: CFSSDA{hue.q}")

        # Single-objective
        rj_s = dragonfly_optimize(
            func=_single_obj, bounds=scipy_bounds, constraints=weight_con,
            maxiter=args.df_maxiter, popsize=args.df_popsize,
            tol=args.opt_tol, seed=args.seed, multi_objective=False,
        )
        true_j_s = AbaqusModel(fidelity="high").run(rj_s.x)
        logger.info(f"  [Single-obj] best x : {rj_s.x}")
        logger.info(f"  Predicted mises     : {hue.c}{rj_s.fun:.6f}{hue.q}  |  "
                     f"Verified: {hue.g}{true_j_s[args.opt_single_idx]:.6f}{hue.q}")
        logger.info(f"  Verified weight     : {true_j_s[0]:.6f}  (ub={weight_ub:.4f})")

        # Multi-objective with Pareto front
        rj_m = dragonfly_optimize(
            func=_multi_obj, bounds=scipy_bounds, constraints=weight_con,
            maxiter=args.df_maxiter, popsize=args.df_popsize,
            tol=args.opt_tol, seed=args.seed,
            multi_objective=True, scalarization="weighted_sum", return_pareto=True,
        )
        true_j_m = AbaqusModel(fidelity="high").run(rj_m.x)
        logger.info(f"  [Multi-obj]  best x : {rj_m.x}")
        logger.info(f"  Predicted [sk, ss]  : {_multi_obj(rj_m.x)}")
        logger.info(f"  Verified  [sk, ss]  : {hue.g}{true_j_m[args.obj_indices]}{hue.q}")

        pareto_f = None
        if hasattr(rj_m, "pareto_f") and rj_m.pareto_f is not None:
            pareto_f = np.array(rj_m.pareto_f)
            logger.info(f"  Pareto front ({len(pareto_f)} pts):")
            for pt in pareto_f:
                logger.info(f"    stress_skin={pt[0]:.4f}, stress_stiff={pt[1]:.4f}")

        results["CFSSDA"] = {"single_x": rj_s.x, "single_fun": rj_s.fun,
                             "multi_x": rj_m.x}
        plot_data["CFSSDA"] = {
            "single_pred": rj_s.fun,
            "single_verified": float(true_j_s[args.opt_single_idx]),
            "single_x": rj_s.x,
            "multi_verified_obj": true_j_m[args.obj_indices],
            "multi_verified_weight": float(true_j_m[0]),
            "pareto_f": pareto_f,
        }

    if args.visualize and plot_data:
        plot_single_opt(plot_data, args)
        plot_multi_opt(plot_data, weight_ub, args)

    return results


# ======================================================================
# Main Entry Point
# ======================================================================

if __name__ == "__main__":
    args = aero_config.get_args()
    seed_everything(args.seed)

    logger.info(f"{hue.b}Aero Optimization Benchmark Platform{hue.q}")
    logger.info(f"  Demos    : {args.demos}")
    logger.info(f"  Visualize: {args.visualize}")
    logger.info(f"  Save dir : {args.save_dir}")

    # 1. Generate or load DoE data
    data = generate_and_cache_doe(args)

    # 2. Ensemble pipeline (A, B)
    if any(d in args.demos for d in ["A", "B"]):
        ensemble_pipeline(
            args, data["x_train"], data["y_train"],
            data["x_test"], data["y_test"],
        )

    # 3. Multi-fidelity pipeline (C, D, E)
    if any(d in args.demos for d in ["C", "D", "E"]):
        multifidelity_pipeline(
            args, data["x_lf"], data["y_lf"],
            data["x_hf"], data["y_hf"],
            data["x_test"], data["y_test"],
        )

    # 4. Sequential pipeline (F, G, H)
    model_krg = None
    if any(d in args.demos for d in ["F", "G", "H"]):
        _, model_krg = sequential_pipeline(
            args, data["x_train"], data["y_train"],
            data["x_test"], data["y_test"],
            data["x_lf"], data["y_lf"],
            data["x_hf"], data["y_hf"],
        )

    # 5. Optimization pipeline (I, J)
    if any(d in args.demos for d in ["I", "J"]):
        if model_krg is None:
            logger.info(f"{hue.c}Training KRG surrogate for optimization...{hue.q}")
            model_krg = KRG(**args.krg_params)
            model_krg.fit(data["x_train"], data["y_train"])
        optimization_pipeline(args, model_krg, data["y_train"])

    logger.info(f"{hue.b}Process completed successfully.{hue.q}")
