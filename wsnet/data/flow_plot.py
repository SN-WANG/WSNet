# Static Visualization Tools for Paper Figures
# Author: Shengning Wang
#
# Provides publication-quality static plots for comparing neural operator
# models on flow simulation tasks. Complements flow_vis.py (animations).

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from torch import Tensor
from typing import Dict, List, Optional


# Paper-quality matplotlib defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for consistent model styling
MODEL_COLORS = {
    "HyperFlowNet": "#2196F3",
    "GeoFNO": "#FF9800",
    "Transolver": "#4CAF50",
    "HyperFlowNet (no LFF)": "#90CAF9",
    "HyperFlowNet (no time)": "#64B5F6",
}


def _get_color(model_name: str, idx: int = 0) -> str:
    """Get color for a model name, falling back to a default palette."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    fallback = ['#E91E63', '#9C27B0', '#00BCD4', '#795548', '#607D8B']
    return fallback[idx % len(fallback)]


# ======================================================================
# 1. Training Curves Comparison
# ======================================================================

def plot_training_curves(
    history_paths: Dict[str, str],
    output_path: str,
    metric: str = "val_loss",
    log_scale: bool = True,
) -> None:
    """Plot loss curves for multiple models from their history.json files.

    Args:
        history_paths: Mapping of model_name -> path to history.json.
        output_path: Path to save the output figure (png/pdf).
        metric: Which metric to plot ('train_loss' or 'val_loss').
        log_scale: Use logarithmic y-axis. Default: True.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, path) in enumerate(history_paths.items()):
        with open(path, 'r') as f:
            history = json.load(f)

        epochs = [h['epoch'] for h in history]
        values = [h[metric] for h in history if h.get(metric) is not None]
        epochs = epochs[:len(values)]

        color = _get_color(name, idx)
        ax.plot(epochs, values, label=name, color=color, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    if log_scale:
        ax.set_yscale('log')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)


# ======================================================================
# 2. Metrics Comparison Bar Chart
# ======================================================================

def plot_metrics_comparison(
    metrics_paths: Dict[str, str],
    output_path: str,
    metric_name: str = "nmse",
    channel_names: Optional[List[str]] = None,
) -> None:
    """Bar chart comparing a metric across models and fields.

    Args:
        metrics_paths: Mapping of model_name -> path to test_metrics.json.
        output_path: Path to save the output figure.
        metric_name: Metric to compare ('nmse', 'r2', 'rmse', 'mae').
        channel_names: Field names to include. If None, auto-detect.
    """
    # Load all metrics
    all_metrics = {}
    for name, path in metrics_paths.items():
        with open(path, 'r') as f:
            data = json.load(f)
        # Average across cases
        case_names = list(data.keys())
        if channel_names is None:
            channel_names = list(data[case_names[0]].keys())
        avg = {}
        for ch in channel_names:
            values = [data[case][ch]["global"][metric_name] for case in case_names]
            avg[ch] = float(np.mean(values))
        all_metrics[name] = avg

    # Plot
    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    n_channels = len(channel_names)
    x = np.arange(n_channels)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_channels * 2), 5))

    for idx, name in enumerate(model_names):
        values = [all_metrics[name][ch] for ch in channel_names]
        offset = (idx - n_models / 2 + 0.5) * bar_width
        color = _get_color(name, idx)
        ax.bar(x + offset, values, bar_width, label=name, color=color, alpha=0.85)

    ax.set_xlabel('Field')
    ax.set_ylabel(metric_name.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, axis='y', alpha=0.3)

    if metric_name in ('nmse', 'mse', 'rmse', 'mae'):
        ax.set_yscale('log')

    fig.savefig(output_path)
    plt.close(fig)


# ======================================================================
# 3. Rollout Error Accumulation Curve
# ======================================================================

def plot_rollout_error(
    pred: Tensor,
    gt: Tensor,
    channel_names: List[str],
    output_path: str,
    metric: str = "nmse",
) -> None:
    """Plot per-timestep error showing error accumulation over rollout.

    Args:
        pred: Predicted sequence. Shape: (T, N, C).
        gt: Ground truth sequence. Shape: (T, N, C).
        channel_names: List of channel names matching C dimension.
        output_path: Path to save the output figure.
        metric: Error metric ('nmse' or 'mse').
    """
    T, N, C = pred.shape
    assert C == len(channel_names), f"Channel count mismatch: {C} vs {len(channel_names)}"

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']

    for c, ch_name in enumerate(channel_names):
        errors = []
        for t in range(T):
            p = pred[t, :, c]
            g = gt[t, :, c]
            if metric == "nmse":
                err = float(((p - g) ** 2).mean() / (g ** 2).mean().clamp(min=1e-10))
            else:
                err = float(((p - g) ** 2).mean())
            errors.append(err)
        ax.plot(range(T), errors, label=ch_name, color=colors[c % len(colors)], linewidth=1.5)

    ax.set_xlabel('Timestep')
    ax.set_ylabel(metric.upper())
    ax.set_yscale('log')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)


# ======================================================================
# 4. Spatial Error Heatmap
# ======================================================================

def plot_error_heatmap(
    gt: Tensor,
    pred: Tensor,
    coords: Tensor,
    timestep: int,
    channel_names: List[str],
    output_path: str,
    figsize_per_col: float = 4.0,
) -> None:
    """Plot spatial error distribution at a specific timestep.

    Creates a scatter plot for each channel showing absolute error as color
    at each mesh node position.

    Args:
        gt: Ground truth sequence. Shape: (T, N, C).
        pred: Predicted sequence. Shape: (T, N, C).
        coords: Node coordinates. Shape: (N, 2).
        timestep: Timestep index to visualize.
        channel_names: List of channel names.
        output_path: Path to save the output figure.
        figsize_per_col: Width per subplot column in inches.
    """
    C = len(channel_names)
    coords_np = coords.numpy() if isinstance(coords, Tensor) else coords
    x, y = coords_np[:, 0], coords_np[:, 1]

    fig, axes = plt.subplots(1, C, figsize=(figsize_per_col * C, figsize_per_col * 0.8))
    if C == 1:
        axes = [axes]

    for c, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        error = torch.abs(pred[timestep, :, c] - gt[timestep, :, c]).numpy()

        # Clip to 2-98 percentile for better visualization
        vmin = np.percentile(error, 2)
        vmax = np.percentile(error, 98)

        sc = ax.scatter(x, y, c=error, cmap='Reds', s=1, vmin=vmin, vmax=vmax,
                        rasterized=True)
        ax.set_title(f'{ch_name} (t={timestep})')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8, label='|Error|')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ======================================================================
# 5. Ablation Study LaTeX Table
# ======================================================================

def generate_ablation_table(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    metric_name: str = "NMSE",
) -> str:
    """Generate a LaTeX table for ablation study results.

    Args:
        results: Nested dict of {variant_name: {field_name: metric_value}}.
            Example: {"Full model": {"Vx": 0.005, "Vy": 0.23, ...}, ...}
        output_path: Path to save the .tex file.
        metric_name: Name of the metric for the table caption.

    Returns:
        LaTeX table string.
    """
    variants = list(results.keys())
    fields = list(results[variants[0]].keys())
    n_fields = len(fields)

    # Build LaTeX
    col_spec = 'l' + 'c' * n_fields
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{Ablation study ({metric_name})}}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        'Variant & ' + ' & '.join(fields) + r' \\',
        r'\midrule',
    ]

    # Find best (lowest) value per field for bolding
    best_per_field = {}
    for field in fields:
        values = [results[v][field] for v in variants]
        best_per_field[field] = min(values)

    for variant in variants:
        cells = []
        for field in fields:
            val = results[variant][field]
            formatted = f'{val:.4f}' if val >= 0.01 else f'{val:.2e}'
            if val == best_per_field[field]:
                formatted = f'\\textbf{{{formatted}}}'
            cells.append(formatted)
        lines.append(f'{variant} & ' + ' & '.join(cells) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    latex_str = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    return latex_str
