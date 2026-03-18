# Main Script for Flow Simulation: Training, Inference, and Baseline Comparison
# Author: Shengning Wang

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple

import flow_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)


from wsnet.models.neural.geofno import GeoFNO
from wsnet.models.neural.hyperflow_net import HyperFlowNet
from wsnet.models.neural.transolver import Transolver

from wsnet.data.flow_data import FlowData
from wsnet.data.flow_vis import FlowVis
from wsnet.data.boundary import BoundaryCondition
from wsnet.data.flow_plot import (
    plot_training_curves, plot_rollout_error,
    plot_error_heatmap, plot_metrics_comparison,
)
from wsnet.data.scaler import StandardScalerTensor, MinMaxScalerTensor

from wsnet.training.rollout_trainer import RolloutTrainer
from wsnet.training.teacher_forcing_trainer import TeacherForcingTrainer
from wsnet.training.base_criterion import Metrics, NMSECriterion

from wsnet.utils.seeder import seed_everything
from wsnet.utils.hue_logger import hue, logger


# ======================================================================
# 1. Dataset Wrapper with Scalers
# ======================================================================

class ScaledCFDataset(Dataset):
    """Wraps FlowData to apply feature standardization and coordinate normalization.

    Attributes:
        dataset: The underlying raw dataset.
        feature_scaler: Fitted StandardScalerTensor.
        coord_scaler: Fitted MinMaxScalerTensor.
    """
    def __init__(
            self, dataset: FlowData,
            feature_scaler: StandardScalerTensor,
            coord_scaler: MinMaxScalerTensor,
        ):
        self.dataset = dataset
        self.feature_scaler = feature_scaler
        self.coord_scaler = coord_scaler

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float, float]:
        seq, coords, start_t_norm, dt_norm = self.dataset[idx]
        seq_std = self.feature_scaler.transform(seq)
        coords_norm = self.coord_scaler.transform(coords)
        return seq_std, coords_norm, start_t_norm, dt_norm


# ======================================================================
# 2. Model Factory
# ======================================================================

def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Instantiate the model selected by --model_type.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Initialized neural operator model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    in_ch = out_ch = args.spatial_dim + 2  # [Vx, Vy, (Vz,) P, T]

    if args.model_type == "hyperflownet":
        return HyperFlowNet(
            in_channels=in_ch, out_channels=out_ch,
            spatial_dim=args.spatial_dim,
            width=args.width, depth=args.depth,
            num_slices=args.num_slices, num_heads=args.num_heads,
            # Ablation switches
            use_spatial_encoding=args.use_spatial_encoding,
            use_temporal_encoding=args.use_temporal_encoding,
            # Encoding params
            coord_features=args.coord_features,
            time_features=args.time_features, freq_base=args.freq_base,
        )

    elif args.model_type == "geofno":
        deform_kw = {"num_layers": args.deform_layers, "hidden_dim": args.deform_hidden}
        return GeoFNO(
            in_channels=in_ch, out_channels=out_ch,
            modes=args.modes, latent_grid_size=args.latent_grid_size,
            depth=args.depth, width=args.width,
            deformation_kwargs=deform_kw,
        )

    elif args.model_type == "transolver":
        return Transolver(
            in_channels=in_ch, out_channels=out_ch,
            spatial_dim=args.spatial_dim,
            width=args.width, depth=args.depth,
            num_slices=args.tsv_num_slices, num_heads=args.tsv_num_heads,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{args.model_type}'. "
            f"Choices: hyperflownet, geofno, transolver"
        )


# ======================================================================
# 3. Trainer Factory
# ======================================================================

def _build_trainer(args: argparse.Namespace, model: torch.nn.Module,
                   scalers: Dict, output_dir: Path, boundary_condition=None):
    """Instantiate the trainer selected by --trainer_type.

    Args:
        args: Parsed command-line arguments.
        model: The model to train.
        scalers: Dictionary of data scalers for checkpoint saving.
        output_dir: Directory for saving artifacts.
        boundary_condition: Optional BoundaryCondition for hard BC enforcement.

    Returns:
        Configured trainer instance.

    Raises:
        ValueError: If trainer_type is not recognized.
    """
    if args.trainer_type == "rollout":
        return RolloutTrainer(
            model=model, lr=args.lr, max_epochs=args.max_epochs,
            scalers=scalers, output_dir=output_dir, device=args.device,
            weight_decay=args.weight_decay, eta_min=args.eta_min,
            max_rollout_steps=args.max_rollout_steps,
            rollout_patience=args.rollout_patience,
            noise_std_init=args.noise_std_init, noise_decay=args.noise_decay,
            boundary_condition=boundary_condition,
            channel_weights=args.channel_weights,
        )

    elif args.trainer_type == "teacher_forcing":
        return TeacherForcingTrainer(
            model=model, lr=args.lr, max_epochs=args.max_epochs,
            scalers=scalers, output_dir=output_dir, device=args.device,
            weight_decay=args.weight_decay, eta_min=args.eta_min,
        )

    else:
        raise ValueError(
            f"Unknown trainer_type '{args.trainer_type}'. "
            f"Choices: rollout, teacher_forcing"
        )


# ======================================================================
# 4. Training Pipeline
# ======================================================================

def train_pipeline(args: argparse.Namespace) -> None:
    """Execute the training workflow: data loading -> scaling -> model init -> training.

    Args:
        args: Parsed command-line arguments.
    """
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Preparation ---
    logger.info("initializing datasets...")
    train_raw, val_raw, _ = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    train_seqs = torch.cat(train_raw.seqs, dim=0)
    train_coords = torch.cat(train_raw.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)
    scalers = {"feature_scaler": feature_scaler, "coord_scaler": coord_scaler}

    # --- Boundary Condition Detection ---
    bc = None
    if getattr(args, "use_hard_bc", False):
        bc = BoundaryCondition()
        bc.fit(train_raw, feature_scaler,
               velocity_channels=list(range(args.spatial_dim)),
               velocity_threshold=args.velocity_threshold)
        scalers["boundary_condition"] = bc

    train_dataset = ScaledCFDataset(train_raw, feature_scaler=feature_scaler,
                                    coord_scaler=coord_scaler)
    val_dataset = ScaledCFDataset(val_raw, feature_scaler=feature_scaler,
                                   coord_scaler=coord_scaler)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Model Initialization ---
    logger.info(f"instantiating model: {hue.b}{args.model_type}{hue.q} "
                f"with trainer: {hue.b}{args.trainer_type}{hue.q}...")
    model = _build_model(args)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    # --- Training ---
    trainer = _build_trainer(args, model, scalers, output_dir, boundary_condition=bc)
    trainer.fit(train_loader, val_loader)


# ======================================================================
# 5. Inference Pipeline
# ======================================================================

def inference_pipeline(args: argparse.Namespace) -> None:
    """Execute the inference workflow using artifacts from the training phase.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    run_dir = Path(args.output_dir)
    model_path = run_dir / "ckpt.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"ckpt.pt not found at {model_path}.")

    # --- Restore State ---
    logger.info("loading training artifacts...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    feature_scaler = StandardScalerTensor()
    feature_scaler.load_state_dict(checkpoint["scaler_state_dict"]["feature_scaler"])
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(checkpoint["scaler_state_dict"]["coord_scaler"])

    # --- Restore Boundary Condition (if saved) ---
    bc = None
    scaler_state = checkpoint["scaler_state_dict"]
    if "boundary_condition" in scaler_state:
        bc = BoundaryCondition()
        bc.load_state_dict(scaler_state["boundary_condition"])
        logger.info(f"boundary condition restored: "
                    f"{hue.m}{int(bc.wall_mask.sum())}{hue.q} wall nodes")

    # --- Data Preparation ---
    _, _, test_raw = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    test_dataset = ScaledCFDataset(
        test_raw, feature_scaler=feature_scaler, coord_scaler=coord_scaler
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Model Restoration ---
    model = _build_model(args)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- Inference ---
    logger.info(f'{hue.g}running inference on test set...{hue.q}')
    visualizer = FlowVis(output_dir=run_dir, spatial_dim=args.spatial_dim)

    case_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    with torch.no_grad():
        for i, (seq_std, coords_norm, _, _) in enumerate(test_loader):
            seq_std = seq_std.to(device)
            coords_norm = coords_norm.to(device)

            case_name = test_raw.case_names[i]
            steps = seq_std.shape[1] - 1
            initial_state = seq_std[:, 0]
            coords_raw = test_raw.coords[i].cpu()

            pred_seq_std = model.predict(initial_state, coords_norm, steps,
                                         boundary_condition=bc)

            pred_seq = feature_scaler.inverse_transform(pred_seq_std).cpu().squeeze(0)
            gt_seq = feature_scaler.inverse_transform(seq_std).cpu().squeeze(0)

            metrics_evaluator = Metrics(channel_names=args.channel_names)
            metrics = metrics_evaluator.compute(pred_seq, gt_seq)
            case_metrics[case_name] = metrics

            log_metrics = []
            for ch in args.channel_names:
                nmse = metrics[ch]["global"]["nmse"]
                r2 = metrics[ch]["global"]["r2"]
                log_metrics.append(
                    f"{hue.c}{ch}:{hue.q} NMSE={hue.m}{nmse:.2e}{hue.q}, R2={hue.m}{r2:.4f}{hue.q}"
                )

            logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(log_metrics))

            torch.save(pred_seq, run_dir / f"{case_name}_pred.pt")

            logger.info(f"rendering animation for case: {hue.b}{case_name}{hue.q}")
            visualizer.animate_comparison(
                gt=gt_seq, pred=pred_seq, coords=coords_raw, case_name=case_name)

            # --- Per-case static paper figures ---
            logger.info(f"generating paper figures for case: {hue.b}{case_name}{hue.q}")
            plot_rollout_error(
                pred=pred_seq, gt=gt_seq,
                channel_names=args.channel_names,
                output_path=str(run_dir / f"{case_name}_rollout_error.png"),
            )

            n_steps = pred_seq.shape[0]
            for t_idx, t_label in [(0, "first"), (n_steps // 2, "mid"), (n_steps - 1, "last")]:
                plot_error_heatmap(
                    gt=gt_seq, pred=pred_seq, coords=coords_raw,
                    timestep=t_idx, channel_names=args.channel_names,
                    output_path=str(run_dir / f"{case_name}_error_t{t_idx}_{t_label}.png"),
                )

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(case_metrics, f, indent=4)

    # --- Post-inference aggregated plots ---
    history_path = run_dir / "history.json"
    if history_path.exists():
        plot_training_curves(
            history_paths={args.model_type: str(history_path)},
            output_path=str(run_dir / "training_curve.png"),
        )

    plot_metrics_comparison(
        metrics_paths={args.model_type: str(run_dir / "test_metrics.json")},
        output_path=str(run_dir / "metrics_comparison.png"),
    )

    logger.info(f"{hue.g}inference and paper figures completed.{hue.q}")


# ======================================================================
# 6. Probe Pipeline (OOM Pre-Flight Check)
# ======================================================================

def probe_pipeline(args: argparse.Namespace) -> None:
    """Run one full-rollout training step on real data to check peak VRAM usage.

    Reports SAFE / WARNING / CRITICAL verdict based on peak memory utilization.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        logger.warning("No CUDA device — probe skipped (CPU has no OOM risk).")
        return

    # --- Data Preparation ---
    logger.info("loading training data for probe...")
    train_raw, _, _ = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    train_seqs = torch.cat(train_raw.seqs, dim=0)
    train_coords = torch.cat(train_raw.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)

    probe_dataset = ScaledCFDataset(
        train_raw, feature_scaler=feature_scaler, coord_scaler=coord_scaler
    )
    probe_loader = DataLoader(probe_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    seq_std, coords_norm, start_t_norm, dt_norm = next(iter(probe_loader))
    seq_std = seq_std.to(device)
    coords_norm = coords_norm.to(device)

    B, T, N, C = seq_std.shape

    # --- Model ---
    model = _build_model(args)
    model.to(device).train()

    logger.info(f"{hue.y}probe config:{hue.q} "
                f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, "
                f"nodes={hue.m}{N}{hue.q}, channels={hue.m}{C}{hue.q}, "
                f"max_rollout={hue.m}{args.max_rollout_steps}{hue.q}, "
                f"model={hue.b}{args.model_type}{hue.q}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = NMSECriterion(channel_weights=args.channel_weights)

    # --- Forward / Backward ---
    torch.cuda.reset_peak_memory_stats(device)

    input_state = seq_std[:, 0]
    loss = torch.tensor(0.0, device=device)
    k = args.max_rollout_steps
    total_weight = k * (k + 1)
    for t in range(k):
        if hasattr(model, "time_encoder") and model.time_encoder is not None:
            t_norm = start_t_norm.to(device) + t * dt_norm.to(device)
            pred = model(input_state, coords_norm, t_norm=t_norm)
        else:
            pred = model(input_state, coords_norm)
        w_t = 2.0 * (t + 1) / total_weight
        loss = loss + w_t * criterion(pred, seq_std[:, t + 1])
        input_state = pred
    loss.backward()
    optimizer.step()

    # --- VRAM Report ---
    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    pct = 100 * peak / total
    if pct < 75:
        status = f"{hue.g}SAFE{hue.q}"
    elif pct < 92:
        status = f"{hue.y}WARNING — close to limit{hue.q}"
    else:
        status = f"{hue.r}CRITICAL — likely OOM in real training{hue.q}"

    logger.info(f"{hue.y}device: {hue.b}{torch.cuda.get_device_name(device)}{hue.q}  "
                f"({hue.m}{total / 1e9:.1f}{hue.q} GB)")
    logger.info(f"peak usage: {hue.m}{peak / 1e9:.2f}{hue.q} GB  "
                f"({hue.m}{pct:.1f}{hue.q} %)  →  {status}")
    logger.info(f"{hue.g}probe completed.{hue.q}\n")


# ======================================================================
# 7. Main Execution
# ======================================================================

if __name__ == "__main__":
    args = flow_config.get_args()

    if "probe" in args.mode: probe_pipeline(args)
    if "train" in args.mode: train_pipeline(args)
    if "infer" in args.mode: inference_pipeline(args)
