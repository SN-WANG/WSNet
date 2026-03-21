# Argument Configuration for HyperFlowNet Training and Inference
# Author: Shengning Wang

import argparse
import torch


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the flow simulation pipeline.

    Supports three model architectures (HyperFlowNet, GeoFNO, Transolver) and
    two training strategies (rollout with curriculum, teacher forcing baseline).

    Returns:
        argparse.Namespace: Parsed arguments containing all hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="HyperFlowNet: A Spatio-Temporal Neural Operator for Flow Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==================================================================
    # 1. General Settings
    # ==================================================================
    general = parser.add_argument_group("General")
    general.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.")
    general.add_argument(
        "--output_dir", type=str, default="./runs",
        help="Directory to save checkpoints, logs, and visualizations.")
    general.add_argument(
        "--mode", type=str, nargs='+', default=["train", "infer", "probe"],
        choices=["train", "infer", "probe"],
        help="Execution phases to run (space-separated, e.g., --mode train infer).")
    general.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cuda, cpu, or cuda:N).")

    # ==================================================================
    # 2. Data Configuration
    # ==================================================================
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="Path to directory containing simulation case folders.")
    data.add_argument(
        "--channel_names", type=list, default=["Vx", "Vy", "P", "T"],
        help="Ordered list of output channel names.")
    data.add_argument(
        "--spatial_dim", type=int, default=2, choices=[2, 3],
        help="Spatial dimensionality of the mesh (2D or 3D).")
    data.add_argument(
        "--win_len", type=int, default=16,
        help="Temporal window length for sequence slicing (input + target).")
    data.add_argument(
        "--win_stride", type=int, default=1,
        help="Stride for sliding window data augmentation.")
    data.add_argument(
        "--batch_size", type=int, default=8,
        help="Mini-batch size for training and validation.")
    data.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader worker subprocesses.")

    # ==================================================================
    # 3. Model Selection
    # ==================================================================
    model = parser.add_argument_group("Model Selection")
    model.add_argument(
        "--model_type", type=str, default="hyperflownet",
        choices=["hyperflownet", "geofno", "transolver"],
        help="Neural operator architecture.")
    model.add_argument(
        "--trainer_type", type=str, default="rollout",
        choices=["rollout", "teacher_forcing"],
        help="Training strategy. 'rollout': curriculum + noise (HyperFlowNet). "
             "'teacher_forcing': GT-input baseline (GeoFNO/Transolver).")

    # ==================================================================
    # 4. Model Architecture — Common
    # ==================================================================
    arch = parser.add_argument_group("Architecture (Common)")
    arch.add_argument(
        "--depth", type=int, default=4,
        help="Number of stacked transformer/operator blocks.")
    arch.add_argument(
        "--width", type=int, default=128,
        help="Hidden channel dimension.")

    # ==================================================================
    # 5. HyperFlowNet-Specific Parameters
    # ==================================================================
    hfn = parser.add_argument_group("HyperFlowNet")
    hfn.add_argument(
        "--num_slices", type=int, default=32,
        help="Number of mesh slice tokens (M). Higher M captures more physics modes.")
    hfn.add_argument(
        "--num_heads", type=int, default=8,
        help="Number of attention heads for slice-space MHA.")

    # Ablation switches
    hfn.add_argument(
        "--use_spatial_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable LFF spatial encoding. Disable for ablation (--no-use_spatial_encoding).")
    hfn.add_argument(
        "--use_temporal_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable sinusoidal temporal encoding. Disable for ablation.")

    # Boundary condition enforcement
    hfn.add_argument(
        "--use_hard_bc", action=argparse.BooleanOptionalAction, default=True,
        help="Enable hard boundary condition enforcement during rollout. "
             "Replaces wall-node velocity predictions with known no-slip values.")
    hfn.add_argument(
        "--velocity_threshold", type=float, default=1e-4,
        help="Velocity magnitude threshold for data-driven wall node detection.")

    # Spatial encoding
    hfn.add_argument(
        "--coord_features", type=int, default=8,
        help="LFF half-dimension (output: 2 * coord_features). Set 0 for raw coords.")

    # Temporal encoding
    hfn.add_argument(
        "--time_features", type=int, default=4,
        help="Sinusoidal PE half-dimension (output: 2 * time_features).")
    hfn.add_argument(
        "--freq_base", type=int, default=1000,
        help="Base for sinusoidal frequency decay (analogous to 10000 in Transformer PE).")

    # ==================================================================
    # 6. GeoFNO-Specific Parameters
    # ==================================================================
    fno = parser.add_argument_group("GeoFNO")
    fno.add_argument(
        "--modes", type=int, nargs='+', default=[12, 12],
        help="Number of retained Fourier modes per spatial dimension.")
    fno.add_argument(
        "--latent_grid_size", type=int, nargs='+', default=[64, 64],
        help="Resolution of the latent FFT grid.")
    fno.add_argument(
        "--deform_layers", type=int, default=2,
        help="Number of layers in the coordinate deformation MLP.")
    fno.add_argument(
        "--deform_hidden", type=int, default=32,
        help="Hidden dimension of the deformation MLP.")

    # ==================================================================
    # 7. Transolver-Specific Parameters
    # ==================================================================
    tsv = parser.add_argument_group("Transolver")
    tsv.add_argument(
        "--tsv_num_slices", type=int, default=32,
        help="Number of physics slice tokens for original Transolver.")
    tsv.add_argument(
        "--tsv_num_heads", type=int, default=8,
        help="Number of attention heads for original Transolver.")
    tsv.add_argument(
        "--mlp_ratio", type=int, default=1,
        help="MLP hidden size multiplier (hidden = width * mlp_ratio).")
    tsv.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate in Transolver attention and MLP.")

    # ==================================================================
    # 8. Optimization
    # ==================================================================
    optim = parser.add_argument_group("Optimization")
    optim.add_argument(
        "--lr", type=float, default=5e-4,
        help="Initial learning rate for AdamW optimizer.")
    optim.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="L2 regularization coefficient for AdamW.")
    optim.add_argument(
        "--max_epochs", type=int, default=700,
        help="Maximum number of training epochs.")
    optim.add_argument(
        "--eta_min", type=float, default=1e-6,
        help="Minimum learning rate for cosine annealing scheduler.")
    optim.add_argument(
        "--channel_weights", type=float, nargs='+', default=[1.0, 3.0, 1.0, 1.0],
        help="Per-channel NMSE loss weights for [Vx, Vy, P, T]. "
             "Default: Vy 3x weighted to improve Y-velocity prediction.")

    # ==================================================================
    # 9. Curriculum Learning
    # ==================================================================
    curriculum = parser.add_argument_group("Curriculum (Rollout Trainer)")
    curriculum.add_argument(
        "--max_rollout_steps", type=int, default=15,
        help="Maximum autoregressive rollout steps (curriculum ceiling).")
    curriculum.add_argument(
        "--rollout_patience", type=int, default=45,
        help="Epochs between curriculum difficulty advances.")
    curriculum.add_argument(
        "--noise_std_init", type=float, default=0.01,
        help="Initial std of Gaussian noise injected into input state.")
    curriculum.add_argument(
        "--noise_decay", type=float, default=0.7,
        help="Multiplicative decay for noise std at each curriculum advance.")

    args = parser.parse_args()
    return args
