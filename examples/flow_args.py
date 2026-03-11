# Argument Configuration for STMFormer Training and Inference
# Author: Shengning Wang

import argparse
import torch


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the flow simulation pipeline.

    Supports three model architectures (STMFormer, GeoFNO, Transolver) and
    two training strategies (rollout with curriculum, teacher forcing baseline).

    Returns:
        argparse.Namespace: Parsed arguments containing all hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="STMFormer: Spatio-Temporal Mesh Transformer for Flow Simulation",
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
        "--mode", type=str, default="train_infer_probe",
        help="Execution mode. Combine: train, infer, probe (e.g., 'train_infer').")
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
        "--win_len", type=int, default=8,
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
        "--model_type", type=str, default="stmformer",
        choices=["stmformer", "geofno", "transolver"],
        help="Neural operator architecture.")
    model.add_argument(
        "--trainer_type", type=str, default="rollout",
        choices=["rollout", "teacher_forcing"],
        help="Training strategy. 'rollout': curriculum + noise (STMFormer). "
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
    # 5. STMFormer-Specific Parameters
    # ==================================================================
    stm = parser.add_argument_group("STMFormer")
    stm.add_argument(
        "--num_slices", type=int, default=32,
        help="Number of mesh slice tokens (M). Higher M captures more physics modes.")
    stm.add_argument(
        "--num_heads", type=int, default=8,
        help="Number of attention heads for slice-space MHA.")

    # Ablation switches
    stm.add_argument(
        "--use_spatial_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable RFF spatial encoding. Disable for ablation (--no-use_spatial_encoding).")
    stm.add_argument(
        "--use_temporal_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable sinusoidal temporal encoding. Disable for ablation.")
    stm.add_argument(
        "--use_boundary_padding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable boundary ghost-node padding for improved boundary accuracy.")
    stm.add_argument(
        "--padding_ratio", type=float, default=0.05,
        help="Ghost node offset as fraction of domain extent.")

    # Spatial encoding
    stm.add_argument(
        "--coord_features", type=int, default=8,
        help="RFF half-dimension (output: 2 * coord_features). Set 0 for raw coords.")
    stm.add_argument(
        "--coord_sigma", type=float, default=1.0,
        help="RFF projection scale controlling spatial frequency bandwidth.")

    # Temporal encoding
    stm.add_argument(
        "--time_features", type=int, default=4,
        help="Sinusoidal PE half-dimension (output: 2 * time_features).")
    stm.add_argument(
        "--max_steps", type=int, default=1000,
        help="Reference max time step for sinusoidal frequency scaling.")

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
    # 7. Transolver-Specific Parameters (Baseline)
    # ==================================================================
    tsv = parser.add_argument_group("Transolver (Baseline)")
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
        "--max_epochs", type=int, default=300,
        help="Maximum number of training epochs.")
    optim.add_argument(
        "--eta_min", type=float, default=1e-6,
        help="Minimum learning rate for cosine annealing scheduler.")

    # ==================================================================
    # 9. Curriculum Learning (Rollout Trainer Only)
    # ==================================================================
    curriculum = parser.add_argument_group("Curriculum (Rollout Trainer)")
    curriculum.add_argument(
        "--max_rollout_steps", type=int, default=7,
        help="Maximum autoregressive rollout steps (curriculum ceiling).")
    curriculum.add_argument(
        "--rollout_patience", type=int, default=40,
        help="Epochs between curriculum difficulty advances.")
    curriculum.add_argument(
        "--noise_std_init", type=float, default=0.01,
        help="Initial std of Gaussian noise injected into input state.")
    curriculum.add_argument(
        "--noise_decay", type=float, default=0.7,
        help="Multiplicative decay for noise std at each curriculum advance.")

    args = parser.parse_args()
    return args
