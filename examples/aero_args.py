# Args Config for Aero Optimization Solver Benchmark Platform
# Author: Shengning Wang

import argparse
import numpy as np


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the aero optimization benchmark.

    Returns:
        argparse.Namespace: Parsed arguments with all hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Aero Optimization Solver Benchmark Platform"
    )

    # ----------------------------------------------------------------------
    # 1. General Settings
    # ----------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory for cached data and plot outputs.")
    parser.add_argument("--demos", type=str, nargs='+', default=["all"],
                        help="Demos to run: A B C D E F G H I J, or group aliases "
                             "(all, ensemble, multifidelity, sequential, optimization).")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Enable matplotlib visualization (saved to save_dir).")

    # ----------------------------------------------------------------------
    # 2. Problem Definition
    # ----------------------------------------------------------------------
    parser.add_argument("--num_features", type=int, default=3,
                        help="Number of design variables (thick1, thick2, thick3).")
    parser.add_argument("--num_outputs", type=int, default=4,
                        help="Number of simulation outputs.")
    parser.add_argument("--bounds_lower", type=float, nargs='+', default=[4.0, 4.0, 4.0],
                        help="Lower bounds for each design variable.")
    parser.add_argument("--bounds_upper", type=float, nargs='+', default=[10.0, 10.0, 10.0],
                        help="Upper bounds for each design variable.")
    parser.add_argument("--input_names", type=str, nargs='+',
                        default=["thick1", "thick2", "thick3"],
                        help="Names of design variables.")
    parser.add_argument("--output_names", type=str, nargs='+',
                        default=["weight", "displacement", "stress_skin", "stress_stiff"],
                        help="Names of simulation outputs.")

    # ----------------------------------------------------------------------
    # 3. Sampling Configuration
    # ----------------------------------------------------------------------
    parser.add_argument("--num_train", type=int, default=20,
                        help="Number of high-fidelity training samples.")
    parser.add_argument("--num_test", type=int, default=10,
                        help="Number of test samples.")
    parser.add_argument("--num_lf", type=int, default=30,
                        help="Number of low-fidelity samples (multi-fidelity).")
    parser.add_argument("--num_hf", type=int, default=10,
                        help="Number of high-fidelity samples (multi-fidelity).")
    parser.add_argument("--num_infill", type=int, default=5,
                        help="Number of sequential infill iterations.")
    parser.add_argument("--lhs_iterations", type=int, default=50,
                        help="Maximin LHS optimization iterations.")

    # ----------------------------------------------------------------------
    # 4. Model Hyperparameters
    # ----------------------------------------------------------------------

    # Kriging (KRG)
    parser.add_argument("--krg_poly", type=str, default="constant",
                        help="KRG regression type (constant, linear, quadratic).")
    parser.add_argument("--krg_kernel", type=str, default="gaussian",
                        help="KRG correlation kernel (gaussian, exponential, cubic, etc.).")
    parser.add_argument("--krg_theta0", type=float, default=1.0,
                        help="KRG initial correlation parameter.")
    parser.add_argument("--krg_theta_bounds", type=float, nargs=2, default=[1e-6, 100.0],
                        help="KRG theta optimization bounds [lower, upper].")

    # Ensemble
    parser.add_argument("--ensemble_threshold", type=float, default=0.5,
                        help="Filtering threshold for TAHS/AESMSI model selection.")

    # Multi-fidelity
    parser.add_argument("--mf_poly_degree", type=int, default=2,
                        help="Polynomial degree for MFS-MLS discrepancy model.")
    parser.add_argument("--mf_sigma_bounds", type=float, nargs=2, default=[0.01, 10.0],
                        help="Sigma bounds for MMFS RBF shape parameter.")

    # Infill / Sequential Sampling
    parser.add_argument("--infill_criterion", type=str, default="ei",
                        choices=["ei", "poi", "lcb", "mse"],
                        help="Acquisition function for single-objective infill.")
    parser.add_argument("--mico_ratio", type=float, default=0.5,
                        help="MICO mutual-information vs distance trade-off ratio.")

    # ----------------------------------------------------------------------
    # 5. Optimization Parameters
    # ----------------------------------------------------------------------
    parser.add_argument("--opt_single_idx", type=int, default=2,
                        help="Output index for single-objective optimization (stress_skin).")
    parser.add_argument("--obj_indices", type=int, nargs='+', default=[2, 3],
                        help="Output indices for multi-objective optimization.")
    parser.add_argument("--constraint_indices", type=int, nargs='+', default=[0],
                        help="Output indices used as constraints.")
    parser.add_argument("--constraint_percentile", type=float, default=50.0,
                        help="Percentile of training data for constraint upper bound.")
    parser.add_argument("--de_popsize", type=int, default=10,
                        help="Differential evolution population size.")
    parser.add_argument("--de_maxiter", type=int, default=50,
                        help="Differential evolution max iterations.")
    parser.add_argument("--df_popsize", type=int, default=20,
                        help="Dragonfly (CFSSDA) population size.")
    parser.add_argument("--df_maxiter", type=int, default=120,
                        help="Dragonfly (CFSSDA) max iterations.")
    parser.add_argument("--opt_tol", type=float, default=1e-6,
                        help="Optimizer convergence tolerance.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Post-parse processing
    # ------------------------------------------------------------------

    # Construct bounds array: shape (num_features, 2)
    args.bounds = np.array(
        list(zip(args.bounds_lower, args.bounds_upper)), dtype=np.float64
    )

    # Pack KRG params into a convenience dict
    args.krg_params = {
        "poly": args.krg_poly,
        "kernel": args.krg_kernel,
        "theta0": args.krg_theta0,
        "theta_bounds": tuple(args.krg_theta_bounds),
    }

    # Expand demo group aliases
    all_demos = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    alias_map = {
        "all": all_demos,
        "ensemble": ["A", "B"],
        "multifidelity": ["C", "D", "E"],
        "sequential": ["F", "G", "H"],
        "optimization": ["I", "J"],
    }
    expanded = []
    for d in args.demos:
        key = d.lower()
        if key in alias_map:
            expanded.extend(alias_map[key])
        else:
            expanded.append(d.upper())
    args.demos = sorted(set(expanded), key=lambda x: all_demos.index(x))

    return args
