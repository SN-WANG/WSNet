# MFS-MLS: Multi-Fidelity Surrogate Model based on Moving Least Squares
# Paper reference: https://doi.org/10.1007/s00158-021-03044-5
# Paper author: Shuo Wang, Yin Liu, Qi Zhou, Yongliang Yuan, Liye Lv, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from typing import Dict, Optional
from itertools import combinations_with_replacement

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class MFSMLS:
    """
    Adaptive Multi-Fidelity Surrogate Model based on Moving Least Squares (MFS-MLS).

    Fuses Low-Fidelity (LF) and High-Fidelity (HF) data using the formulation:
    y_hf(x) = rho(x) * y_lf(x) + d(x), where d(x) is approximated via polynomial basis
    and rho(x) is implicitly modeled through the augmented basis.

    The model employs Weighted Least Squares with an adaptive influence domain
    based on normalized distances to HF training samples.

    Attributes:
        lf_model (RBF): Trained low-fidelity surrogate model.
        poly_degree (int): Degree of polynomial basis for discrepancy (1=linear, 2=quadratic).
        scaler_x (StandardScalerNP): Normalization processor for input features.
        scaler_y (StandardScalerNP): Normalization processor for target responses.
        x_hf_train_ (np.ndarray): Stored high-fidelity training inputs.
            Shape: (num_hf_samples, input_dim), dtype: float64.
        y_hf_train_ (np.ndarray): Stored high-fidelity training targets.
            Shape: (num_hf_samples, target_dim), dtype: float64.
        p_train_ (np.ndarray): Pre-computed augmented basis matrix for HF inputs.
            Shape: (num_hf_samples, num_basis), dtype: float64.
        d_max_ (float): Maximum pairwise distance among HF samples, used as influence radius.
        is_fitted (bool): Status flag indicating if the model has been trained.
    """

    def __init__(self, lf_model_params: Optional[Dict] = None, poly_degree: int = 2) -> None:
        """
        Initialize the MFS-MLS model configuration.

        Args:
            lf_model_params (Optional[Dict]): Dictionary of kwargs to pass to RBF class.
                If None, default RBF parameters are used.
            poly_degree (int): Degree of the polynomial basis function.
                1 for linear, 2 for quadratic. Default is 2.
        """
        # parameters
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)

        self.poly_degree = poly_degree

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.p_train_: Optional[np.ndarray] = None
        self.d_max_: float = 0.0
        self.is_fitted: bool = False

    # ------------------------------------------------------------------

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances between two point sets.

        Uses the algebraic expansion: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c^T
        for vectorized computation without explicit loops.

        Args:
            x (np.ndarray): Query points of shape (num_queries, num_features), dtype: float64.
            c (np.ndarray): Reference points of shape (num_refs, num_features), dtype: float64.

        Returns:
            np.ndarray: Squared distance matrix of shape (num_queries, num_refs), dtype: float64.
                Entry (i, j) contains the squared Euclidean distance between x[i] and c[j].
        """
        # Compute squared L2 norms for each point set: (num_queries, 1) and (num_refs,)
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)  # Shape: (num_queries, 1)
        c_norm_sq = np.sum(c ** 2, axis=1)                 # Shape: (num_refs,)

        # Compute cross term: 2 * x @ c^T, shape: (num_queries, num_refs)
        cross_term = 2.0 * (x @ c.T)

        # Combine: ||x||^2 + ||c||^2 - 2 * x @ c^T
        # Broadcasting: (num_queries, 1) + (num_refs,) -> (num_queries, num_refs)
        dists_sq = x_norm_sq + c_norm_sq - cross_term

        # Numerical stability: clamp small negative values to zero (floating point errors)
        np.maximum(dists_sq, 0.0, out=dists_sq)

        return dists_sq

    # ------------------------------------------------------------------

    def _generate_polynomial_powers(self, input_dim: int) -> np.ndarray:
        """
        Generate exponent combinations for polynomial basis expansion.

        Constructs all monomials up to the specified degree using combinations
        with replacement. For example, degree 2 with 2 features generates:
        [1, x1, x2, x1^2, x1*x2, x2^2].

        Args:
            input_dim (int): Number of input features (dimensionality).

        Returns:
            np.ndarray: Exponent matrix of shape (num_terms, input_dim), dtype: int64.
                Each row represents the exponents for one polynomial term.
        """
        powers = []
        for degree in range(self.poly_degree + 1):
            for combo in combinations_with_replacement(range(input_dim), degree):
                power_vec = np.zeros(input_dim, dtype=np.int64)
                for idx in combo:
                    power_vec[idx] += 1
                powers.append(power_vec)
        return np.stack(powers, axis=0)

    # ------------------------------------------------------------------

    def _build_polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """
        Construct polynomial feature matrix from input data.

        Args:
            x (np.ndarray): Input features of shape (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Polynomial feature matrix of shape (num_samples, num_terms), dtype: float64.
                Includes bias term (all ones) as the first column.
        """
        num_samples, input_dim = x.shape
        powers = self._generate_polynomial_powers(input_dim)
        num_terms = powers.shape[0]

        phi = np.ones((num_samples, num_terms), dtype=x.dtype)

        for dim in range(input_dim):
            exp_dim = powers[:, dim]
            mask = exp_dim > 0
            if not np.any(mask):
                continue
            phi[:, mask] *= np.power(x[:, dim:dim+1], exp_dim[mask])

        return phi

    # ------------------------------------------------------------------

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Perform model training.

        The training process involves:
        1. Training the LF surrogate model on raw LF data.
        2. Normalizing HF inputs and targets using StandardScalerNP.
        3. Constructing augmented basis P = [y_lf(x_hf), 1, x, x^2, ...].
        4. Computing adaptive influence radius d_max from HF sample distribution.

        Args:
            x_lf (np.ndarray): Low-fidelity inputs of shape (num_lf_samples, input_dim).
            y_lf (np.ndarray): Low-fidelity targets of shape (num_lf_samples, target_dim).
            x_hf (np.ndarray): High-fidelity inputs of shape (num_hf_samples, input_dim).
            y_hf (np.ndarray): High-fidelity targets of shape (num_hf_samples, target_dim).
        """
        # Step 1: Train LF model on raw data
        self.lf_model.fit(x_lf, y_lf)

        # Step 2: Scale HF data
        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        # Step 3: Construct augmented basis matrix P = [y_lf(x_hf), poly_basis(x_hf)]
        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple):  y_lf_at_hf = y_lf_at_hf[0]  # Prevent KRG's var_pred

        y_lf_at_hf_scaled = self.scaler_y.transform(y_lf_at_hf)

        poly_basis = self._build_polynomial_features(self.x_hf_train_)
        self.p_train_ = np.concatenate([y_lf_at_hf_scaled, poly_basis], axis=1)

        # Step 4: Calculate adaptive influence radius d_max (max distance in HF set)
        dist_matrix = self._compute_dists(self.x_hf_train_, self.x_hf_train_)
        self.d_max_ = np.max(dist_matrix)

        self.is_fitted = True

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction.

        For each query point, solves a local Weighted Least Squares problem:
        A(x) = (P^T * W(x) * P)^{-1} * P^T * W(x) * Y
        where W(x) is a diagonal weight matrix based on normalized distances.

        Args:
            x_pred (np.ndarray): Prediction inputs of shape (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets of shape (num_samples, target_dim), dtype: float 64.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        num_samples = x_pred.shape[0]
        num_hf_samples, num_basis = self.p_train_.shape
        target_dim = self.y_hf_train_.shape[1]

        # Step 1: Scale pred inputs
        x_pred_scaled = self.scaler_x.transform(x_pred)

        # Step 2: Prepare basis for query points
        y_lf_at_pred = self.lf_model.predict(x_pred)
        if isinstance(y_lf_at_pred, tuple):  y_lf_at_pred = y_lf_at_pred[0]  # Prevent KRG's var_pred

        y_lf_at_pred_scaled = self.scaler_y.transform(y_lf_at_pred)

        poly_basis_pred = self._build_polynomial_features(x_pred_scaled)
        p_pred = np.concatenate([y_lf_at_pred_scaled, poly_basis_pred], axis=1)

        # Step 3: Pre-calculate distances to all HF samples
        dists = self._compute_dists(x_pred_scaled, self.x_hf_train_)

        y_pred_scaled = np.zeros((num_samples, target_dim), dtype=np.float64)

        # Step 4: MLS logic per pred point
        for i in range(num_samples):
            # Normalized distance: d_norm = dist / d_max
            di = dists[i] / (self.d_max_ + 1e-12)

            # Weighting function: cubic spline with local support
            # w(d) = 1 - 6*d^2 + 8*d^3 - 3*d^4 for d <= 1, else 0
            wi = np.zeros(num_hf_samples, dtype=np.float64)
            mask = di <= 1.0
            wi[mask] = 1.0 - 6.0 * di[mask]**2 + 8.0 * di[mask]**3 - 3.0 * di[mask]**4

            W = np.diag(wi)

            # Weighted Least Squares system
            # LHS: (num_basis, num_basis), RHS: (num_basis, target_dim)
            lhs = self.p_train_.T @ W @ self.p_train_
            rhs = self.p_train_.T @ W @ self.y_hf_train_

            # Solve for coefficients A (num_basis, target_dim)
            try:
                coeffs = np.linalg.solve(lhs + 1e-8 * np.eye(num_basis), rhs)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                coeffs = np.linalg.pinv(lhs) @ rhs

            # Prediction: p(x_pred) * A -> (1, num_basis) @ (num_basis, target_dim)
            y_pred_scaled[i, :] = p_pred[i:i+1, :] @ coeffs

        # Step 5: Inverse scaling to original space
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred
