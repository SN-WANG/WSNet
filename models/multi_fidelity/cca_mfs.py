# CCA-MFS: Multi-Fidelity Surrogate Model Based on Canonical Correlation Analysis and Least Squares
# Paper reference: https://doi.org/10.1115/1.4047686
# Paper author: Liye Lv, Chaoyang Zong, Chao Zhang, Xueguan Song, Wei Sun
# Code author: Shengning Wang

import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.optimize import minimize, Bounds
from typing import Dict, Optional, Tuple

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class CCAMFS:
    """
    Multi-Fidelity Surrogate Model based on Canonical Correlation Analysis and Least Squares.

    The CCA-MFS model fuses Low-Fidelity (LF) and High-Fidelity (HF) data using a three-stage approach:
    1. CCA Stage: Construct transition matrices U and V to maximize correlation between HF and LF 
       sample matrices Ph = [Xh, yh] and Pl = [Xl, yl].
    2. Discrepancy Stage: Build RBF-based discrepancy function in the CCA-transformed canonical space.
    3. Optimization Stage: Determine optimal scaling factor rho and weight matrices W1, W2 via 
       Least Squares minimization.

    The prediction formula is:
        y_hf(x) = rho * y_lf(x) + Rh_ts @ W1 + Rl_ts @ W2
    where Rh_ts and Rl_ts are correlation matrices (Euclidean distances) in the CCA-transformed space,
    and d(x) = Rh_ts @ W1 + Rl_ts @ W2 is the discrepancy function.

    Attributes:
        lf_model (RBF): Trained low-fidelity surrogate model.
        scaler_x (StandardScalerNP): Normalization processor for input features.
        scaler_y (StandardScalerNP): Normalization processor for target responses.
        U_ (np.ndarray): CCA transition matrix for HF samples.
            Shape: (ndv, ndv), dtype: float64, where ndv = input_dim + target_dim.
        V_ (np.ndarray): CCA transition matrix for LF samples.
            Shape: (ndv, ndv), dtype: float64.
        x_hf_train_ (np.ndarray): Stored high-fidelity training inputs (scaled).
            Shape: (num_hf_samples, input_dim), dtype: float64.
        y_hf_train_ (np.ndarray): Stored high-fidelity training targets (scaled).
            Shape: (num_hf_samples, target_dim), dtype: float64.
        y_lf_at_hf_ (np.ndarray): LF model predictions at HF input locations (scaled).
            Shape: (num_hf_samples, target_dim), dtype: float64.
        Ph_transformed_ (np.ndarray): HF samples transformed to CCA space.
            Shape: (num_hf_samples, ndv), dtype: float64.
        Pl_transformed_ (np.ndarray): LF samples transformed to CCA space.
            Shape: (num_lf_samples, ndv), dtype: float64.
        Rh_ (np.ndarray): Distance matrix between HF samples in CCA space.
            Shape: (num_hf_samples, num_hf_samples), dtype: float64.
        Rhl_ (np.ndarray): Cross-distance matrix between HF and LF samples in CCA space.
            Shape: (num_hf_samples, num_lf_samples), dtype: float64.
        rho_ (np.ndarray): Optimal scaling factors for each target dimension.
            Shape: (target_dim,), dtype: float64.
        W1_ (np.ndarray): Weight matrix for HF correlations.
            Shape: (num_hf_samples, target_dim), dtype: float64.
        W2_ (np.ndarray): Weight matrix for LF correlations.
            Shape: (num_lf_samples, target_dim), dtype: float64.
        is_fitted (bool): Status flag indicating if the model has been trained.
    """

    def __init__(self, lf_model_params: Optional[Dict] = None) -> None:
        """
        Initialize the CCA-MFS model configuration.

        Args:
            lf_model_params (Optional[Dict]): Dictionary of kwargs to pass to RBF class for the LF model.
                If None, default RBF parameters are used.
        """
        # Parameters
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)

        # Scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # Model state
        self.U_: Optional[np.ndarray] = None
        self.V_: Optional[np.ndarray] = None
        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.y_lf_at_hf_: Optional[np.ndarray] = None
        self.Ph_transformed_: Optional[np.ndarray] = None
        self.Pl_transformed_: Optional[np.ndarray] = None
        self.Rh_: Optional[np.ndarray] = None
        self.Rhl_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.W1_: Optional[np.ndarray] = None
        self.W2_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------

    def _compute_covariance_matrices(
        self,
        Ph: np.ndarray,
        Pl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute sample covariance matrices for CCA following the paper's formulation.

        Computes S11 (HF covariance), S22 (LF covariance), and S12 (cross-covariance)
        using the formulas from the paper:
        - S11 = 1/(M-1) * sum((Ph_i - ph_bar)^T @ (Ph_i - ph_bar))
        - S22 = 1/(N-1) * sum((Pl_j - pl_bar)^T @ (Pl_j - pl_bar))
        - S12 = 1/(M*N-1) * sum((Ph_i - ph_bar)^T @ (Pl_j - pl_bar))

        Args:
            Ph (np.ndarray): HF sample matrix [Xh, yh].
                Shape: (num_hf_samples, ndv), dtype: float64, where ndv = input_dim + target_dim.
            Pl (np.ndarray): LF sample matrix [Xl, yl].
                Shape: (num_lf_samples, ndv), dtype: float64.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - S11 (np.ndarray): HF covariance matrix. Shape: (ndv, ndv), dtype: float64.
                - S22 (np.ndarray): LF covariance matrix. Shape: (ndv, ndv), dtype: float64.
                - S12 (np.ndarray): Cross-covariance matrix. Shape: (ndv, ndv), dtype: float64.
        """
        num_hf = Ph.shape[0]
        num_lf = Pl.shape[0]

        # Compute means
        ph_bar = np.mean(Ph, axis=0, keepdims=True)  # Shape: (1, ndv)
        pl_bar = np.mean(Pl, axis=0, keepdims=True)  # Shape: (1, ndv)

        # S11: HF covariance matrix (unbiased estimator)
        Ph_centered = Ph - ph_bar  # Shape: (num_hf, ndv)
        S11 = (Ph_centered.T @ Ph_centered) / (num_hf - 1)  # Shape: (ndv, ndv)

        # S22: LF covariance matrix (unbiased estimator)
        Pl_centered = Pl - pl_bar  # Shape: (num_lf, ndv)
        S22 = (Pl_centered.T @ Pl_centered) / (num_lf - 1)  # Shape: (ndv, ndv)

        # S12: Cross-covariance matrix following paper's formula: 1/(M*N-1) * sum over all pairs
        S12 = (Ph_centered.T @ np.ones((num_hf, num_lf)) @ Pl_centered) / (num_hf * num_lf - 1)
        # Shape: (ndv, ndv)

        return S11, S22, S12

    # ------------------------------------------------------------------

    def _compute_cca_transition_matrices(
        self,
        S11: np.ndarray,
        S22: np.ndarray,
        S12: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CCA transition matrices U and V via SVD.

        Solves the CCA optimization problem:
            maximize u^T @ S12 @ v
            subject to u^T @ S11 @ u = 1, v^T @ S22 @ v = 1

        This is equivalent to performing SVD on the matrix:
            C = S11^(-1/2) @ S12 @ S22^(-1/2)

        The transition matrices are then:
            U = S11^(-1/2) @ L
            V = S22^(-1/2) @ R
        where L and R are the left and right singular vectors of C.

        Args:
            S11 (np.ndarray): HF covariance matrix. Shape: (ndv, ndv), dtype: float64.
            S22 (np.ndarray): LF covariance matrix. Shape: (ndv, ndv), dtype: float64.
            S12 (np.ndarray): Cross-covariance matrix. Shape: (ndv, ndv), dtype: float64.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - U (np.ndarray): HF transition matrix. Shape: (ndv, ndv), dtype: float64.
                - V (np.ndarray): LF transition matrix. Shape: (ndv, ndv), dtype: float64.
        """
        # Compute matrix square roots and their inverses: S11^(-1/2) and S22^(-1/2)
        S11_sqrt = sqrtm(S11).astype(np.float64)
        S22_sqrt = sqrtm(S22).astype(np.float64)
        S11_inv_sqrt = inv(S11_sqrt)
        S22_inv_sqrt = inv(S22_sqrt)

        # Compute C = S11^(-1/2) @ S12 @ S22^(-1/2)
        C = S11_inv_sqrt @ S12 @ S22_inv_sqrt  # Shape: (ndv, ndv)

        # SVD of C: C = L @ S @ R^T
        L, S, R_T = np.linalg.svd(C, full_matrices=True)
        R = R_T.T

        # Compute transition matrices following the paper
        # U = S11^(-1/2) @ L
        U = S11_inv_sqrt @ L  # Shape: (ndv, ndv)

        # V = S22^(-1/2) @ R
        V = S22_inv_sqrt @ R  # Shape: (ndv, ndv)

        return U, V

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

        return np.sqrt(dists_sq)

    # ------------------------------------------------------------------

    def _objective_function(
        self,
        params: np.ndarray,
        y_hf: np.ndarray,
        y_lf_at_hf: np.ndarray,
        Rh: np.ndarray,
        Rhl: np.ndarray,
        num_hf: int
    ) -> float:
        """
        Least squares objective function for parameter optimization.

        Minimizes the combined error:
            J = sum((y_hf_hat - y_hf)^2) + sum((y_hf_hat - y_lf_at_hf)^2)
        where:
            y_hf_hat = rho * y_lf_at_hf + Rh @ W1 + Rhl @ W2

        The first term ensures accuracy against HF data, while the second term
        regularizes the discrepancy to be small when LF predictions are accurate.

        Args:
            params (np.ndarray): Parameter vector [rho, W1, W2] flattened.
                Shape: (1 + num_hf + num_lf,), dtype: float64.
            y_hf (np.ndarray): Actual HF targets for a single output dimension.
                Shape: (num_hf,), dtype: float64.
            y_lf_at_hf (np.ndarray): LF predictions at HF locations.
                Shape: (num_hf,), dtype: float64.
            Rh (np.ndarray): HF distance matrix in CCA space.
                Shape: (num_hf, num_hf), dtype: float64.
            Rhl (np.ndarray): Cross-distance matrix between HF and LF in CCA space.
                Shape: (num_hf, num_lf), dtype: float64.
            num_hf (int): Number of HF samples.

        Returns:
            float: The total error value.
        """
        # Unpack parameters
        rho = params[0]
        W1 = params[1:1 + num_hf]  # Shape: (num_hf,)
        W2 = params[1 + num_hf:]  # Shape: (num_lf,)

        # Predict HF responses
        y_hf_hat = rho * y_lf_at_hf + Rh @ W1 + Rhl @ W2  # Shape: (num_hf,)

        # Error 1: Squared error between predicted and actual HF
        error1 = np.sum((y_hf_hat - y_hf) ** 2)

        # Error 2: Squared error between predicted HF and LF predictions
        error2 = np.sum((y_hf_hat - y_lf_at_hf) ** 2)

        return error1 + error2

    # ------------------------------------------------------------------

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Perform model training following the three-stage CCA-MFS procedure.

        Stage 1 - CCA (Canonical Correlation Analysis):
            - Train LF RBF model on raw LF data
            - Construct sample matrices Ph = [Xh, yh] and Pl = [Xl, yl]
            - Compute covariance matrices S11, S22, S12
            - Compute transition matrices U and V via SVD

        Stage 2 - Discrepancy Function Construction:
            - Transform samples to CCA space using U and V
            - Compute distance matrices Rh (HF-HF) and Rhl (HF-LF)

        Stage 3 - Parameter Optimization:
            - Optimize rho, W1, W2 via least squares for each target dimension

        Args:
            x_lf (np.ndarray): Low-fidelity inputs.
                Shape: (num_lf_samples, input_dim), dtype: float64.
            y_lf (np.ndarray): Low-fidelity targets.
                Shape: (num_lf_samples, target_dim), dtype: float64.
            x_hf (np.ndarray): High-fidelity inputs.
                Shape: (num_hf_samples, input_dim), dtype: float64.
            y_hf (np.ndarray): High-fidelity targets.
                Shape: (num_hf_samples, target_dim), dtype: float64.
        """
        num_hf_samples = x_hf.shape[0]
        num_lf_samples = x_lf.shape[0]
        target_dim = y_hf.shape[1]

        # Stage 1: CCA
        # Step 1.1: Train LF RBF model on raw data
        self.lf_model.fit(x_lf, y_lf)

        # Step 1.2: Scale HF data
        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        # Get LF predictions at HF locations (for constructing Pl and discrepancy)
        y_lf_at_hf_raw = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf_raw, tuple):
            y_lf_at_hf_raw = y_lf_at_hf_raw[0]  # Prevent KRG's var_pred
        self.y_lf_at_hf_ = self.scaler_y.transform(y_lf_at_hf_raw)

        # Step 1.3: Construct sample matrices for CCA
        # Scale LF data using the same scalers
        x_lf_scaled = self.scaler_x.transform(x_lf)
        y_lf_scaled = self.scaler_y.transform(y_lf)

        # Ph = [Xh, yh] and Pl = [Xl, yl]
        Ph = np.concatenate([self.x_hf_train_, self.y_hf_train_], axis=1)
        Pl = np.concatenate([x_lf_scaled, y_lf_scaled], axis=1)

        # Step 1.4: Compute covariance matrices
        S11, S22, S12 = self._compute_covariance_matrices(Ph, Pl)

        # Step 1.5: Compute CCA transition matrices
        self.U_, self.V_ = self._compute_cca_transition_matrices(S11, S22, S12)

        # Stage 2: Construct discrepancy function
        # Step 2.1: Transform samples to CCA space
        self.Ph_transformed_ = Ph @ self.U_  # Shape: (num_hf, ndv)
        self.Pl_transformed_ = Pl @ self.V_  # Shape: (num_lf, ndv)

        # Step 2.2: Compute distance matrices (Euclidean distances in CCA space)
        self.Rh_ = self._compute_dists(self.Ph_transformed_, self.Ph_transformed_)
        self.Rhl_ = self._compute_dists(self.Ph_transformed_, self.Pl_transformed_)

        # Stage 3: Optimize parameters via Least Squares
        self.rho_ = np.zeros(target_dim, dtype=np.float64)
        self.W1_ = np.zeros((num_hf_samples, target_dim), dtype=np.float64)
        self.W2_ = np.zeros((num_lf_samples, target_dim), dtype=np.float64)

        # Bounds for optimization
        lb = np.concatenate([[1e-6], 1e-6 * np.ones(num_hf_samples + num_lf_samples)])
        ub = np.concatenate([[2.0], 2.0 * np.ones(num_hf_samples + num_lf_samples)])
        bounds = Bounds(lb, ub)

        for m in range(target_dim):
            # Initial guess using uniform random in bounds
            np.random.seed(42 + m)  # For reproducibility
            x0 = np.random.uniform(lb, ub)

            # Optimize parameters
            res = minimize(
                fun=self._objective_function,
                x0=x0,
                args=(
                    self.y_hf_train_[:, m],
                    self.y_lf_at_hf_[:, m],
                    self.Rh_,
                    self.Rhl_,
                    num_hf_samples
                ),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500}
            )

            # Store optimized parameters
            self.rho_[m] = res.x[0]
            self.W1_[:, m] = res.x[1:1 + num_hf_samples]
            self.W2_[:, m] = res.x[1 + num_hf_samples:]

        self.is_fitted = True

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction using the trained CCA-MFS model.

        The prediction process:
        1. Get LF predictions at query points using the trained LF RBF model
        2. Construct test sample matrix P_test = [x_pred, y_lf_pred] and transform to CCA space
        3. Compute distance matrices Rh_ts (with HF samples) and Rl_ts (with LF samples)
        4. Apply the multi-fidelity correction:
           y_pred = rho * y_lf + Rh_ts @ W1 + Rl_ts @ W2

        Args:
            x_pred (np.ndarray): Prediction inputs.
                Shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets.
                Shape: (num_samples, target_dim), dtype: float64.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        num_samples = x_pred.shape[0]
        target_dim = self.rho_.shape[0]

        # Step 1: Scale prediction inputs
        x_pred_scaled = self.scaler_x.transform(x_pred)

        # Step 2: Get LF predictions at query points
        y_lf_pred_raw = self.lf_model.predict(x_pred)
        if isinstance(y_lf_pred_raw, tuple):
            y_lf_pred_raw = y_lf_pred_raw[0]
        y_lf_pred_scaled = self.scaler_y.transform(y_lf_pred_raw)

        # Step 3: Construct test sample matrix and transform to CCA space
        # P_test = [x_pred, y_lf_pred]
        P_test = np.concatenate([x_pred_scaled, y_lf_pred_scaled], axis=1)
        P_test_transformed_U = P_test @ self.U_  # For HF correlation (uses U)
        P_test_transformed_V = P_test @ self.V_  # For LF correlation (uses V)

        # Step 4: Compute distance matrices (distances in CCA space)
        Rh_ts = self._compute_dists(P_test_transformed_U, self.Ph_transformed_)
        Rl_ts = self._compute_dists(P_test_transformed_V, self.Pl_transformed_)

        # Step 5: Compute predictions for each target dimension
        y_pred_scaled = np.zeros((num_samples, target_dim), dtype=np.float64)

        for m in range(target_dim):
            y_pred_scaled[:, m] = (
                self.rho_[m] * y_lf_pred_scaled[:, m] +
                Rh_ts @ self.W1_[:, m] +
                Rl_ts @ self.W2_[:, m]
            )

        # Step 6: Inverse scale predictions to original space
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred
