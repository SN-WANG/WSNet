# MMFS: Modified Multi-Fidelity Surrogate Model based on RBF with Adaptive Scale Factor
# Paper reference: https://doi.org/10.1186/s10033-022-00742-z
# Paper author: Yin Liu, Shuo Wang, Qi Zhou, Liye Lv, Wei Sun, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from scipy.linalg import pinv
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional, List

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class MMFS:
    """
    Modified Multi-Fidelity Surrogate Model based on RBF with Adaptive Scale Factor (MMFS).

    Implements the MMFS algorithm which utilizes a Low-Fidelity (LF) RBF model to approximate
    the trend and constructs a High-Fidelity (HF) correction model. The correction involves
    an adaptive scale factor and a deviation term, solved via a comprehensive correction matrix.

    Key Features:
    - Adaptive scaling and shifting based on HF sample locations.
    - Multiquadric (MQ) kernel usage for the HF correction layer.
    - Leave-One-Out Cross-Validation (LOOCV) for optimizing the HF shape parameter (sigma).
    - Generalized Inverse (Minimum Norm Solution) for solving model coefficients.

    Attributes:
        lf_model (RBF): The Low-Fidelity surrogate model instance.
        sigma_bounds (Tuple[float, float]): Search bounds for the RBF shape parameter sigma.
        scaler_x (StandardScalerNP): Preprocessor for HF inputs.
        scaler_y (StandardScalerNP): Preprocessor for HF targets.
        x_hf_train_ (np.ndarray): Scaled HF training inputs.
            Shape: (num_hf_samples, input_dim), dtype: float64.
        y_hf_train_ (np.ndarray): Scaled HF training targets.
            Shape: (num_hf_samples, target_dim), dtype: float64.
        beta_ (List[np.ndarray]): List of coefficient vectors [beta_1, ..., beta_m] for each target dimension.
            Each beta has shape (2 * num_hf_samples, 1).
        sigma_ (float): Optimized shape parameter for the HF correction RBF.
        is_fitted (bool): Status flag indicating if the model has been trained.
    """

    def __init__(
            self, lf_model_params: Optional[Dict] = None, sigma_bounds: Tuple[float, float] = (0.01, 10.0)
    ) -> None:
        """
        Initialize the MMFS model configuration.

        Args:
            lf_model_params (Optional[Dict]): Dictionary of kwargs to pass to the base RBF class
                for the LF model. If None, default RBF parameters are used.
            sigma_bounds (Tuple[float, float]): The lower and upper bounds for optimizing
                the shape parameter sigma.
        """
        # parameters
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)

        self.sigma_bounds = sigma_bounds

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.beta_: List[np.ndarray] = []
        self.sigma_: float = 1.0
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

    def _multiquadric_kernel(self, dists_sq: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute the Multiquadric (MQ) basis function: phi(r) = sqrt(r^2 + sigma^2).

        Args:
            dists_sq (np.ndarray): Squared Euclidean distance matrix of shape (n, m), dtype: float64.
            sigma (float): Shape parameter (bandwidth).

        Returns:
            np.ndarray: Kernel matrix of shape (n, m), dtype: float64.
        """
        return np.sqrt(dists_sq + sigma ** 2)

    # ------------------------------------------------------------------

    def _construct_expansion_matrix(self, phi: np.ndarray, y_lf: np.ndarray) -> np.ndarray:
        """
        Construct the expansion matrix H = [diag(y_lf) * phi, phi].

        This matrix combines the adaptive scaling (first block) and shifting (second block)
        operations for the multi-fidelity correction.

        Args:
            phi (np.ndarray): RBF kernel matrix of shape (num_samples, num_centers), dtype: float64.
            y_lf (np.ndarray): Low-fidelity predictions of shape (num_samples, 1), dtype: float64.

        Returns:
            np.ndarray: Expansion matrix H of shape (num_samples, 2 * num_centers), dtype: float64.
        """
        # Element-wise multiplication of y_lf with each column of phi (equivalent to diag(y_lf) @ phi)
        scaled_phi = y_lf * phi
        return np.concatenate([scaled_phi, phi], axis=1)

    # ------------------------------------------------------------------

    def _loocv_error(self, sigma: float, dist_matrix: np.ndarray, y_lf_at_hf: np.ndarray, y_hf: np.ndarray) -> float:
        """
        Calculate the Leave-One-Out Cross-Validation (LOOCV) error for a specific sigma.

        Iterates through all samples and output dimensions to compute the total MSE.

        Args:
            sigma (float): The candidate shape parameter.
            dist_matrix (np.ndarray): Pre-computed squared distance matrix of HF samples
                of shape (num_hf, num_hf), dtype: float64.
            y_lf_at_hf (np.ndarray): Scaled LF predictions at HF sites
                of shape (num_hf, target_dim), dtype: float64.
            y_hf (np.ndarray): Scaled HF targets of shape (num_hf, target_dim), dtype: float64.

        Returns:
            float: The total Mean Squared Error across all LOO folds and outputs.
        """
        num_hf, target_dim = y_hf.shape

        # Avoid singularity
        if sigma < 1e-6:
            return 1e10

        # Compute full kernel matrix for this sigma
        phi_full = self._multiquadric_kernel(dist_matrix, sigma)

        total_error = 0.0

        # LOOCV loop
        for i in range(num_hf):
            # Split indices: leave one out
            train_idx = np.delete(np.arange(num_hf), i)
            test_idx = i

            # Extract sub-matrices
            phi_train = phi_full[np.ix_(train_idx, train_idx)]  # Shape: (num_hf - 1, num_hf - 1)
            phi_test = phi_full[np.ix_([test_idx], train_idx)]  # Shape: (1, num_hf - 1)

            # Solve for each output dimension
            for m in range(target_dim):
                y_c_train = y_lf_at_hf[train_idx, m:m + 1]
                y_e_train = y_hf[train_idx, m:m + 1]

                # Construct H matrix for training set
                H_train = self._construct_expansion_matrix(phi_train, y_c_train)

                # Solve beta using generalized inverse (minimum norm solution)
                try:
                    beta = pinv(H_train) @ y_e_train
                except np.linalg.LinAlgError:
                    total_error += 1e10
                    continue

                # Predict validation point
                y_c_test = y_lf_at_hf[test_idx, m:m + 1]
                H_test = self._construct_expansion_matrix(phi_test, y_c_test)

                y_pred = H_test @ beta
                y_true = y_hf[test_idx, m:m + 1]

                total_error += np.sum((y_pred - y_true) ** 2)

        return total_error / num_hf

    # ------------------------------------------------------------------

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Perform model training.

        The training process involves:
        1. Training the LF surrogate model on raw LF data.
        2. Normalizing HF inputs and targets using StandardScalerNP.
        3. Optimizing the shape parameter sigma via LOOCV on HF data.
        4. Solving for adaptive scale coefficients (beta) using the full HF dataset.

        Args:
            x_lf (np.ndarray): Low-fidelity inputs of shape (num_lf_samples, input_dim).
            y_lf (np.ndarray): Low-fidelity targets of shape (num_lf_samples, target_dim).
            x_hf (np.ndarray): High-fidelity inputs of shape (num_hf_samples, input_dim).
            y_hf (np.ndarray): High-fidelity targets of shape (num_hf_samples, target_dim).
        """
        # Step 1: Train LF model
        self.lf_model.fit(x_lf, y_lf)

        # Step 2: Scale HF data
        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        # Step 3: Get LF predictions at HF points
        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple):  y_lf_at_hf = y_lf_at_hf[0]  # Prevent KRG's var_pred

        y_lf_at_hf_scaled = self.scaler_y.transform(y_lf_at_hf)

        # Step 4: Optimize sigma via LOOCV
        dist_matrix = self._compute_dists(self.x_hf_train_, self.x_hf_train_)

        res = minimize_scalar(fun=self._loocv_error, bounds=self.sigma_bounds,
            args=(dist_matrix, y_lf_at_hf_scaled, self.y_hf_train_), method="bounded"
        )
        self.sigma_ = res.x

        # Step 5: Compute final coefficients (beta) for each output
        phi_train = self._multiquadric_kernel(dist_matrix, self.sigma_)

        self.beta_ = []
        target_dim = self.y_hf_train_.shape[1]

        for m in range(target_dim):
            y_c = y_lf_at_hf_scaled[:, m:m + 1]
            y_e = self.y_hf_train_[:, m:m + 1]

            # Construct expansion matrix H
            H = self._construct_expansion_matrix(phi_train, y_c)

            # Solve beta using generalized inverse
            beta_m = pinv(H) @ y_e
            self.beta_.append(beta_m)

        self.is_fitted = True

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction.

        The prediction process involves:
        1. Predicting LF response at query points.
        2. Computing cross-correlation matrix between query points and HF training points.
        3. Applying the expansion matrix transformation and correction coefficients.

        Args:
            x_pred (np.ndarray): Prediction inputs of shape (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets of shape (num_samples, target_dim), dtype: float64.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        num_samples = x_pred.shape[0]
        target_dim = len(self.beta_)

        # Step 1: Scale prediction inputs
        x_pred_scaled = self.scaler_x.transform(x_pred)

        # Step 2: Get LF predictions at query points
        y_lf_at_pred = self.lf_model.predict(x_pred)
        if isinstance(y_lf_at_pred, tuple):  y_lf_at_pred = y_lf_at_pred[0]  # Prevent KRG's var_pred

        y_lf_at_pred_scaled = self.scaler_y.transform(y_lf_at_pred)

        # Step 3: Calculate cross-distances and kernel matrix
        dists_pred = self._compute_dists(x_pred_scaled, self.x_hf_train_)
        phi_pred = self._multiquadric_kernel(dists_pred, self.sigma_)

        # Step 4: Compute predictions for each output dimension
        y_pred_scaled = np.zeros((num_samples, target_dim), dtype=np.float64)

        for m in range(target_dim):
            y_c_pred = y_lf_at_pred_scaled[:, m:m + 1]
            beta_m = self.beta_[m]

            # Construct expansion matrix H*
            H_star = self._construct_expansion_matrix(phi_pred, y_c_pred)

            # Y(x) = H* @ beta
            y_pred_scaled[:, m:m + 1] = H_star @ beta_m

        # Step 5: Inverse scale predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred
