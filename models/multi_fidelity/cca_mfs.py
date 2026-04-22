# CCA-MFS: Multi-Fidelity Surrogate Model Based on Canonical Correlation Analysis and Least Squares
# Paper reference: https://doi.org/10.1115/1.4047686
# Paper author: Liye Lv, Chaoyang Zong, Chao Zhang, Xueguan Song, Wei Sun
# Code author: Shengning Wang

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import inv, sqrtm

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class CCAMFS:
    """
    Multi-fidelity surrogate model based on canonical correlation analysis.
    """

    def __init__(self, lf_model_params: Optional[Dict] = None, residual_ridge: float = 1.0) -> None:
        """
        Initialize the CCA-MFS surrogate.

        Args:
            lf_model_params (Optional[Dict]): Parameters for internal RBF models.
            residual_ridge (float): Ridge factor for residual correction.
        """
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)
        self.hf_rbf_model_ = RBF(**params)
        self.lf_rbf_model_ = RBF(**params)
        self.residual_ridge = residual_ridge

        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        self.U_: Optional[np.ndarray] = None
        self.V_: Optional[np.ndarray] = None
        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.y_lf_at_hf_: Optional[np.ndarray] = None
        self.Ph_transformed_: Optional[np.ndarray] = None
        self.Pl_transformed_: Optional[np.ndarray] = None
        self.Rh_: Optional[np.ndarray] = None
        self.Rhl_: Optional[np.ndarray] = None
        self.bias_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.W1_: Optional[np.ndarray] = None
        self.W2_: Optional[np.ndarray] = None
        self.is_fitted = False

    def _compute_covariance_matrices(self, Ph: np.ndarray, Pl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute paired covariance matrices for CCA.

        Args:
            Ph (np.ndarray): HF paired samples. (N, D).
            Pl (np.ndarray): LF paired samples. (N, D).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: S11, S22, S12.
        """
        # In the paper notation, the two fidelity views are treated as matrices
        # P_h in R^{N x D_h} and P_l in R^{N x D_l}.
        # CCA is driven by centered second-order statistics of these two blocks.
        # Canonical correlation starts from the covariance blocks
        # S11 = cov(P_h, P_h), S22 = cov(P_l, P_l), S12 = cov(P_h, P_l).
        # More explicitly, with
        # P_h_tilde = P_h - 1_N mean(P_h),
        # P_l_tilde = P_l - 1_N mean(P_l),
        # the covariance estimates become
        # S11 = P_h_tilde^T P_h_tilde / (N - 1),
        # S22 = P_l_tilde^T P_l_tilde / (N - 1),
        # S12 = P_h_tilde^T P_l_tilde / (N - 1).
        ph_centered = Ph - np.mean(Ph, axis=0, keepdims=True)
        pl_centered = Pl - np.mean(Pl, axis=0, keepdims=True)
        scale = Ph.shape[0] - 1
        S11 = (ph_centered.T @ ph_centered) / scale
        S22 = (pl_centered.T @ pl_centered) / scale
        S12 = (ph_centered.T @ pl_centered) / scale
        return S11, S22, S12

    def _compute_cca_transition_matrices(
        self,
        S11: np.ndarray,
        S22: np.ndarray,
        S12: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CCA transition matrices.

        Args:
            S11 (np.ndarray): HF covariance. (D, D).
            S22 (np.ndarray): LF covariance. (D, D).
            S12 (np.ndarray): Cross covariance. (D, D).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transition matrices U and V.
        """
        # CCA solves for directions u and v maximizing
        # corr(P_h u, P_l v)
        # under the variance constraints
        # u^T S11 u = 1 and v^T S22 v = 1.
        # Whitening + SVD:
        # C = S11^{-1/2} S12 S22^{-1/2} = L Sigma R^T,
        # U = S11^{-1/2} L, V = S22^{-1/2} R.
        # The singular values in Sigma are the canonical correlations, and the
        # columns of U, V give the maximally correlated directions in each view.
        S11_inv_sqrt = inv(sqrtm(S11).astype(np.float64))
        S22_inv_sqrt = inv(sqrtm(S22).astype(np.float64))
        C = S11_inv_sqrt @ S12 @ S22_inv_sqrt
        L, _, R_t = np.linalg.svd(C, full_matrices=True)
        U = S11_inv_sqrt @ L
        V = S22_inv_sqrt @ R_t.T
        return U, V

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances.

        Args:
            x (np.ndarray): Query points. (N, D).
            c (np.ndarray): Reference points. (M, D).

        Returns:
            np.ndarray: Distances. (N, M).
        """
        # Euclidean distance is built from the quadratic expansion of ||x - c||^2.
        # For every query-center pair (i, j),
        # d_ij^2 = x_i^T x_i + c_j^T c_j - 2 x_i^T c_j.
        # The final kernel uses d_ij after the square root below.
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(c ** 2, axis=1)
        dists_sq = x_norm_sq + c_norm_sq - 2.0 * (x @ c.T)
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return np.sqrt(dists_sq)

    def _build_rbf_correlation_from_model(self, x_query: np.ndarray, model: RBF) -> np.ndarray:
        """
        Build an RBF correlation matrix from a trained RBF model.

        Args:
            x_query (np.ndarray): Query points. (N, D).
            model (RBF): Trained RBF model.

        Returns:
            np.ndarray: Correlation matrix. (N, M).
        """
        dists = self._compute_dists(x_query, model.centers)
        # Reuse the internal Gaussian RBF kernel:
        # R_ij = exp(-gamma ||x_i - c_j||^2).
        # Here gamma plays the role of the inverse bandwidth, so larger gamma
        # produces a more localized correlation matrix around each RBF center.
        return np.exp(-model.gamma * (dists ** 2))

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Fit the CCA-MFS surrogate.

        Args:
            x_lf (np.ndarray): LF inputs. (N_L, D).
            y_lf (np.ndarray): LF targets. (N_L, C).
            x_hf (np.ndarray): HF inputs. (N_H, D).
            y_hf (np.ndarray): HF targets. (N_H, C).
        """
        # The model combines three ingredients:
        # 1. An LF surrogate y_l_hat(x) learned from abundant LF samples.
        # 2. A global affine mapping from LF response to HF response.
        # 3. Canonical-space RBF residuals that correct the nonlinear mismatch.
        # For each output component m, the final form is
        # y_h_hat^(m)(x) =
        #     b_m + rho_m y_l_hat^(m)(x)
        #     + delta_h^(m)(x) + delta_l^(m)(x).
        num_hf, _ = x_hf.shape
        num_lf = x_lf.shape[0]
        target_dim = y_hf.shape[1]

        # Step 1:
        # fit y_l_hat(x) on the raw low-fidelity dataset (x_lf, y_lf).
        self.lf_model.fit(x_lf, y_lf)

        # Step 2:
        # standardize the HF design and response so the covariance and least-
        # squares systems are built in numerically balanced coordinates.
        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        # Step 3:
        # evaluate the LF surrogate on the HF design sites to obtain paired data
        # y_l_hat(x_hf_1), ..., y_l_hat(x_hf_Nh).
        # These paired LF values are required because CCA assumes matched samples
        # across the two views.
        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple):
            y_lf_at_hf = y_lf_at_hf[0]
        self.y_lf_at_hf_ = self.scaler_y.transform(y_lf_at_hf)

        x_lf_scaled = self.scaler_x.transform(x_lf)
        y_lf_scaled = self.scaler_y.transform(y_lf)

        # Build paired state vectors
        # P_h = [x_h, y_h], P_l = [x_l, y_l], and the paired LF proxy
        # P_l^cca = [x_h, y_l(x_h)] for cross-fidelity correlation analysis.
        # P_h contains the true expensive response at the HF sites.
        # P_l contains the original LF state over the full LF sample set.
        # P_l^cca replaces y_l with y_l_hat(x_h) so that its rows are aligned
        # one-to-one with the rows of P_h.
        Ph = np.concatenate([self.x_hf_train_, self.y_hf_train_], axis=1)
        Pl = np.concatenate([x_lf_scaled, y_lf_scaled], axis=1)
        Pl_cca = np.concatenate([self.x_hf_train_, self.y_lf_at_hf_], axis=1)

        # Step 4:
        # solve the CCA alignment problem on the paired blocks (P_h, P_l^cca).
        # This extracts transforms U and V such that the canonical variables
        # Z_h = P_h U and Z_l = P_l V are maximally linearly correlated.
        S11, S22, S12 = self._compute_covariance_matrices(Ph, Pl_cca)
        self.U_, self.V_ = self._compute_cca_transition_matrices(S11, S22, S12)

        # Step 5:
        # project the HF and LF augmented states into the canonical spaces.
        # Z_h = P_h U acts as the HF latent coordinate.
        # Z_l = P_l V acts as the LF latent coordinate.
        self.Ph_transformed_ = Ph @ self.U_
        self.Pl_transformed_ = Pl @ self.V_

        # Step 6:
        # train two RBF surrogates in the canonical coordinates.
        # One interpolates HF responses over Z_h and the other interpolates LF
        # responses over Z_l.
        self.hf_rbf_model_.fit(self.Ph_transformed_, self.y_hf_train_)
        self.lf_rbf_model_.fit(self.Pl_transformed_, y_lf_scaled)

        # Evaluate the RBF basis matrices at the HF canonical samples:
        # R_h  = K_h(Z_h, Z_h),
        # R_hl = K_l(Z_h, Z_l_train).
        # The first block is an HF-to-HF correction basis.
        # The second block is an LF-canonical basis sampled at HF locations.
        self.Rh_ = self._build_rbf_correlation_from_model(self.Ph_transformed_, self.hf_rbf_model_)
        self.Rhl_ = self._build_rbf_correlation_from_model(self.Ph_transformed_, self.lf_rbf_model_)

        # Residual correction uses the stacked basis [R_h, R_hl].
        # Writing H = [R_h, R_hl], the residual fit is performed in the space
        # spanned jointly by the HF and LF canonical kernels.
        # The regularized objective is
        # min_w ||H w - r||_2^2 + lambda ||w||_2^2,
        # whose normal equations are
        # (H^T H + lambda I) w = H^T r.
        correction_matrix = np.concatenate([self.Rh_, self.Rhl_], axis=1)
        gram = correction_matrix.T @ correction_matrix
        if self.residual_ridge > 0.0:
            gram = gram + self.residual_ridge * np.eye(gram.shape[0], dtype=np.float64)

        # The stored parameters have the following roles:
        # bias_[m] -> intercept b_m,
        # rho_[m]  -> LF scale factor rho_m,
        # W1_[:, m] -> HF canonical kernel coefficients,
        # W2_[:, m] -> LF canonical kernel coefficients.
        self.bias_ = np.zeros(target_dim, dtype=np.float64)
        self.rho_ = np.zeros(target_dim, dtype=np.float64)
        self.W1_ = np.zeros((num_hf, target_dim), dtype=np.float64)
        self.W2_ = np.zeros((num_lf, target_dim), dtype=np.float64)

        ones = np.ones((num_hf, 1), dtype=np.float64)
        for m in range(target_dim):
            # Fit the affine bridge
            # y_h^(m) ~= b_m + rho_m y_l^(m)(x_h).
            # In matrix form:
            # A_m theta_m ~= y_h^(m),
            # A_m = [1, y_l_hat^(m)(x_h)],
            # theta_m = [b_m, rho_m]^T.
            affine_matrix = np.concatenate([ones, self.y_lf_at_hf_[:, m:m + 1]], axis=1)
            affine_target = self.y_hf_train_[:, m:m + 1]

            try:
                affine_theta, _, _, _ = np.linalg.lstsq(affine_matrix, affine_target, rcond=None)
            except np.linalg.LinAlgError:
                affine_theta = np.linalg.pinv(affine_matrix) @ affine_target

            self.bias_[m] = float(affine_theta[0, 0])
            self.rho_[m] = float(affine_theta[1, 0])

            # The remaining discrepancy is expanded in the coupled RBF space:
            # r_m ~= R_h w_{1, m} + R_hl w_{2, m}.
            # Equivalently,
            # r_m = y_h^(m) - (b_m + rho_m y_l_hat^(m)(x_h)).
            # This residual collects the nonlinear difference that is not
            # explained by the affine LF-to-HF bridge.
            residual_target = affine_target - (self.bias_[m] + self.rho_[m] * self.y_lf_at_hf_[:, m:m + 1])
            # Right-hand side of the regularized normal equations:
            # rhs = H^T r_m.
            rhs = correction_matrix.T @ residual_target
            try:
                correction_theta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                correction_theta = np.linalg.pinv(gram) @ rhs

            # Partition the stacked coefficient vector into the two kernel blocks:
            # correction_theta = [w_{1, m}; w_{2, m}].
            self.W1_[:, m] = correction_theta[:num_hf, 0]
            self.W2_[:, m] = correction_theta[num_hf:, 0]

        self.is_fitted = True

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new inputs.

        Args:
            x_pred (np.ndarray): Prediction inputs. (N, D).

        Returns:
            np.ndarray: Predicted targets. (N, C).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        # Prediction repeats the same augmentation used during fitting:
        # from x_* we first compute y_l_hat(x_*), then we construct
        # P_*(x) = [x_*, y_l_hat(x_*)].
        x_pred_scaled = self.scaler_x.transform(x_pred)
        y_lf_pred = self.lf_model.predict(x_pred)
        if isinstance(y_lf_pred, tuple):
            y_lf_pred = y_lf_pred[0]
        y_lf_pred_scaled = self.scaler_y.transform(y_lf_pred)

        # Apply the learned CCA transforms to the test augmented state:
        # Z_h,* = P_* U,
        # Z_l,* = P_* V.
        # The same x_* is therefore viewed through the HF- and LF-aligned
        # canonical coordinates before the residual kernels are evaluated.
        P_test = np.concatenate([x_pred_scaled, y_lf_pred_scaled], axis=1)
        P_test_U = P_test @ self.U_
        P_test_V = P_test @ self.V_

        # Kernel responses from the test point to the stored canonical centers:
        # R_h,*(x) = K_h(Z_h,*, Z_h,train),
        # R_l,*(x) = K_l(Z_l,*, Z_l,train).
        Rh_ts = self._build_rbf_correlation_from_model(P_test_U, self.hf_rbf_model_)
        Rl_ts = self._build_rbf_correlation_from_model(P_test_V, self.lf_rbf_model_)

        # Final predictor:
        # y_h(x) = b + rho * y_l(x) + R_h(x) W1 + R_l(x) W2.
        # The first term transports the dominant LF trend to the HF scale.
        # The kernel terms restore the remaining correlated residual structure.
        y_pred_scaled = self.bias_[None, :] + self.rho_[None, :] * y_lf_pred_scaled + Rh_ts @ self.W1_ + Rl_ts @ self.W2_
        return self.scaler_y.inverse_transform(y_pred_scaled)
