# MFS-MLS: Multi-Fidelity Surrogate Model based on Moving Least Squares
# Paper reference: https://doi.org/10.1007/s00158-021-03044-5
# Paper author: Shuo Wang, Yin Liu, Qi Zhou, Yongliang Yuan, Liye Lv, Xueguan Song
# Code author: Shengning Wang

from itertools import combinations_with_replacement
from typing import Dict, Optional

import numpy as np

from models.classical.rbf import RBF
from utils.scaler import StandardScalerNP


class MFSMLS:
    """
    Multi-fidelity surrogate model based on moving least squares.
    """

    def __init__(
        self,
        lf_model_params: Optional[Dict] = None,
        poly_degree: int = 2,
        neighbor_factor: float = 1.0,
        ridge: float = 1.0e-8,
    ) -> None:
        """
        Initialize the MFS-MLS surrogate.

        Args:
            lf_model_params (Optional[Dict]): Parameters for the LF RBF model.
            poly_degree (int): Polynomial basis degree.
            neighbor_factor (float): Expansion factor for required HF neighbors.
            ridge (float): Ridge factor for local weighted least squares.
        """
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)
        self.poly_degree = poly_degree
        self.neighbor_factor = neighbor_factor
        self.ridge = ridge

        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.p_train_: Optional[np.ndarray] = None
        self.required_hf_samples_: int = 0
        self.is_fitted = False

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances.

        Args:
            x (np.ndarray): Query points. (N, D).
            c (np.ndarray): Reference points. (M, D).

        Returns:
            np.ndarray: Distance matrix. (N, M).
        """
        # MLS neighborhoods are built from Euclidean distances in the scaled design space.
        # For each pair (i, j),
        # d_ij^2 = x_i^T x_i + c_j^T c_j - 2 x_i^T c_j,
        # and d_ij is used later to determine the local support radius.
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(c ** 2, axis=1)
        dists_sq = x_norm_sq + c_norm_sq - 2.0 * (x @ c.T)
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return np.sqrt(dists_sq)

    def _generate_polynomial_powers(self, input_dim: int) -> np.ndarray:
        """
        Generate monomial exponent vectors.

        Args:
            input_dim (int): Input dimension.

        Returns:
            np.ndarray: Exponent matrix. (P, D).
        """
        # The polynomial basis is indexed by multi-indices
        # alpha = [alpha_1, ..., alpha_D] with total degree
        # |alpha| = alpha_1 + ... + alpha_D <= poly_degree.
        # Each such alpha corresponds to one monomial x^alpha.
        powers = []
        for degree in range(self.poly_degree + 1):
            # Enumerate all polynomial exponents with total degree <= poly_degree.
            # A tuple like (0, 0, 2) means dimension 0 appears twice and
            # dimension 2 appears once, which yields the monomial x_0^2 x_2.
            for combo in combinations_with_replacement(range(input_dim), degree):
                power = np.zeros(input_dim, dtype=np.int64)
                for idx in combo:
                    power[idx] += 1
                powers.append(power)
        # Stacking all exponent vectors gives the basis descriptor matrix
        # A = [alpha^(1); ...; alpha^(P)].
        return np.stack(powers, axis=0)

    def _build_polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """
        Build polynomial basis features.

        Args:
            x (np.ndarray): Input points. (N, D).

        Returns:
            np.ndarray: Polynomial features. (N, P).
        """
        powers = self._generate_polynomial_powers(x.shape[1])
        # Polynomial basis:
        # p_j(x) = prod_k x_k^{power_{j, k}}.
        # This is the standard monomial map phi(x) associated with the exponent
        # table above, evaluated rowwise for every input sample.
        phi = np.ones((x.shape[0], powers.shape[0]), dtype=x.dtype)
        for dim in range(x.shape[1]):
            exp_dim = powers[:, dim]
            mask = exp_dim > 0
            if np.any(mask):
                # Multiply x_dim^{alpha_dim} into every monomial column whose
                # exponent on the current dimension is nonzero.
                phi[:, mask] *= np.power(x[:, dim:dim + 1], exp_dim[mask])
        return phi

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Fit the MFS-MLS surrogate.

        Args:
            x_lf (np.ndarray): LF inputs. (N_L, D).
            y_lf (np.ndarray): LF targets. (N_L, C).
            x_hf (np.ndarray): HF inputs. (N_H, D).
            y_hf (np.ndarray): HF targets. (N_H, C).
        """
        # MFS-MLS approximates the HF response locally as
        # y_h_hat(x) = p(x)^T a(x),
        # where p(x) contains LF-informed basis terms and polynomial drift terms,
        # and a(x) is solved separately around each query location.

        # Step 1:
        # fit the global LF surrogate y_l_hat(x) from the LF dataset.
        self.lf_model.fit(x_lf, y_lf)

        # Step 2:
        # standardize the HF inputs and outputs so that neighborhood radii and
        # weighted least-squares systems are built in scaled coordinates.
        self.x_hf_train_ = self.scaler_x.fit(x_hf, channel_dim=1).transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit(y_hf, channel_dim=1).transform(y_hf)

        # Step 3:
        # lift the HF design into the LF response space via y_l_hat(x_hf).
        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple):
            y_lf_at_hf = y_lf_at_hf[0]

        y_lf_at_hf_scaled = self.scaler_y.transform(y_lf_at_hf)
        # Step 4:
        # build the polynomial drift block phi(x_hf).
        poly_basis = self._build_polynomial_features(self.x_hf_train_)
        # The local basis stacks the LF trend and the polynomial correction terms:
        # P(x) = [y_l(x), p_1(x), ..., p_P(x)].
        # Each training row is therefore
        # p(x_hf_i)^T = [y_l_hat(x_hf_i), phi_1(x_hf_i), ..., phi_P(x_hf_i)].
        # In matrix form, the full design matrix is
        # P_train = [p(x_hf_1)^T; ...; p(x_hf_Nh)^T].
        # This matrix is reused for every query, while only the weights W(x)
        # change from one location to another.
        self.p_train_ = np.concatenate([y_lf_at_hf_scaled, poly_basis], axis=1)

        min_required = self.p_train_.shape[1]
        # The local weighted least-squares system needs at least as many HF samples
        # as basis functions, then expands that neighborhood by neighbor_factor.
        # If the basis dimension is M, the neighborhood size is chosen as
        # k = max(M, ceil(neighbor_factor * M)),
        # then truncated by the available HF sample count.
        expanded_required = int(np.ceil(self.neighbor_factor * min_required))
        self.required_hf_samples_ = min(max(min_required, expanded_required), self.x_hf_train_.shape[0])
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

        # Query-side basis evaluation:
        # p(x_*) = [y_l_hat(x_*), phi(x_*)].
        x_pred_scaled = self.scaler_x.transform(x_pred)
        y_lf_at_pred = self.lf_model.predict(x_pred)
        if isinstance(y_lf_at_pred, tuple):
            y_lf_at_pred = y_lf_at_pred[0]

        y_lf_at_pred_scaled = self.scaler_y.transform(y_lf_at_pred)
        poly_basis_pred = self._build_polynomial_features(x_pred_scaled)
        # Query design rows:
        # P_pred = [p(x_1)^T; ...; p(x_N)^T].
        # Each row multiplies the local coefficient matrix solved at the same
        # query location.
        p_pred = np.concatenate([y_lf_at_pred_scaled, poly_basis_pred], axis=1)

        # Distances from every query point to every HF sample define the MLS
        # neighborhood and the diagonal weight matrix W(x_*).
        dists = self._compute_dists(x_pred_scaled, self.x_hf_train_)
        num_samples = x_pred.shape[0]
        num_basis = self.p_train_.shape[1]
        target_dim = self.y_hf_train_.shape[1]
        y_pred_scaled = np.zeros((num_samples, target_dim), dtype=np.float64)

        for i in range(num_samples):
            sorted_dists = np.sort(dists[i])
            # Use the required_hf_samples-th nearest HF point as the influence radius r_i.
            # Formally, r_i = d_(k)(x_i) with k = required_hf_samples_.
            # This ensures at least k nonzero candidate weights around x_i.
            influence_radius = max(float(sorted_dists[self.required_hf_samples_ - 1]), 1.0e-12)
            # Normalized distances:
            # d_bar_j(x_i) = ||x_i - x_j^hf||_2 / r_i.
            di = dists[i] / influence_radius

            wi = np.zeros(self.x_hf_train_.shape[0], dtype=np.float64)
            mask = di <= 1.0
            # Compact Gaussian-like MLS weights:
            # w_j(x) = exp(-4 (d_j / r_i)^2), d_j <= r_i.
            # Outside the support radius, w_j(x) = 0.
            # Hence only local HF samples contribute to the least-squares fit.
            wi[mask] = np.exp(-4.0 * di[mask] ** 2)

            W = np.diag(wi)
            # Local normal equations:
            # (P^T W P + lambda I) a(x) = P^T W y_h.
            # This is the normal-equation form of the weighted objective
            # min_a ||W^{1/2} (P a - y_h)||_2^2 + lambda ||a||_2^2.
            # The coefficient matrix a(x_i) has shape (num_basis, target_dim),
            # so every output channel shares the same scalar weights W(x_i) but
            # has its own fitted local regression coefficients.
            lhs = self.p_train_.T @ W @ self.p_train_
            rhs = self.p_train_.T @ W @ self.y_hf_train_

            try:
                coeffs = np.linalg.solve(lhs + self.ridge * np.eye(num_basis), rhs)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.pinv(lhs) @ rhs

            # The prediction is the local basis evaluated at x times the local coefficients.
            # Equivalently, y_h_hat(x_i) = p(x_i)^T a(x_i).
            # This preserves the MLS interpretation: fit locally, then evaluate
            # the fitted local polynomial/LF basis at the query itself.
            y_pred_scaled[i] = p_pred[i] @ coeffs

        return self.scaler_y.inverse_transform(y_pred_scaled)
