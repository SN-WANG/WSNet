# Multi-Fidelity Infill via MICO (Mutual Information and Correlation Criterion)
# Paper reference: https://doi.org/10.1007/s00366-023-01858-z
# Paper author: Shuo Wang, Xiaonan Lai, Xiwang He, Kunpeng Li, Liye Lv, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from typing import List, Optional, Tuple

from sampling.base_infill import BaseInfill
from utils.scaler import MinMaxScalerNP


class MultiFidelityInfill(BaseInfill):
    """
    Greedy multi-fidelity infill based on the MICO score.

    The LF pool is treated as a discrete candidate set. evaluate() maps
    continuous queries to the nearest normalised LF node, while propose()
    greedily returns the remaining LF node with the largest MICO score.
    """

    # ============================================================
    # Initialization And MF Prior
    # ============================================================
    def __init__(
        self,
        model,
        x_hf: np.ndarray,
        y_hf: np.ndarray,
        x_lf: np.ndarray,
        y_lf: np.ndarray,
        target_idx: int = 0,
        ratio: float = 0.5,
        theta_v: Optional[np.ndarray] = None,
        theta_d: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the MICO-based multi-fidelity infill strategy.

        Args:
            model: Fitted surrogate model passed through the base contract.
            x_hf (np.ndarray): HF training locations. (num_hf, input_dim).
            y_hf (np.ndarray): HF training responses. (num_hf, output_dim).
            x_lf (np.ndarray): LF candidate pool. (num_lf, input_dim).
            y_lf (np.ndarray): LF responses on the pool. (num_lf, output_dim).
            target_idx (int): Output dimension used for scoring. Default 0.
            ratio (float): Kept for compatibility, but not used in scoring.
            theta_v (Optional[np.ndarray]): LF correlation lengths. If None,
                estimate them from the normalised LF point set.
            theta_d (Optional[np.ndarray]): Discrepancy correlation lengths.
                If None, copy theta_v.
        """
        super().__init__(model, bounds=None, target_idx=target_idx, num_restarts=0)

        self.x_hf = np.asarray(x_hf, dtype=np.float64)
        self.y_hf = np.asarray(y_hf, dtype=np.float64)
        self.x_lf = np.asarray(x_lf, dtype=np.float64)
        self.y_lf = np.asarray(y_lf, dtype=np.float64)

        self.num_lf, self.input_dim = self.x_lf.shape
        self.num_hf = self.x_hf.shape[0]
        self.output_dim = self.y_lf.shape[1]
        self.ratio = float(ratio)

        # Both fidelities are embedded in the same unit hypercube:
        # \hat{x} = (x - x_min) / (x_max - x_min).
        # This makes the LF candidate lattice and HF observations comparable
        # under the same distance and kernel metrics.
        self._scaler_x = MinMaxScalerNP(norm_range="unit")
        self.x_lf_norm = self._scaler_x.fit(self.x_lf, channel_dim=1).transform(self.x_lf)
        self.x_hf_norm = self._scaler_x.transform(self.x_hf)

        # Nearest-node anchoring from HF to LF:
        # j*(i) = argmin_j ||\hat{x}^{HF}_i - \hat{x}^{LF}_j||_2^2.
        # The unique anchor set is treated as the already selected LF subset A.
        self.hf_idxs = self._map_hf_to_lf()
        self.selected_idxs = np.unique(self.hf_idxs).astype(np.int64)
        # y_lf_at_hf provides the paired LF response used by the autoregressive model.
        self.y_lf_at_hf = self.y_lf[self.hf_idxs]

        # Autoregressive multi-fidelity decomposition:
        # y_H(x) = rho y_L(x) + delta(x),
        # Cov[y_H(x), y_H(x')] = rho^2 sigma_v^2 psi_v(x, x')
        #                      + sigma_d^2 psi_d(x, x').
        self.theta_v = self._init_theta(theta_v)
        self.theta_d = self._init_theta(theta_d) if theta_d is not None else self.theta_v.copy()
        self.rho = self._estimate_rho()
        self.sigma_sq_v = self._estimate_sigma_sq_v()
        self.sigma_sq_d = self._estimate_sigma_sq_d()
        # The MICO score later combines
        # score = delta_N * delta_D
        # to balance uncertainty and correlation information.

    # ============================================================
    # Correlation-Length Setup
    # ============================================================
    def _init_theta(self, theta: Optional[np.ndarray]) -> np.ndarray:
        if theta is not None:
            # User-provided theta is interpreted as the full inverse length-scale vector.
            return np.asarray(theta, dtype=np.float64).flatten()
        # When theta is not provided, infer one inverse length scale per dimension.
        return self._estimate_theta_from_points(self.x_lf_norm)

    def _estimate_theta_from_points(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 2:
            # Degenerate single-point pools fall back to isotropic unit length scales.
            return np.ones(self.input_dim, dtype=np.float64)

        theta = np.ones(self.input_dim, dtype=np.float64)
        for dim in range(self.input_dim):
            diffs = x[:, dim][:, None] - x[:, dim][None, :]
            sq = diffs ** 2
            nz = sq[sq > 1e-12]
            if nz.size > 0:
                # Median heuristic:
                # theta_k ~= 1 / median((x_{ik} - x_{jk})^2).
                theta[dim] = 1.0 / max(float(np.median(nz)), 1e-12)
        return theta

    # ============================================================
    # Autoregressive Coefficients
    # ============================================================
    def _estimate_rho(self) -> np.ndarray:
        rho = np.ones(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            yl = self.y_lf_at_hf[:, d]
            yh = self.y_hf[:, d]
            denom = float(yl @ yl)
            if denom > 1e-12:
                # Least-squares autoregressive bridge:
                # rho = argmin ||y_h - rho y_l||_2^2 = (y_l^T y_h) / (y_l^T y_l).
                rho[d] = float((yl @ yh) / denom)
            # rho_d links the LF trend to the HF response for output d.
        return rho

    def _estimate_sigma_sq_v(self) -> np.ndarray:
        ddof = 1 if self.y_lf.shape[0] > 1 else 0
        # sigma_v^2 models the LF process variance.
        # This is the variance term attached to psi_v in the MF covariance.
        sigma_sq_v = np.var(self.y_lf, axis=0, ddof=ddof).astype(np.float64)
        return np.maximum(sigma_sq_v, 1e-12)

    def _estimate_sigma_sq_d(self) -> np.ndarray:
        ddof = 1 if self.y_hf.shape[0] > 1 else 0
        sigma_sq_d = np.zeros(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            # Discrepancy variance uses
            # delta(x) = y_h(x) - rho y_l(x).
            discrepancy = self.y_hf[:, d] - self.rho[d] * self.y_lf_at_hf[:, d]
            sigma_sq_d[d] = max(float(np.var(discrepancy, ddof=ddof)), 1e-12)
        # sigma_d^2 is the scale of the independent discrepancy process delta(x).
        return sigma_sq_d

    # ============================================================
    # HF-To-LF Mapping
    # ============================================================
    def _map_hf_to_lf(self) -> np.ndarray:
        dists = self._compute_sq_dists(self.x_hf_norm, self.x_lf_norm)
        # Each HF point is tied to its nearest LF node in the normalized pool.
        # The mapping is discrete because the LF design is treated as a finite candidate set.
        return np.argmin(dists, axis=1).astype(np.int64)

    def _remaining_candidate_indices(self, selected_idxs: np.ndarray) -> np.ndarray:
        candidate_mask = np.ones(self.num_lf, dtype=bool)
        candidate_mask[np.asarray(selected_idxs, dtype=np.int64)] = False
        # Only unselected LF nodes remain in the greedy candidate set.
        # These are precisely the candidates over which MICO is maximized.
        return np.where(candidate_mask)[0].astype(np.int64)

    # ============================================================
    # Distance And Correlation Kernels
    # ============================================================
    def _compute_sq_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        # Use ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T.
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_sq = np.sum(c ** 2, axis=1)
        d_sq = x_sq + c_sq - 2.0 * (x @ c.T)
        np.maximum(d_sq, 0.0, out=d_sq)
        return d_sq

    def _correlation_matrix(self, x1: np.ndarray, x2: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # After scaling x by sqrt(theta), the Gaussian correlation becomes
        # psi(x_i, x_j) = exp(-sum_k theta_k (x_{ik} - x_{jk})^2).
        # The implementation first rescales coordinates and then reuses the
        # isotropic Gaussian form exp(-||x_i' - x_j'||^2).
        x1s = x1 * np.sqrt(theta)
        x2s = x2 * np.sqrt(theta)
        x1_sq = np.sum(x1s ** 2, axis=1, keepdims=True)
        x2_sq = np.sum(x2s ** 2, axis=1)
        d_sq = x1_sq + x2_sq - 2.0 * (x1s @ x2s.T)
        np.maximum(d_sq, 0.0, out=d_sq)
        return np.exp(-d_sq)

    def _mf_covariance(self, x1: np.ndarray, x2: np.ndarray, out_idx: int) -> np.ndarray:
        rho_d = self.rho[out_idx]
        sigma_v = self.sigma_sq_v[out_idx]
        sigma_d = self.sigma_sq_d[out_idx]
        psi_v = self._correlation_matrix(x1, x2, self.theta_v)
        psi_d = self._correlation_matrix(x1, x2, self.theta_d)
        # Autoregressive covariance model:
        # C(x, x') = rho^2 sigma_v^2 psi_v(x, x') + sigma_d^2 psi_d(x, x').
        # The first term propagates LF information, while the second term
        # captures the irreducible HF discrepancy.
        return (rho_d ** 2 * sigma_v) * psi_v + sigma_d * psi_d

    # ============================================================
    # MICO Score
    # ============================================================
    def _compute_mico_scores(
        self,
        candidate_idxs: np.ndarray,
        selected_idxs: List[int],
        out_idx: int,
    ) -> np.ndarray:
        candidate_idxs = np.asarray(candidate_idxs, dtype=np.int64)
        selected_idxs = np.unique(np.asarray(selected_idxs, dtype=np.int64))
        num_cands = candidate_idxs.size
        if num_cands == 0:
            return np.zeros(0, dtype=np.float64)

        # V collects the remaining LF candidates, while A collects the already
        # anchored LF nodes associated with existing HF evaluations.
        x_candidates = self.x_lf_norm[candidate_idxs]
        if selected_idxs.size == 0:
            c_vv = self._mf_covariance(x_candidates, x_candidates, out_idx)
            c_vv += np.eye(num_cands, dtype=np.float64) * 1e-6
            icov_vv = np.linalg.pinv(c_vv)
            # With no selected set, MICO reduces to the product of local variance
            # and precision: score_i = delta_N,i * delta_D,i.
            delta_n = np.maximum(np.diag(c_vv), 1e-12)
            delta_d = np.maximum(np.diag(icov_vv), 1e-12)
            # In this case delta_N comes purely from the marginal covariance C_vv.
            return delta_n * delta_d

        x_selected = self.x_lf_norm[selected_idxs]
        c_aa = self._mf_covariance(x_selected, x_selected, out_idx)
        c_aa += np.eye(selected_idxs.size, dtype=np.float64) * 1e-6
        icov_a = np.linalg.pinv(c_aa)
        # C_aa is the covariance among the already selected LF subset A.

        c_ya = self._mf_covariance(x_candidates, x_selected, out_idx)
        c_vv = self._mf_covariance(x_candidates, x_candidates, out_idx)
        c_vv += np.eye(num_cands, dtype=np.float64) * 1e-6
        icov_vv = np.linalg.pinv(c_vv)
        # C_ya is the cross-covariance between remaining candidates V and selected nodes A.

        # Conditional covariance of a candidate given the selected set:
        # Sigma_{V|A} = C_vv - C_va C_aa^{-1} C_av.
        # The diagonal of Sigma_{V|A} is the residual variance after
        # conditioning on the already sampled subset A.
        temp = c_ya @ icov_a @ c_ya.T
        delta_n = np.maximum(np.diag(c_vv - temp), 1e-12)
        # The precision diagonal acts as the correlation-information term.
        delta_d = np.maximum(np.diag(icov_vv), 1e-12)
        # MICO score:
        # MICO_i = delta_N,i * delta_D,i.
        # High scores therefore favor candidates that are both uncertain and
        # weakly explained by the current selected subset.
        return delta_n * delta_d

    def _compute_full_pool_scores(self, out_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # This helper evaluates MICO on the full remaining LF pool C = X_LF \ A.
        candidates = self._remaining_candidate_indices(self.selected_idxs)
        scores = self._compute_mico_scores(candidates, self.selected_idxs.tolist(), out_idx)
        return candidates, scores

    # ============================================================
    # Continuous Query Evaluation
    # ============================================================
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the MICO score at continuous coordinates.

        Args:
            x (np.ndarray): Query coordinates. (N, input_dim).

        Returns:
            np.ndarray: MICO scores. (N, 1).
        """
        x = np.asarray(x, dtype=np.float64)
        x_norm = self._scaler_x.transform(x)
        dists = self._compute_sq_dists(x_norm, self.x_lf_norm)
        # Nearest-node projection:
        # j*(x) = argmin_j ||\hat{x} - \hat{x}^{LF}_j||_2^2.
        # Continuous queries are snapped to the nearest discrete LF pool node.
        idxs = np.argmin(dists, axis=1).astype(np.int64)

        candidates, scores = self._compute_full_pool_scores(self.target_idx)
        score_map = {int(idx): float(score) for idx, score in zip(candidates, scores)}
        # Unselected LF nodes keep their MICO score; previously selected nodes map to 0.
        values = np.array([score_map.get(int(idx), 0.0) for idx in idxs], dtype=np.float64)
        return values.reshape(-1, 1)

    # ============================================================
    # Greedy Proposal
    # ============================================================
    def propose(self) -> np.ndarray:
        """
        Greedily return the remaining LF node with the largest MICO score.

        Returns:
            np.ndarray: Proposed design point. (1, input_dim).
        """
        # Greedy discrete update:
        # x_next = argmax_{j in C} MICO_j.
        candidates, scores = self._compute_full_pool_scores(self.target_idx)
        if candidates.size == 0:
            # If every LF node is already selected, fall back to a random pool node.
            fallback = int(np.random.randint(self.num_lf))
            return self.x_lf[fallback].reshape(1, -1)

        # Greedy MICO chooses the remaining LF node with the maximum score.
        best_idx = int(candidates[int(np.argmax(scores))])
        # The returned point is still expressed in the original, unnormalized design space.
        return self.x_lf[best_idx].reshape(1, -1)
