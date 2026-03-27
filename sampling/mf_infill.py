# Multi-Fidelity Infill via MICO (Mutual Information and Correlation Criterion)
# Paper reference: https://doi.org/10.1007/s00366-023-01858-z
# Paper author: Shuo Wang, Xiaonan Lai, Xiwang He, Kunpeng Li, Liye Lv, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from typing import List, Optional

from sampling.base_infill import BaseInfill
from utils.scaler import MinMaxScalerNP


class MultiFidelityInfill(BaseInfill):
    """
    Multi-fidelity infill via the MICO (Mutual Information and Correlation) criterion.

    Proposes new high-fidelity sample locations by selecting from a discrete
    low-fidelity candidate pool. The MICO score maximises mutual information
    between selected and unobserved locations, leveraging the multi-fidelity
    covariance structure derived from co-kriging.

    Unlike single- and multi-objective infill, ``propose()`` performs a greedy
    search over the discrete ``x_lf`` pool rather than continuous L-BFGS-B
    optimisation. ``evaluate()`` accepts continuous coordinates and maps each
    query point to the nearest LF pool location before scoring.

    MICO criterion (Eq. 40 + 57 in Wang et al. 2024):

    .. code-block::

        score = delta_n * delta_d

        delta_n = diag(C_vv - C_yA @ C_AA^{-1} @ C_Ay)   # conditional variance
        delta_d = diag(C_vv^{-1})                         # inverse self-covariance

    Attributes:
        model: Pre-trained Kriging model on HF data.
        x_hf (np.ndarray): HF training locations. shape: (num_hf, input_dim).
        y_hf (np.ndarray): HF training responses. shape: (num_hf, output_dim).
        x_lf (np.ndarray): LF candidate pool. shape: (num_lf, input_dim).
        y_lf (np.ndarray): LF responses at pool locations. shape: (num_lf, output_dim).
        target_idx (int): Output dimension used to score candidates.
        ratio (float): Weight for MI term vs. distance-diversity term, in [0, 1].
        theta_v (np.ndarray): LF process correlation lengths. shape: (input_dim,).
        theta_d (np.ndarray): Discrepancy correlation lengths. shape: (input_dim,).
        rho (np.ndarray): HF/LF scaling factors. shape: (output_dim,).
        sigma_sq_v (np.ndarray): LF process variance. shape: (output_dim,).
        sigma_sq_d (np.ndarray): Discrepancy variance. shape: (output_dim,).
        hf_idxs (np.ndarray): LF pool indices nearest to each HF observation.
            shape: (num_hf,).
    """

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
        Initialize the multi-fidelity MICO infill strategy.

        Args:
            model: A fitted Kriging model on HF data.
            x_hf (np.ndarray): HF training locations. shape: (num_hf, input_dim).
            y_hf (np.ndarray): HF training responses. shape: (num_hf, output_dim).
            x_lf (np.ndarray): LF candidate pool (discrete node locations).
                shape: (num_lf, input_dim).
            y_lf (np.ndarray): LF responses at candidate locations.
                shape: (num_lf, output_dim).
            target_idx (int): Output dimension to optimise. Default 0.
            ratio (float): Weight of MI term vs. diversity term, in [0, 1].
                ratio=1 is pure MI; ratio=0 is pure distance-diversity. Default 0.5.
            theta_v (Optional[np.ndarray]): Correlation lengths for the LF process.
                If None, extracted from ``model.theta``. shape: (input_dim,).
            theta_d (Optional[np.ndarray]): Correlation lengths for the discrepancy
                process. If None, copied from ``theta_v``. shape: (input_dim,).

        Raises:
            RuntimeError: If ``model`` is not fitted.
        """
        super().__init__(model, bounds=None, target_idx=target_idx, num_restarts=0)

        self.x_hf = np.array(x_hf, dtype=np.float64)
        self.y_hf = np.array(y_hf, dtype=np.float64)
        self.x_lf = np.array(x_lf, dtype=np.float64)
        self.y_lf = np.array(y_lf, dtype=np.float64)

        self.num_lf, self.input_dim = self.x_lf.shape
        self.num_hf = self.x_hf.shape[0]
        self.output_dim = self.y_lf.shape[1]

        self.ratio = float(ratio)

        self.theta_v = self._init_theta(theta_v)
        self.theta_d = (
            self._init_theta(theta_d) if theta_d is not None
            else self.theta_v.copy()
        )

        self.rho        = self._estimate_rho()
        self.sigma_sq_v = self._estimate_sigma_sq_v()
        self.sigma_sq_d = self._estimate_sigma_sq_d()

        self.hf_idxs = self._map_hf_to_lf()
        self._scaler_dist = MinMaxScalerNP(norm_range="unit")

    # ------------------------------------------------------------------
    # Private initialisation helpers
    # ------------------------------------------------------------------

    def _init_theta(self, theta: Optional[np.ndarray]) -> np.ndarray:
        """
        Resolve correlation length parameters from user input or model attribute.

        Args:
            theta (Optional[np.ndarray]): User-supplied theta, or None.

        Returns:
            np.ndarray: Correlation lengths. shape: (input_dim,).
        """
        if theta is not None:
            return np.array(theta, dtype=np.float64).flatten()

        if hasattr(self.model, "theta") and self.model.theta is not None:
            model_theta = self.model.theta
            if np.isscalar(model_theta) or np.asarray(model_theta).size == 1:
                return np.full(self.input_dim, float(model_theta), dtype=np.float64)
            return np.array(model_theta, dtype=np.float64).flatten()

        return np.ones(self.input_dim, dtype=np.float64)

    def _estimate_rho(self) -> np.ndarray:
        """
        Estimate the HF/LF scaling factor ``rho`` from observed HF data.

        Returns:
            np.ndarray: Per-output scaling factors. shape: (output_dim,).
        """
        y_lf_at_hf, _ = self.model.predict(self.x_hf)
        rho = np.ones(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            yl = y_lf_at_hf[:, d]
            yh = self.y_hf[:, d]
            mask = np.abs(yl) > 1e-10
            if np.any(mask):
                rho[d] = np.mean(yh[mask] / yl[mask])
        return rho

    def _estimate_sigma_sq_v(self) -> np.ndarray:
        """
        Estimate the LF process variance from the fitted model or LF data.

        Returns:
            np.ndarray: LF variance per output. shape: (output_dim,).
        """
        if hasattr(self.model, "sigma2") and self.model.sigma2 is not None:
            sigma2 = self.model.sigma2
            if np.isscalar(sigma2):
                return np.full(self.output_dim, float(sigma2), dtype=np.float64)
            return np.array(sigma2, dtype=np.float64).flatten()
        return np.var(self.y_lf, axis=0).astype(np.float64)

    def _estimate_sigma_sq_d(self) -> np.ndarray:
        """
        Estimate the discrepancy variance from initial HF observations.

        Discrepancy is defined as ``y_hf - rho * y_lf_predicted``.

        Returns:
            np.ndarray: Discrepancy variance per output. shape: (output_dim,).
        """
        y_lf_at_hf, _ = self.model.predict(self.x_hf)
        sigma_sq_d = np.zeros(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            discrepancy = self.y_hf[:, d] - self.rho[d] * y_lf_at_hf[:, d]
            sigma_sq_d[d] = np.var(discrepancy)
        return sigma_sq_d

    def _map_hf_to_lf(self) -> np.ndarray:
        """
        Map each HF location to the nearest LF pool index (nearest-neighbour).

        Returns:
            np.ndarray: LF pool indices for each HF location. shape: (num_hf,).
        """
        dists = self._compute_sq_dists(self.x_hf, self.x_lf)
        return np.argmin(dists, axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    # Covariance helpers
    # ------------------------------------------------------------------

    def _compute_sq_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances via the algebraic expansion.

        ||x_i - c_j||^2 = ||x_i||^2 + ||c_j||^2 - 2 x_i c_j^T

        Args:
            x (np.ndarray): Query points. shape: (n_x, num_features).
            c (np.ndarray): Reference points. shape: (n_c, num_features).

        Returns:
            np.ndarray: Squared distance matrix. shape: (n_x, n_c).
        """
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_sq = np.sum(c ** 2, axis=1)
        d_sq = x_sq + c_sq - 2.0 * (x @ c.T)
        np.maximum(d_sq, 0.0, out=d_sq)
        return d_sq

    def _correlation_matrix(
        self, x1: np.ndarray, x2: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute the anisotropic Gaussian correlation matrix.

        Psi[i, j] = exp(-sum_k theta_k * (x1[i,k] - x2[j,k])^2)

        Matches Eq. 47-48 in Wang et al. (2024).

        Args:
            x1 (np.ndarray): First point set. shape: (n1, input_dim).
            x2 (np.ndarray): Second point set. shape: (n2, input_dim).
            theta (np.ndarray): Correlation lengths. shape: (input_dim,).

        Returns:
            np.ndarray: Correlation matrix. shape: (n1, n2).
        """
        x1s = x1 * np.sqrt(theta)
        x2s = x2 * np.sqrt(theta)
        x1_sq = np.sum(x1s ** 2, axis=1, keepdims=True)
        x2_sq = np.sum(x2s ** 2, axis=1)
        d_sq = x1_sq + x2_sq - 2.0 * (x1s @ x2s.T)
        np.maximum(d_sq, 0.0, out=d_sq)
        return np.exp(-d_sq)

    def _mf_covariance(
        self, x1: np.ndarray, x2: np.ndarray, out_idx: int
    ) -> np.ndarray:
        """
        Compute the multi-fidelity co-kriging covariance matrix.

        C_HF = rho^2 * sigma_sq_v * Psi_v + sigma_sq_d * Psi_d

        Matches Eq. 16 in Wang et al. (2024).

        Args:
            x1 (np.ndarray): First point set. shape: (n1, input_dim).
            x2 (np.ndarray): Second point set. shape: (n2, input_dim).
            out_idx (int): Index of the output dimension.

        Returns:
            np.ndarray: Covariance matrix. shape: (n1, n2).
        """
        rho_d   = self.rho[out_idx]
        sigma_v = self.sigma_sq_v[out_idx]
        sigma_d = self.sigma_sq_d[out_idx]
        psi_v   = self._correlation_matrix(x1, x2, self.theta_v)
        psi_d   = self._correlation_matrix(x1, x2, self.theta_d)
        return (rho_d ** 2 * sigma_v) * psi_v + sigma_d * psi_d

    # ------------------------------------------------------------------
    # Core MICO Computation
    # ------------------------------------------------------------------

    def _compute_mico_scores(
        self,
        candidate_idxs: np.ndarray,
        selected_idxs: List[int],
        out_idx: int,
    ) -> np.ndarray:
        """
        Compute raw MICO ``delta = delta_n * delta_d`` scores for candidates.

        Shared helper used by both ``evaluate()`` and ``propose()``, eliminating
        the code duplication present in the original implementation.

        Args:
            candidate_idxs (np.ndarray): Indices into ``x_lf`` to score.
                shape: (num_cands,).
            selected_idxs (List[int]): Indices into ``x_lf`` already chosen
                (seed from nearest-neighbour HF mapping).
            out_idx (int): Output dimension index.

        Returns:
            np.ndarray: MICO delta scores. shape: (num_cands,).
                Returns zeros if no candidates are provided.
        """
        num_cands = len(candidate_idxs)
        if num_cands == 0:
            return np.zeros(0, dtype=np.float64)

        x_selected   = self.x_lf[selected_idxs]
        x_candidates = self.x_lf[candidate_idxs]

        # Covariance of the already-selected set: C_AA
        num_sel = len(selected_idxs)
        c_aa = self._mf_covariance(x_selected, x_selected, out_idx)
        c_aa += np.eye(num_sel, dtype=np.float64) * 1e-6

        try:
            icov_a = np.linalg.inv(c_aa)
        except np.linalg.LinAlgError:
            icov_a = np.linalg.pinv(c_aa)

        # Cross-covariance candidates → selected: C_yA
        c_ya = self._mf_covariance(x_candidates, x_selected, out_idx)  # (num_cands, num_sel)

        # Self-covariance of candidates: C_vv
        c_vv = self._mf_covariance(x_candidates, x_candidates, out_idx)  # (num_cands, num_cands)
        c_vv += np.eye(num_cands, dtype=np.float64) * 1e-6

        try:
            icov_vv = np.linalg.inv(c_vv)
        except np.linalg.LinAlgError:
            icov_vv = np.linalg.pinv(c_vv)

        # MICO: delta_n = diag(C_vv - C_yA @ C_AA^{-1} @ C_yA^T)
        #       delta_d = diag(C_vv^{-1})
        temp    = c_ya @ icov_a @ c_ya.T
        delta_n = np.maximum(np.diag(c_vv - temp), 1e-12)
        delta_d = np.maximum(np.diag(icov_vv), 1e-12)
        return delta_n * delta_d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the MICO acquisition score at continuous design coordinates.

        Each query point is mapped to the nearest LF pool location before
        scoring, making the API consistent with other ``BaseInfill`` subclasses.
        Scores are **not** normalised (raw delta values); use ``propose()``
        for normalised greedy pool selection.

        Args:
            x (np.ndarray): Query coordinates. shape: (N, input_dim).

        Returns:
            np.ndarray: MICO scores for the target output dimension.
                shape: (N, 1).
        """
        x = np.array(x, dtype=np.float64)
        dists = self._compute_sq_dists(x, self.x_lf)
        idxs  = np.argmin(dists, axis=1).astype(np.int64)

        selected_idxs = list(self.hf_idxs.copy())
        return self._compute_mico_scores(idxs, selected_idxs, self.target_idx).reshape(-1, 1)

    def propose(self) -> np.ndarray:
        """
        Propose one new HF sampling location from the LF candidate pool.

        Algorithm (Eq. 40 + 57 in Wang et al. 2024):
        1. Seed selected set with nearest-neighbour LF matches of HF points.
        2. Compute MICO delta scores for all remaining pool candidates.
        3. Compute distance-diversity scores (min dist to selected set).
        4. Normalise both scores to [0, 1] and combine with ratio weight.
        5. Return the LF location achieving the highest combined score.

        Returns:
            np.ndarray: Coordinates of the proposed point. shape: (1, input_dim).
        """
        selected_idxs = list(self.hf_idxs.copy())

        # Candidate set: pool locations not already selected
        candidate_mask = np.ones(self.num_lf, dtype=bool)
        candidate_mask[selected_idxs] = False
        candidates = np.where(candidate_mask)[0]

        if len(candidates) == 0:
            fallback = int(np.random.randint(self.num_lf))
            return self.x_lf[fallback].reshape(1, -1)

        # MICO scores for all candidates
        delta = self._compute_mico_scores(candidates, selected_idxs, self.target_idx)

        d_min, d_max = np.min(delta), np.max(delta)
        delta_norm = (
            (delta - d_min) / (d_max - d_min)
            if d_max > d_min
            else np.ones_like(delta)
        )

        # Distance-diversity score: min Euclidean distance to selected set
        x_candidates = self.x_lf[candidates]
        x_selected   = self.x_lf[selected_idxs]
        dists_all    = self._compute_sq_dists(x_candidates, x_selected)
        dists        = np.min(dists_all, axis=1)

        if len(dists) > 1 and np.max(dists) > np.min(dists):
            dists_norm = self._scaler_dist.fit(
                dists.reshape(-1, 1), channel_dim=0
            ).transform(dists.reshape(-1, 1)).flatten()
        else:
            dists_norm = np.zeros_like(dists)

        criterion = self.ratio * delta_norm + (1.0 - self.ratio) * dists_norm
        best_in_candidates = int(np.argmax(criterion))
        best_idx = int(candidates[best_in_candidates])

        return self.x_lf[best_idx].reshape(1, -1)
