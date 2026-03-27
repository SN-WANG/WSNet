# Multi-Objective Infill via IS-EHVI (Importance Sampling Expected Hypervolume Improvement)
# Paper reference: https://doi.org/10.1109/TEVC.2022.3228516
# Paper author: Yong Pang, Yitang Wang, Shuai Zhang, Xiaonan Lai, Wei Sun, Xueguan Song
# Code author: Shengning Wang

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from typing import List, Optional

from sampling.base_infill import BaseInfill


class MultiObjectiveInfill(BaseInfill):
    """
    Multi-objective infill via IS-EHVI with constraint handling (cEHVI).

    Implements the algorithm from Pang et al. (2023), proposing new sample
    locations to efficiently improve the Pareto front of expensive black-box
    functions. Constraint handling uses Probability of Feasibility (PoF).

    Key formula (Eq. 12, with corrected N_E denominator from Eq. 8 typo):
        EHVI(x) ≈ (1/N_E) * Σ_{q_i ∈ S_nd} HVI(q_i) · PDF_F(q_i | x)

    where:
        N_E    = total IS samples (denominator — NOT N_nd, paper typo in Eq. 8)
        S_nd   = uniform samples NOT dominated by current Pareto front
        HVI    = MC hypervolume contribution (count of S_nd weakly dominated)
        PDF_F  = product of normalised Gaussian PDFs (Eq. 17)

    ``evaluate(x)`` returns cEHVI scores for arbitrary continuous coordinates.
    ``propose()`` overrides the base: diversity-selection within the top-w%
    by cEHVI, followed by L-BFGS-B local refinement (Eq. 19-20).

    Attributes:
        model: Pre-trained Kriging surrogate model.
        bounds (np.ndarray): Design variable bounds. shape: (num_features, 2).
        obj_idxs (List[int]): Output indices minimised as objectives.
        constraint_idxs (Optional[List[int]]): Output indices for constraints.
        constraint_ubs (Optional[np.ndarray]): Upper bounds for each constraint.
        num_samples (int): Total IS sample count (N_E). Default 5000.
        num_candidates (int): Candidate pool size in ``propose()``. Default 200.
        num_restarts (int): L-BFGS-B restarts for local refinement. Default 5.
        beta (float): Diversity selection hyperparameter. Default 0.3.
        obj_lb (float): IS sample lower bound (normalised space). Default -0.5.
        obj_ub (float): IS sample upper bound (normalised space). Default 1.2.
        y_obj_min (np.ndarray): Per-objective minimum from training data. shape: (M,).
        y_obj_range (np.ndarray): Per-objective range from training data. shape: (M,).
        pf_norm (np.ndarray): Normalised Pareto front. shape: (n_pf, M).
        nd_samples (np.ndarray): Non-dominated IS samples in normalised obj space. shape: (N_nd, M).
        hvi_nd (np.ndarray): HVI weights for each nd_samples point. shape: (N_nd,).
    """

    def __init__(
        self,
        model,
        bounds,
        y_train: np.ndarray,
        obj_idxs: List[int],
        constraint_idxs: Optional[List[int]] = None,
        constraint_ubs: Optional[np.ndarray] = None,
        num_samples: int = 5000,
        num_candidates: int = 200,
        num_restarts: int = 5,
        beta: float = 0.3,
        obj_lb: float = -0.5,
        obj_ub: float = 1.2,
    ) -> None:
        """
        Initialize the multi-objective IS-EHVI infill strategy.

        Args:
            model: A fitted Kriging model instance.
            bounds: Design variable bounds. shape: (num_features, 2).
            y_train (np.ndarray): Training output data. shape: (n_train, num_outputs).
            obj_idxs (List[int]): Output indices treated as objectives (minimised).
            constraint_idxs (Optional[List[int]]): Output indices for constraints.
            constraint_ubs (Optional[np.ndarray]): Upper bounds for constraint outputs.
            num_samples (int): Number of uniform IS samples (N_E). Default 5000.
            num_candidates (int): Candidate pool size in ``propose()``. Default 200.
            num_restarts (int): L-BFGS-B restarts for local refinement. Default 5.
            beta (float): Diversity selection hyperparameter. Default 0.3.
            obj_lb (float): Lower bound for IS sample generation (normalised). Default -0.5.
            obj_ub (float): Upper bound for IS sample generation (normalised). Default 1.2.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If ``constraint_idxs`` and ``constraint_ubs`` are inconsistent.
        """
        super().__init__(model, bounds=bounds, target_idx=0, num_restarts=num_restarts)

        self.obj_idxs        = list(obj_idxs)
        self.constraint_idxs = list(constraint_idxs) if constraint_idxs else None

        if constraint_ubs is not None:
            self.constraint_ubs = np.array(constraint_ubs, dtype=np.float64)
            if (self.constraint_idxs is not None
                    and len(self.constraint_ubs) != len(self.constraint_idxs)):
                raise ValueError(
                    f"constraint_ubs length {len(self.constraint_ubs)} must match "
                    f"constraint_idxs length {len(self.constraint_idxs)}."
                )
        else:
            self.constraint_ubs = None

        self.num_samples    = int(num_samples)
        self.num_candidates = int(num_candidates)
        self.beta           = float(beta)
        self.obj_lb         = float(obj_lb)
        self.obj_ub         = float(obj_ub)

        y_obj = np.array(y_train[:, self.obj_idxs], dtype=np.float64)
        self.y_obj_min   = np.min(y_obj, axis=0)
        self.y_obj_max   = np.max(y_obj, axis=0)
        self.y_obj_range = self.y_obj_max - self.y_obj_min
        self.y_obj_range = np.where(self.y_obj_range < 1e-12, 1.0, self.y_obj_range)

        self._precompute_samples(y_obj)

    # ------------------------------------------------------------------
    # Pareto Utility
    # ------------------------------------------------------------------

    def _compute_pareto_mask(self, y: np.ndarray) -> np.ndarray:
        """
        Compute non-dominance mask for minimisation objectives.

        Point i is non-dominated iff no other point j satisfies:
            y[j, k] <= y[i, k]  for all objectives k, and
            y[j, k] <  y[i, k]  for at least one objective k.

        Args:
            y (np.ndarray): Objective values. shape: (n, M).

        Returns:
            np.ndarray: Boolean mask, True = non-dominated. shape: (n,).
        """
        y_i = y[:, np.newaxis, :]
        y_j = y[np.newaxis, :, :]
        diff = y_j - y_i
        dominated_by_j = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
        np.fill_diagonal(dominated_by_j, False)
        return ~np.any(dominated_by_j, axis=1)

    # ------------------------------------------------------------------
    # Pre-computation (IS samples + HVI weights)
    # ------------------------------------------------------------------

    def _precompute_samples(self, y_obj: np.ndarray) -> None:
        """
        Generate IS samples and compute HVI weights (called once at init).

        Steps:
        1. Draw num_samples uniform samples in [obj_lb, obj_ub]^M (normalised space).
        2. Compute normalised Pareto front from training objectives.
        3. Filter to nd_samples: rows NOT dominated by any PF point.
        4. Compute hvi_nd[i] = count of nd_samples weakly dominated by nd_samples[i]
           (MC hypervolume contribution, Eq. 7).

        Args:
            y_obj (np.ndarray): Training objective values. shape: (n_train, M).
        """
        num_obj    = len(self.obj_idxs)
        chunk_size = 1000

        is_samples = np.random.uniform(
            self.obj_lb, self.obj_ub, size=(self.num_samples, num_obj)
        )

        pf_norm_all = (y_obj - self.y_obj_min) / self.y_obj_range
        pf_mask     = self._compute_pareto_mask(pf_norm_all)
        self.pf_norm = pf_norm_all[pf_mask]

        nd_mask = np.ones(self.num_samples, dtype=bool)
        for start in range(0, self.num_samples, chunk_size):
            end    = min(start + chunk_size, self.num_samples)
            chunk  = is_samples[start:end]
            s_exp  = chunk[:, np.newaxis, :]
            pf_exp = self.pf_norm[np.newaxis, :, :]
            diff   = pf_exp - s_exp
            dominated = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
            nd_mask[start:end] = ~np.any(dominated, axis=1)

        self.nd_samples = is_samples[nd_mask]
        num_nd          = self.nd_samples.shape[0]

        self.hvi_nd = np.zeros(num_nd, dtype=np.float64)
        for start in range(0, num_nd, chunk_size):
            end   = min(start + chunk_size, num_nd)
            batch = self.nd_samples[start:end]
            b_exp = batch[:, np.newaxis, :]
            s_exp = self.nd_samples[np.newaxis, :, :]
            diff  = s_exp - b_exp
            self.hvi_nd[start:end] = np.sum(
                np.all(diff >= 0, axis=2), axis=1
            ).astype(np.float64)

    # ------------------------------------------------------------------
    # Acquisition Functions
    # ------------------------------------------------------------------

    def _compute_pof_batch(
        self, mu_c: np.ndarray, sigma_c: np.ndarray
    ) -> np.ndarray:
        """
        Compute Probability of Feasibility (PoF) for constraint satisfaction.

        PoF(x) = Π_j Φ((ub_j - μ_j(x)) / σ_j(x))

        Args:
            mu_c (np.ndarray): Constraint output means. shape: (num_cands, num_constraints).
            sigma_c (np.ndarray): Constraint output std devs. shape: (num_cands, num_constraints).

        Returns:
            np.ndarray: PoF values. shape: (num_cands,).
        """
        z = (self.constraint_ubs[np.newaxis, :] - mu_c) / sigma_c
        return np.prod(norm.cdf(z), axis=1)

    def _compute_ehvi_batch(self, x_candidates: np.ndarray) -> np.ndarray:
        """
        Compute cEHVI (or EHVI) for a batch of design candidates.

        Implements Eq. 12 from Pang et al. (2023) with corrected N_E denominator:
            EHVI(x) ≈ (1/N_E) * Σ_{q_i ∈ S_nd} HVI(q_i) · PDF_F(q_i | x)

        Normalisation follows Eq. 15-17; log-space accumulation for stability.
        If ``constraint_idxs`` are set, multiplies by PoF to yield cEHVI.

        Args:
            x_candidates (np.ndarray): Candidate design points. shape: (num_cands, num_features).

        Returns:
            np.ndarray: cEHVI values per candidate. shape: (num_cands, 1).
        """
        num_cands = x_candidates.shape[0]
        num_nd    = self.nd_samples.shape[0]

        if num_nd == 0:
            return np.zeros(num_cands)

        mu_raw, var_raw = self.model.predict(x_candidates)

        mu_obj    = mu_raw[:, self.obj_idxs]
        var_obj   = var_raw[:, self.obj_idxs]
        zero_var  = np.all(var_obj < 1e-12, axis=1)
        sigma_obj = np.sqrt(np.maximum(var_obj, 1e-12))

        mu_norm    = (mu_obj - self.y_obj_min) / self.y_obj_range
        sigma_norm = sigma_obj / self.y_obj_range

        s_exp     = self.nd_samples[np.newaxis, :, :]
        mu_exp    = mu_norm[:, np.newaxis, :]
        sigma_exp = sigma_norm[:, np.newaxis, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (s_exp - mu_exp) / sigma_exp

        log_phi   = norm.logpdf(z)
        log_pdf_f = np.sum(log_phi, axis=2)
        pdf_f     = np.exp(log_pdf_f)

        ehvi           = (pdf_f @ self.hvi_nd) / self.num_samples
        ehvi[zero_var] = 0.0

        if self.constraint_idxs is not None and self.constraint_ubs is not None:
            mu_c    = mu_raw[:, self.constraint_idxs]
            var_c   = var_raw[:, self.constraint_idxs]
            sigma_c = np.sqrt(np.maximum(var_c, 1e-12))
            pof     = self._compute_pof_batch(mu_c, sigma_c)
            ehvi    = ehvi * pof

        return ehvi.reshape(-1, 1)

    def _compute_diversity_batch(self, mu_obj_norm: np.ndarray) -> np.ndarray:
        """
        Compute diversity score: min distance to Pareto front in normalised space.

        d(x) = min_{y_PF ∈ PF} ||μ_obj_norm(x) - y_PF||_2  (Eq. 19)

        Args:
            mu_obj_norm (np.ndarray): Normalised predicted objectives. shape: (num_cands, M).

        Returns:
            np.ndarray: Min-distance-to-PF per candidate. shape: (num_cands,).
        """
        diff  = mu_obj_norm[:, np.newaxis, :] - self.pf_norm[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.min(dists, axis=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate cEHVI at arbitrary continuous design coordinates.

        Args:
            x (np.ndarray): Design points. shape: (N, num_features).

        Returns:
            np.ndarray: cEHVI values. shape: (N, 1).
        """
        return self._compute_ehvi_batch(x)

    def propose(self) -> np.ndarray:
        """
        Propose one new sampling point via IS-EHVI and diversity selection.

        Algorithm (Eq. 19-20 from Pang et al. 2023):
        1. Generate ``num_candidates`` uniform random candidates in design space.
        2. Compute cEHVI for all candidates via ``evaluate()``.
        3. Select top-w% subset P_w (w ~ U(0, beta), min 5%).
        4. Among P_w, choose x_new = argmax diversity (min distance to PF).
        5. Local L-BFGS-B refinement from best candidate.

        Returns:
            np.ndarray: Proposed design point. shape: (1, num_features).
        """
        num_features = self.bounds.shape[0]

        x_candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.num_candidates, num_features),
        )

        ehvi = self.evaluate(x_candidates)

        w     = np.random.uniform(0.0, self.beta)
        frac  = max(w, 0.05)
        n_top = max(1, int(self.num_candidates * frac))
        top_idxs       = np.argsort(ehvi[:, 0])[::-1][:n_top]
        p_w            = x_candidates[top_idxs]

        mu_raw_pw, _  = self.model.predict(p_w)
        mu_obj_pw     = mu_raw_pw[:, self.obj_idxs]
        mu_norm_pw    = (mu_obj_pw - self.y_obj_min) / self.y_obj_range
        diversity     = self._compute_diversity_batch(mu_norm_pw)
        best_candidate = p_w[int(np.argmax(diversity))]

        scipy_bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])

        def neg_ehvi(x_vec: np.ndarray) -> float:
            return -float(self.evaluate(x_vec[np.newaxis, :])[0, 0])

        best_x   = best_candidate.copy()
        best_val = -neg_ehvi(best_x)

        for _ in range(self.num_restarts):
            try:
                res = minimize(neg_ehvi, x0=best_candidate, bounds=scipy_bounds,
                               method="L-BFGS-B")
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_x   = res.x
            except Exception:
                continue

        if best_x is None:
            warnings.warn(
                "L-BFGS-B failed entirely; returning diversity-selected candidate.",
                RuntimeWarning,
            )
            best_x = best_candidate

        return best_x[np.newaxis, :]
