# Distance-informed single-objective infill criteria
# Author: Shengning Wang

import numpy as np
from typing import List, Optional, Union

from sampling.so_infill import SingleObjectiveInfill


class DISOInfill(SingleObjectiveInfill):
    """
    Distance-informed single-objective infill for sequential sampling.

    This strategy keeps the base single-objective acquisition from
    ``SingleObjectiveInfill`` and multiplies it by a distance penalty factor
    computed from the nearest existing sample in the normalized design space.

    The default use case is the distance-informed EI criterion:

    .. code-block::

        d_min(x) = min_i ||x - x_i||
        P_d(x) = 1 - exp(-alpha * d_min(x) / h)
        EId(x) = EI(x) * P_d(x)

    Candidates with ``d_min(x) < min_distance`` are suppressed.
    """

    # ============================================================
    # Initialization
    # ============================================================
    def __init__(
        self,
        model,
        bounds: Union[List[float], np.ndarray],
        x_train: np.ndarray,
        y_train: np.ndarray,
        criterion: str = "ei",
        target_idx: int = 0,
        num_restarts: int = 10,
        kappa: float = 2.0,
        alpha: float = 2.0,
        min_distance: float = 0.0,
        distance_scale: Optional[float] = None,
    ) -> None:
        """
        Initialize the distance-informed infill strategy.

        Args:
            model: A fitted Kriging model instance.
            bounds (Union[List[float], np.ndarray]): Design space bounds. (D, 2).
            x_train (np.ndarray): Existing sample locations. (N, D).
            y_train (np.ndarray): Existing sample responses. (N, M).
            criterion (str): Base acquisition name. Default ``"ei"``.
            target_idx (int): Output index to optimize. Default ``0``.
            num_restarts (int): Number of L-BFGS-B restarts. Default ``10``.
            kappa (float): LCB exploration coefficient. Default ``2.0``.
            alpha (float): Distance penalty intensity. Default ``2.0``.
            min_distance (float): Minimum normalized nearest distance. Default ``0.0``.
            distance_scale (Optional[float]): Distance scale ``h``. If ``None``,
                use the maximum nearest-neighbor distance among existing samples.
        """
        super().__init__(
            model=model,
            bounds=bounds,
            y_train=y_train,
            criterion=criterion,
            target_idx=target_idx,
            num_restarts=num_restarts,
            kappa=kappa,
        )
        # The training archive is stored as
        # X = {x_i}_{i=1}^N,  \hat{X} = {\hat{x}_i}_{i=1}^N,
        # with \hat{x}_i = (x_i - l) / (u - l).
        # All subsequent distance terms are computed in the normalized space.
        self.x_train = np.asarray(x_train, dtype=np.float64)
        self.alpha = float(alpha)
        self.min_distance = float(min_distance)
        self.distance_scale = self._resolve_distance_scale(distance_scale)
        self._x_train_norm = self._normalize_points(self.x_train)
        # The final infill utility implemented by this class is
        # U_DISO(x) = U_base(x) * P_d(x) * 1_{d_min(x) >= d_0}.

    # ============================================================
    # Unit-Hypercube Mapping
    # ============================================================
    def _normalize_points(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize design points to the unit hypercube.

        Args:
            x (np.ndarray): Design points. (N, D).

        Returns:
            np.ndarray: Normalized design points. (N, D).
        """
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - self.bounds[:, 0]
        # Affine normalization:
        # \hat{x} = (x - l) / (u - l).
        # This removes unit inconsistency between coordinates before
        # Euclidean distances are evaluated.
        # Map the design space to [0, 1]^D so distances are dimensionless and comparable.
        return (np.asarray(x, dtype=np.float64) - lower) / span

    # ============================================================
    # Distance Scale h
    # ============================================================
    def _resolve_distance_scale(self, distance_scale: Optional[float]) -> float:
        """
        Resolve the distance scale ``h`` used in the penalty factor.

        Args:
            distance_scale (Optional[float]): User-specified distance scale.

        Returns:
            float: Positive distance scale.
        """
        if distance_scale is not None:
            # User-specified h overrides the default nearest-neighbor estimate.
            return max(float(distance_scale), 1.0e-12)

        if self.x_train.shape[0] < 2:
            # With only one sample, any positive h gives the same monotone penalty shape.
            return 1.0

        # When h is not supplied, use
        # h = max_i min_{j != i} ||\hat{x}_i - \hat{x}_j||_2,
        # i.e. the largest nearest-neighbor gap in the current design.
        x_norm = self._normalize_points(self.x_train)
        diff = x_norm[:, np.newaxis, :] - x_norm[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=1)
        # Default scale h is the largest nearest-neighbor spacing in the current design.
        return max(float(np.max(nearest)), 1.0e-12)

    # ============================================================
    # Nearest-Neighbor Radius
    # ============================================================
    def _compute_nearest_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the nearest normalized distance to the existing sample set.

        Args:
            x (np.ndarray): Query points. (N, D).

        Returns:
            np.ndarray: Nearest distances. (N, 1).
        """
        x_norm = self._normalize_points(x)
        diff = x_norm[:, np.newaxis, :] - self._x_train_norm[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # d_min(x) = min_i ||x - x_i|| in the normalized design space.
        # The minimum is taken against the existing archive only, never against
        # the query points themselves.
        return np.min(dists, axis=1, keepdims=True)

    # ============================================================
    # Exponential Distance Penalty
    # ============================================================
    def _distance_penalty(self, d_min: np.ndarray) -> np.ndarray:
        """
        Compute the distance penalty factor.

        Args:
            d_min (np.ndarray): Nearest distances. (N, 1).

        Returns:
            np.ndarray: Penalty factors in ``[0, 1)``. (N, 1).
        """
        # Distance penalty:
        # P_d(x) = 1 - exp(-alpha * d_min(x) / h).
        # Larger alpha makes the penalty saturate faster as a point moves away
        # from the sampled archive.
        return 1.0 - np.exp(-self.alpha * d_min / self.distance_scale)

    # ============================================================
    # Distance-Informed Utility
    # ============================================================
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the distance-informed acquisition function.

        Args:
            x (np.ndarray): Query points. (N, D).

        Returns:
            np.ndarray: Distance-informed utility values. (N, 1).
        """
        base_utility = super().evaluate(x)
        d_min = self._compute_nearest_distance(x)
        # Distance-informed acquisition:
        # U_d(x) = U_base(x) * P_d(x).
        # This preserves the parent acquisition ordering among equally distant points.
        utility = base_utility * self._distance_penalty(d_min)
        # Rejection threshold:
        # U_d(x) = 0,  if d_min(x) < d_0.
        # This keeps the optimizer from repeatedly sampling the same local basin.
        utility[d_min < self.min_distance] = 0.0
        return utility
