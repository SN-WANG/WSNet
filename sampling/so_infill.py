# Single-Objective Infill Criteria for Sequential Sampling
# Author: Shengning Wang

import numpy as np
from scipy.stats import norm
from typing import Union, List

from sampling.base_infill import BaseInfill


class SingleObjectiveInfill(BaseInfill):
    """
    Single-objective infill criteria for sequential sampling.

    Wraps a suite of classical acquisition functions that guide exploration
    and exploitation using a pre-trained Kriging surrogate model.

    Supported Criteria:
        ``"mse"``: Mean Squared Error (pure exploration / maximum entropy).
        ``"poi"``: Probability of Improvement.
        ``"ei"``:  Expected Improvement (balances exploration and exploitation).
        ``"lcb"``: Lower Confidence Bound.

    All criteria are implemented to return a utility to be *maximised*. The
    default ``propose()`` inherited from ``BaseInfill`` maximises this utility
    over the continuous design space via L-BFGS-B with random restarts.

    Attributes:
        model: Pre-trained Kriging surrogate model.
        bounds (np.ndarray): Design space bounds. shape: (num_features, 2).
        target_idx (int): Output dimension to optimise.
        y_min (float): Current best (minimum) observed target value.
        criterion_name (str): Active criterion identifier.
        kappa (float): Exploration parameter for LCB. Default 2.0.
        num_restarts (int): Optimizer random restarts. Default 10.
    """

    def __init__(
        self,
        model,
        bounds: Union[List[float], np.ndarray],
        y_train: np.ndarray,
        criterion: str = "ei",
        target_idx: int = 0,
        num_restarts: int = 10,
        kappa: float = 2.0,
    ) -> None:
        """
        Initialize the single-objective infill strategy.

        Args:
            model: A fitted Kriging model instance.
            bounds (Union[List[float], np.ndarray]): Design space bounds.
                shape: (num_features, 2).
            y_train (np.ndarray): Training target values used to find ``y_min``.
                shape: (n_train, num_outputs).
            criterion (str): Acquisition function. Options: ``"mse"``, ``"poi"``,
                ``"ei"``, ``"lcb"``. Default ``"ei"``.
            target_idx (int): Output dimension to optimise. Default 0.
            num_restarts (int): Random restarts for L-BFGS-B. Default 10.
            kappa (float): Exploration coefficient for LCB. Default 2.0.

        Raises:
            RuntimeError: If the model is not fitted.
            ValueError: If ``criterion`` is not recognised.
        """
        super().__init__(model, bounds=bounds, target_idx=target_idx,
                         num_restarts=num_restarts)

        self.kappa = float(kappa)
        self.y_min = float(np.min(y_train[:, self.target_idx]))

        criterion_map = {
            "mse": self._crit_mse,
            "poi": self._crit_poi,
            "ei":  self._crit_ei,
            "lcb": self._crit_lcb,
        }

        key = criterion.lower()
        if key not in criterion_map:
            raise ValueError(
                f"unknown criterion: '{criterion}'. "
                f"available: {list(criterion_map.keys())}"
            )

        self.criterion_name = key
        self._criterion_func = criterion_map[key]

    # ------------------------------------------------------------------
    # Acquisition Functions (return utility to maximise)
    # ------------------------------------------------------------------

    def _crit_mse(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Mean Squared Error (MSE) criterion — pure exploration.

        Utility = sigma^2

        Args:
            mu (np.ndarray): Predicted mean. shape: (N, 1).
            sigma (np.ndarray): Predicted std dev. shape: (N, 1).

        Returns:
            np.ndarray: Utility values. shape: (N, 1).
        """
        return sigma ** 2

    def _crit_poi(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Probability of Improvement (POI) criterion.

        Utility = P(y < y_min) = Phi((y_min - mu) / sigma)

        Args:
            mu (np.ndarray): Predicted mean. shape: (N, 1).
            sigma (np.ndarray): Predicted std dev. shape: (N, 1).

        Returns:
            np.ndarray: Utility values. shape: (N, 1).
        """
        with np.errstate(divide="ignore"):
            z = (self.y_min - mu) / sigma
        poi = norm.cdf(z)
        poi[sigma < 1e-9] = 0.0
        return poi

    def _crit_ei(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Expected Improvement (EI) criterion.

        Utility = (y_min - mu) * Phi(z) + sigma * phi(z)
        where z = (y_min - mu) / sigma.

        Args:
            mu (np.ndarray): Predicted mean. shape: (N, 1).
            sigma (np.ndarray): Predicted std dev. shape: (N, 1).

        Returns:
            np.ndarray: Utility values. shape: (N, 1).
        """
        with np.errstate(divide="ignore"):
            improvement = self.y_min - mu
            z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def _crit_lcb(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Lower Confidence Bound (LCB) criterion, negated for maximisation.

        Utility = -(mu - kappa * sigma)

        Args:
            mu (np.ndarray): Predicted mean. shape: (N, 1).
            sigma (np.ndarray): Predicted std dev. shape: (N, 1).

        Returns:
            np.ndarray: Utility values (negated LCB). shape: (N, 1).
        """
        return -1.0 * (mu - self.kappa * sigma)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the active acquisition function at design coordinates.

        Args:
            x (np.ndarray): Query points. shape: (N, num_features).

        Returns:
            np.ndarray: Utility values. shape: (N, 1).
        """
        y_pred, mse_pred = self.model.predict(x)

        mu  = y_pred[:, self.target_idx].reshape(-1, 1)
        var = mse_pred[:, self.target_idx].reshape(-1, 1)
        sigma = np.sqrt(np.maximum(var, 1e-12))

        return self._criterion_func(mu, sigma)
