# Base Infill Criterion for Sequential Sampling
# Author: Shengning Wang

import warnings
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Optional, Union, List


class BaseInfill:
    """
    Abstract base class for sequential sampling infill criteria.

    Defines the unified API contract shared by all infill strategies:
      - ``evaluate(x)``: score acquisition function at continuous coordinates
      - ``propose()``:   propose one new sampling point

    Subclasses must override ``evaluate()``. The default ``propose()``
    implementation performs L-BFGS-B multi-restart optimization over the
    continuous design space by maximising ``evaluate()``. Subclasses that
    operate on a discrete candidate pool (e.g. ``MultiFidelityInfill``)
    should also override ``propose()``.

    Attributes:
        model: Pre-trained surrogate model (must expose a ``predict`` method).
        bounds (np.ndarray): Design variable bounds. shape: (num_features, 2).
        target_idx (int): Output dimension index used by the criterion.
        num_restarts (int): Number of random restarts for the default optimizer.
    """

    def __init__(
        self,
        model,
        bounds: Optional[Union[List[float], np.ndarray]] = None,
        target_idx: int = 0,
        num_restarts: int = 10,
    ) -> None:
        """
        Initialize the base infill configuration.

        Args:
            model: A fitted surrogate model with a ``predict`` method.
                Must expose a fitted indicator attribute (``beta`` or ``theta``).
            bounds (Optional[Union[List[float], np.ndarray]]): Design space bounds.
                shape: (num_features, 2). Required by the default ``propose()``.
            target_idx (int): Output dimension to optimise. Default 0.
            num_restarts (int): Random restarts for L-BFGS-B in ``propose()``.
                Default 10.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        self._validate_model(model)
        self.model = model

        if bounds is not None:
            self.bounds = np.array(bounds, dtype=np.float64)
            if self.bounds.ndim == 1:
                self.bounds = self.bounds.reshape(-1, 2)
        else:
            self.bounds = None

        self.target_idx = int(target_idx)
        self.num_restarts = int(num_restarts)

    # ------------------------------------------------------------------
    # Model Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model(model) -> None:
        """
        Assert the surrogate model has been fitted before use.

        Checks for the presence of either a ``beta`` or ``theta`` attribute
        (covers both KRG-beta and KRG-theta conventions).

        Args:
            model: Surrogate model instance to validate.

        Raises:
            RuntimeError: If neither ``beta`` nor ``theta`` is set on the model.
        """
        beta_ok = hasattr(model, "beta") and model.beta is not None
        theta_ok = hasattr(model, "theta") and model.theta is not None
        if not (beta_ok or theta_ok):
            raise RuntimeError("provided surrogate model is not fitted.")

    # ------------------------------------------------------------------
    # Abstract Interface
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the acquisition function at a batch of design coordinates.

        Must be overridden by every subclass.

        Args:
            x (np.ndarray): Query points in design space. shape: (N, num_features).

        Returns:
            np.ndarray: Acquisition scores. shape: (N, 1).
                Higher values indicate more promising candidates.

        Raises:
            NotImplementedError: Always — subclasses must implement this.
        """
        raise NotImplementedError("subclasses must implement evaluate().")

    # ------------------------------------------------------------------
    # Default Proposal (Continuous L-BFGS-B)
    # ------------------------------------------------------------------

    def _propose_continuous(self) -> np.ndarray:
        """
        Propose one new point by maximising ``evaluate()`` via L-BFGS-B.

        Uses random restarts to escape local optima on the acquisition surface.
        Requires ``self.bounds`` to be set.

        Returns:
            np.ndarray: Proposed design point. shape: (1, num_features).

        Raises:
            RuntimeError: If ``bounds`` was not provided at construction.
        """
        if self.bounds is None:
            raise RuntimeError(
                "bounds must be specified at construction to use _propose_continuous()."
            )

        num_features = self.bounds.shape[0]
        scipy_bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])

        def min_obj(x_vec: np.ndarray) -> float:
            utility = self.evaluate(x_vec[np.newaxis, :])
            return -float(np.asarray(utility).flatten()[0])

        best_x = None
        best_utility = -np.inf

        for _ in range(self.num_restarts):
            x0 = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=num_features
            )
            try:
                res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method="L-BFGS-B")
                curr_utility = -res.fun
                if curr_utility > best_utility:
                    best_utility = curr_utility
                    best_x = res.x
            except Exception:
                continue

        if best_x is None:
            warnings.warn(
                "optimization failed to converge; returning random point.",
                RuntimeWarning,
            )
            best_x = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=num_features
            )

        return best_x[np.newaxis, :]

    def propose(self) -> np.ndarray:
        """
        Propose one new sampling point.

        Default implementation maximises ``evaluate()`` over the continuous
        design space using L-BFGS-B with random restarts. Subclasses that
        operate on a discrete candidate pool should override this method.

        Returns:
            np.ndarray: Proposed design point. shape: (1, num_features).
        """
        return self._propose_continuous()
