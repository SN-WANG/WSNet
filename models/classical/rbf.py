# Radial Basis Function (RBF) Surrogate Model
# Author: Shengning Wang

import numpy as np
from typing import Optional

from utils.scaler import StandardScalerNP


class RBF:
    """
    Radial Basis Function (RBF) using K-Means clustering and Gaussian kernels.
    """

    def __init__(self, num_centers: int = 20, gamma: Optional[float] = None, alpha: float = 0.0, max_iter: int = 500):
        """
        Args:
            num_centers (int): Number of RBF centers.
            gamma (Optional[float]): Kernel width parameter.
            alpha (float): Ridge regularization strength.
            max_iter (int): Maximum iterations for KMeans.
        """
        # parameters
        self.num_centers = num_centers
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

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

    def _init_centers(self, x: np.ndarray) -> np.ndarray:
        """
        PCA initialization using SVD.

        Args:
            x (np.ndarray): Inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Initial centers of shape: (num_centers, input_dim), dtype: float64.
        """
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        x_pca = u @ np.diag(s)

        idx = np.linspace(0, x.shape[0] - 1, self.num_centers, dtype=int)
        scores = x_pca[idx]

        return scores @ vt

    # ------------------------------------------------------------------

    def _kmeans(self, x: np.ndarray) -> np.ndarray:
        """
        Simple Lloyd KMeans.

        Args:
            x (np.ndarray): Inputs of shape: (num_samples, spatial_dim), dtype: float64.

        Returns:
            np.ndarray: Centers of shape: (num_centers, spatial_dim), dtype: float64.
        """
        centers = self._init_centers(x)

        for _ in range(self.max_iter):
            dists = self._compute_dists(x, centers)
            labels = np.argmin(dists, axis=1)

            new_centers = np.array(
                [
                    x[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else centers[k]
                    for k in range(self.num_centers)
                ]
            )

            if np.allclose(new_centers, centers):
                break

            centers = new_centers

        return centers

    # ------------------------------------------------------------------

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """
        Constructs Gaussian RBF feature matrix.

        Args:
            x (np.ndarray): Inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Feature matrix of shape: (num_samples, num_centers), dtype: float64.
        """
        dists = self._compute_dists(x, self.centers)
        return np.exp(-self.gamma * dists)

    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform model training.

        Args:
            x_train (np.ndarray): Training inputs of shape: (num_samples, input_dim), dtype: float64.
            y_train (np.ndarray): Training targets of shape: (num_samples, target_dim), dtype: float64.
        """
        x_scaled = self.scaler_x.fit(x_train, channel_dim=1).transform(x_train)
        y_scaled = self.scaler_y.fit(y_train, channel_dim=1).transform(y_train)

        num_samples = x_scaled.shape[0]
        if self.num_centers >= num_samples:
            self.centers = x_scaled
            self.num_centers = num_samples
        else:
            self.centers = self._kmeans(x_scaled)

        if self.gamma is None:
            x_var = x_scaled.var()
            self.gamma = 1.0 / (x_scaled.shape[1] * x_var) if x_var > 0 else 1.0

        phi = self._build_features(x_scaled)

        xtx = phi.T @ phi
        xty = phi.T @ y_scaled

        if self.alpha > 0.0:
            np.fill_diagonal(xtx, xtx.diagonal() + self.alpha)

        self.weights = np.linalg.solve(xtx, xty)

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction.

        Args:
            x_pred (np.ndarray): Prediction inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets of shape: (num_samples, target_dim), dtype: float64.
        """
        if self.centers is None or self.weights is None:
            raise RuntimeError("Model has not been fitted.")

        x_scaled = self.scaler_x.transform(x_pred)
        phi = self._build_features(x_scaled)

        y_scaled = phi @ self.weights
        y_pred = self.scaler_y.inverse_transform(y_scaled)

        return y_pred
