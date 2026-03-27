# AES-MSI: Adaptive Ensemble of Surrogate Models by Minimum Screening Index
# Paper reference: https://doi.org/10.1115/1.4054243
# Paper author: Shuai Zhang, Yong Pang, Pengwei Liang, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from typing import Dict, List, Optional

from models.classical.prs import PRS
from models.classical.rbf import RBF
from models.classical.krg import KRG
from models.classical.svr import SVR


class AESMSI:
    """
    Adaptive Ensemble of Surrogate Models by Minimum Screening Index (AES-MSI).

    A robust ensemble method that:
    1. Constructs a screening index based on correlation coefficient and CV error
       to identify and eliminate globally poor models from the model library.
    2. Determines the baseline model as the one with minimum screening index.
    3. Proposes a novel weight calculation strategy based on local errors relative
       to the baseline model for adaptive ensemble.

    Attributes:
        threshold (float): The normalized screening index threshold [0, 1] for 
            model selection. Models with NSI > threshold are discarded.
        models_pool (List[object]): The list of instantiated surrogate model objects.
        activate_indices_ (List[int]): Indices of models that passed the filtering stage.
        baseline_index_ (int): Index of the best performing model (baseline).
        is_fitted (bool): Status flag indicating if the model has been trained.
    """

    def __init__(self, threshold: float = 0.5,
        prs_params: Optional[Dict] = None,
        rbf_params: Optional[Dict] = None,
        krg_params: Optional[Dict] = None,
        svr_params: Optional[Dict] = None,
    ) -> None:
        """
        Initializes the AES-MSI model configuration.

        Args:
            threshold (float): Threshold for filtering models based on normalized 
                screening index. Suggested: 0.5 or 0.8. Default is 0.5.
            prs_params (Optional[Dict]): Dictionary of kwargs to pass to PRS class.
                If None, default PRS parameters are used.
            rbf_params (Optional[Dict]): Dictionary of kwargs to pass to RBF class.
                If None, default RBF parameters are used.
            krg_params (Optional[Dict]): Dictionary of kwargs to pass to KRG class.
                If None, default KRG parameters are used.
            svr_params (Optional[Dict]): Dictionary of kwargs to pass to SVR class.
                If None, default SVR parameters are used.
        """
        # parameters
        self.threshold = threshold

        prs_params = prs_params if prs_params is not None else {}
        rbf_params = rbf_params if rbf_params is not None else {}
        krg_params = krg_params if krg_params is not None else {}
        svr_params = svr_params if svr_params is not None else {}

        self.models_pool = [
            PRS(**prs_params),
            RBF(**rbf_params),
            KRG(**krg_params),
            SVR(**svr_params),
        ]

        # model state
        self.activate_indices_: List[int] = []
        self.baseline_index_: int = -1
        self.is_fitted: bool = False

    # ------------------------------------------------------------------

    def _calculate_loo_error(self, model: object, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the Leave-One-Out (LOO) Mean Squared Error for a given model.

        Args:
            model (object): The surrogate model instance to evaluate.
            x (np.ndarray): Training inputs of shape: (num_samples, input_dim), dtype: float64.
            y (np.ndarray): Training targets of shape: (num_samples, target_dim), dtype: float64.

        Returns:
            float: The aggregated Mean Squared Error across all samples and outputs.
        """
        num_samples = x.shape[0]
        y_true_all = []
        y_pred_all = []

        for i in range(num_samples):
            # Leave one out
            train_idx = np.delete(np.arange(num_samples), i)
            test_idx = i

            x_train_fold, x_test_fold = x[train_idx], x[test_idx : test_idx + 1]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx : test_idx + 1]

            model.fit(x_train_fold, y_train_fold)
            y_pred_fold = model.predict(x_test_fold)
            if isinstance(y_pred_fold, tuple):
                y_pred_fold = y_pred_fold[0]  # Prevent KRG's var_pred

            y_true_all.append(y_test_fold)
            y_pred_all.append(y_pred_fold)

        y_true_stack = np.vstack(y_true_all)
        y_pred_stack = np.vstack(y_pred_all)

        mse = np.mean((y_true_stack - y_pred_stack) ** 2)
        return float(mse)

    # ------------------------------------------------------------------

    def _calculate_correlation_matrix(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pairwise correlation coefficients between model predictions.

        Args:
            predictions (List[np.ndarray]): List of prediction arrays from each model.
                Each array has shape: (num_samples, target_dim), dtype: float64.

        Returns:
            np.ndarray: Correlation matrix of shape (num_models, num_models), dtype: float64.
        """
        num_models = len(predictions)
        corr_matrix = np.eye(num_models, dtype=np.float64)

        for i in range(num_models):
            for j in range(i + 1, num_models):
                # Flatten predictions for correlation calculation
                pred_i = predictions[i].flatten()
                pred_j = predictions[j].flatten()

                # Calculate Pearson correlation coefficient
                mean_i = np.mean(pred_i)
                mean_j = np.mean(pred_j)

                numerator = np.sum((pred_i - mean_i) * (pred_j - mean_j))
                denominator = np.sqrt(
                    np.sum((pred_i - mean_i) ** 2) * np.sum((pred_j - mean_j) ** 2)
                )

                if denominator > 1e-12:
                    corr = numerator / denominator
                else:
                    corr = 0.0

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform model training.

        Stage 1: Fits all candidate models, computes LOO-CV error and correlation
            coefficients, constructs screening index, and filters based on threshold.
        Stage 2: Retrains active models on full data and determines baseline.

        Args:
            x_train (np.ndarray): Training inputs of shape: (num_samples, input_dim), dtype: float64.
            y_train (np.ndarray): Training targets of shape: (num_samples, target_dim), dtype: float64.
        """
        num_models = len(self.models_pool)

        # --- Stage 1: Model Selection via Screening Index ---
        # Step 1: Calculate LOO-CV errors for each model
        cv_errors = np.zeros(num_models, dtype=np.float64)
        for i, model in enumerate(self.models_pool):
            cv_errors[i] = self._calculate_loo_error(model, x_train, y_train)

        # Step 2: Fit all models on full training set to get predictions
        full_predictions = []
        for i, model in enumerate(self.models_pool):
            model.fit(x_train, y_train)
            pred = model.predict(x_train)
            if isinstance(pred, tuple):
                pred = pred[0]  # Prevent KRG's var_pred
            full_predictions.append(pred)

        # Step 3: Calculate correlation matrix
        corr_matrix = self._calculate_correlation_matrix(full_predictions)

        # Step 4: Construct Screening Index (SI) = MCV / max_correlation
        # MCV: Mean of CV errors
        mcv = cv_errors  # Already mean squared error

        # max_correlation: maximum absolute correlation with other models
        max_correlations = np.zeros(num_models, dtype=np.float64)
        for i in range(num_models):
            # Get correlations with all other models (excluding self)
            other_corrs = np.delete(np.abs(corr_matrix[i, :]), i)
            max_correlations[i] = np.max(other_corrs) if len(other_corrs) > 0 else 0.0

        # Avoid division by zero
        max_correlations[max_correlations < 1e-12] = 1e-12

        # Screening Index: SI = MCV / max_correlation
        # Smaller SI indicates better performance (lower error, higher correlation)
        si = mcv / max_correlations

        # Step 5: Normalize Screening Index (NSI)
        si_min = np.min(si)
        si_max = np.max(si)

        if np.isclose(si_max, si_min):
            nsi = np.zeros_like(si)
        else:
            nsi = (si - si_min) / (si_max - si_min)

        # Step 6: Filter models based on threshold
        # Models with NSI <= threshold are selected
        self.activate_indices_ = np.where(nsi <= self.threshold)[0].tolist()

        # Identify Baseline (Best) Model: minimum NSI
        self.baseline_index_ = int(np.argmin(nsi))

        # Ensure the baseline is strictly in the active set
        if self.baseline_index_ not in self.activate_indices_:
            self.activate_indices_.append(self.baseline_index_)

        # Sort activate_indices for consistent ordering
        self.activate_indices_.sort()

        # --- Stage 2: Final Fitting ---
        # Refit all ACTIVE models on the FULL training set
        for i in self.activate_indices_:
            self.models_pool[i].fit(x_train, y_train)

        self.is_fitted = True

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction.

        Adaptive Weighting Logic:
            y_ensemble(x) = w_base * y_base(x) + sum(w_i * y_i(x))
        where:
            - Baseline model weight: w_base = 0.5
            - Other model weights: w_i = (1 / l_i) / (2 * sum(1 / l_j))
            - l_i = |y_i(x) - y_base(x)| / sum(|y_j(x) - y_base(x)|)

        Args:
            x_pred (np.ndarray): Prediction inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets of shape: (num_samples, target_dim), dtype: float64.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        num_samples = x_pred.shape[0]

        # Get predictions from all active models
        active_predictions = {}
        for i in self.activate_indices_:
            model = self.models_pool[i]
            pred = model.predict(x_pred)
            if isinstance(pred, tuple):
                pred = pred[0]  # Prevent KRG's var_pred
            active_predictions[i] = pred

        # Get baseline prediction
        y_base = active_predictions[self.baseline_index_]

        # If only baseline model is active, return its prediction
        if len(self.activate_indices_) == 1:
            y_pred = y_base
        else:
            # Calculate local errors and weights for each sample
            y_pred = np.zeros_like(y_base, dtype=np.float64)

            for n in range(num_samples):
                # Get predictions at this sample point for all active models
                y_base_n = y_base[n : n + 1, :]  # Shape: (1, target_dim)

                # Collect predictions from non-baseline models
                other_preds = []
                other_indices = []
                for idx in self.activate_indices_:
                    if idx != self.baseline_index_:
                        other_preds.append(active_predictions[idx][n : n + 1, :])
                        other_indices.append(idx)

                if len(other_preds) == 0:
                    y_pred[n : n + 1, :] = y_base_n
                    continue

                other_preds = np.vstack(other_preds)  # Shape: (num_other, target_dim)

                # Calculate local errors: e_i = |y_i - y_base|
                local_errors = np.abs(other_preds - y_base_n)  # Shape: (num_other, target_dim)

                # Normalize errors: l_i = e_i / sum(e_j)
                error_sums = np.sum(local_errors, axis=0, keepdims=True)  # Shape: (1, target_dim)
                error_sums[error_sums < 1e-12] = 1e-12  # Avoid division by zero

                normalized_errors = local_errors / error_sums  # Shape: (num_other, target_dim)
                normalized_errors[normalized_errors < 1e-12] = 1e-12  # Avoid division by zero

                # Calculate weights: w_i = (1 / l_i) / (2 * sum(1 / l_j))
                inv_errors = 1.0 / normalized_errors  # Shape: (num_other, target_dim)
                inv_error_sums = np.sum(inv_errors, axis=0, keepdims=True)  # Shape: (1, target_dim)

                other_weights = inv_errors / (2.0 * inv_error_sums)  # Shape: (num_other, target_dim)
                base_weight = 0.5  # Baseline model weight

                # Weighted sum: y = w_base * y_base + sum(w_i * y_i)
                weighted_sum = base_weight * y_base_n
                for i, idx in enumerate(other_indices):
                    weighted_sum += other_weights[i : i + 1, :] * other_preds[i : i + 1, :]

                y_pred[n : n + 1, :] = weighted_sum

        return y_pred
