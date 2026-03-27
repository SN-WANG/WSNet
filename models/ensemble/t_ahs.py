# T-AHS: Two-Stage Adaptive Hybrid Surrogate Model
# Paper reference: https://doi.org/10.1115/1.4039128
# Paper author: Xueguan Song, Liye Lv, Jieling Li, Wei Sun, Jie Zhang
# Code author: Shengning Wang

import numpy as np
from typing import Dict, List, Optional

from models.classical.prs import PRS
from models.classical.rbf import RBF
from models.classical.krg import KRG
from models.classical.svr import SVR


class TAHS:
    """
    Two-Stage Adaptive Hybrid Surrogate (T-AHS) Model.

    A robust ensemble method that:
    1. Filters out poor-performing surrogate models using Leave-One-Out Cross-Validation (Stage 1).
    2. Constructs a weighted average prediction where weights are adaptively determined
       by the local uncertainty estimated via a Gaussian Process (Stage 2).

    Attributes:
        threshold (float): The normalized CV error threshold [0, 1] for model selection.
            Models with normalized error > threshold are discarded.
        models_pool (List[object]): The list of instantiated surrogate model objects.
        activate_indices_ (List[int]): Indices of models that passed the filtering stage.
        baseline_index_ (int): Index of the best performing model (baseline).
        uncertainty_model_ (KRG): A dedicated Kriging model used solely for estimating
            local process variance (s^2) for weighting.
        is_fitted (bool): Status flag indicating if the model has been trained.
    """

    def __init__(self, threshold: float = 0.5,
                 prs_params: Optional[Dict] = None,
                 rbf_params: Optional[Dict] = None,
                 krg_params: Optional[Dict] = None,
                 svr_params: Optional[Dict] = None):
        """
        Initializes the T-AHS model configuration.

        Args:
            threshold (float): Threshold for filtering models based on normalized CV error.
                Suggested: 0.2 or 0.5. Default is 0.5.
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

        # initialize the pool of component models
        self.models_pool = [
            PRS(**prs_params),
            RBF(**rbf_params),
            KRG(**krg_params),
            SVR(**svr_params)
        ]

        # dedicated model for variance estimation
        self.uncertainty_model_ = KRG(**krg_params)

        # model state
        self.activate_indices_: List[int] = []
        self.baseline_index_: int = -1
        self.is_fitted: bool = False

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

            x_train_fold, x_test_fold = x[train_idx], x[test_idx:test_idx + 1]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx:test_idx + 1]

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

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform model training.

        Stage 1: Fits all candidate models, computes LOO-CV error, and filters based on threshold.
        Stage 2: Retrains active models on full data and trains the uncertainty model (KRG).

        Args:
            x_train (np.ndarray): Training inputs of shape: (num_samples, input_dim), dtype: float64.
            y_train (np.ndarray): Training targets of shape: (num_samples, target_dim), dtype: float64.
        """
        # --- Stage 1: Model Selection via LOO-CV ---
        cv_errors = []

        for model in self.models_pool:
            error = self._calculate_loo_error(model, x_train, y_train)
            cv_errors.append(error)

        cv_errors = np.array(cv_errors)

        # Normalize CV errors: E_norm = (E - E_min) / (E_max - E_min)
        e_min = np.min(cv_errors)
        e_max = np.max(cv_errors)

        if np.isclose(e_max, e_min):
            cv_norm = np.zeros_like(cv_errors)
        else:
            cv_norm = (cv_errors - e_min) / (e_max - e_min)

        # Filter models
        self.activate_indices_ = np.where(cv_norm <= self.threshold)[0].tolist()

        # Identify Baseline (Best) Model
        self.baseline_index_ = int(np.argmin(cv_errors))

        # Ensure the baseline is strictly in the active set
        if self.baseline_index_ not in self.activate_indices_:
            self.activate_indices_.append(self.baseline_index_)

        # --- Stage 2: Final Fitting ---
        # 1. Fit all ACTIVE models on the FULL training set
        for i in self.activate_indices_:
            self.models_pool[i].fit(x_train, y_train)

        # 2. Fit the Uncertainty Model (KRG)
        self.uncertainty_model_.fit(x_train, y_train)

        self.is_fitted = True

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Perform model prediction.

        Adaptive Weighting Logic:
            y_hybrid(x) = sum(w_i(x) * y_i(x))
        where:
            w_i(x) ~ P_i(x) / sum(P_j(x))
            P_i(x) = exp( - (y_baseline(x) - y_i(x))^2 / (2 * s^2(x)) )
        where s^2(x) is the predicted variance from the Kriging model.

        Args:
            x_pred (np.ndarray): Prediction inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            np.ndarray: Prediction targets of shape: (num_samples, target_dim), dtype: float64.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        num_samples = x_pred.shape[0]
        target_dim = self.uncertainty_model_.beta.shape[1]

        # Step 1: Get Uncertainty Estimate (s^2)
        _, var_pred = self.uncertainty_model_.predict(x_pred)
        s2 = np.maximum(var_pred, 1e-12)

        # Step 2: Get Baseline Prediction
        def safe_predict(model, x):
            res = model.predict(x)
            return res[0] if isinstance(res, tuple) else res

        y_base = safe_predict(self.models_pool[self.baseline_index_], x_pred)

        # Step 3: Calculate Weighted Predictions
        numerator_sum = np.zeros((num_samples, target_dim))
        denominator_sum = np.zeros((num_samples, target_dim))

        for i in self.activate_indices_:
            model = self.models_pool[i]
            y_curr = safe_predict(model, x_pred)

            # Calculate Probability Coefficient P_i
            # P_i(x) = exp( - (y_baseline(x) - y_i(x))^2 / (2 * s^2(x)) )
            # If model is baseline, P_i = 1.0 (Maximum weight)
            squared_diff = (y_base - y_curr) ** 2
            exponent = -squared_diff / (2.0 * s2)

            exponent = np.maximum(exponent, -100.0)
            p_i = np.exp(exponent)

            # Accumulate
            # y_hybrid(x) = sum(w_i(x) * y_i(x)) = sum( (P_i / sum_P) * y_i ) = sum(P_i * y_i) / sum(P_i)
            numerator_sum += p_i * y_curr
            denominator_sum += p_i

        # Step 4: Final aggregation
        denominator_sum = np.maximum(denominator_sum, 1e-12)
        y_pred = numerator_sum / denominator_sum

        return y_pred
