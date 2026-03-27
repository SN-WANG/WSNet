# Kriging (KRG, Gaussian Process Regression) Surrogate Model
# Author: Shengning Wang

import numpy as np
from scipy.linalg import cholesky, solve_triangular, qr
from scipy.optimize import minimize, Bounds
from typing import Dict, Any, Tuple, Union, Optional, Callable

from utils.scaler import StandardScalerNP


class KRG:
    """
    Kriging (KRG) using Generalized Least Squares (GLS) and Maximum Likelihood Estimation (MLE).
    """

    def __init__(self, poly: Union[str, Callable] = "constant", kernel: Union[str, Callable] = "gaussian",
                 theta0: Union[float, np.ndarray] = 1.0, theta_bounds: Tuple[float, float] = (1e-6, 100.0)):
        """
        Initializes the Kriging model configuration.

        Args:
            poly (Union[str, Callable]): Regression model type.
                Options: "constant", "linear", "quadratic" or a custom function.
            kernel (Union[str, Callable]): Correlation model type.
                Options: "exponential", "exponential_general", "gaussian",
                         "linear", "spherical", "cubic", "spline" or a custom function.
            theta0 (Union[float, np.ndarray]): Initial guess for theta.
            theta_bounds (Tuple[float, float]): Lower and upper bounds for theta optimization.
        """
        # dispatch logic for regression function
        if isinstance(poly, str):
            poly_map = {
                "constant": self._reg_constant,
                "linear": self._reg_linear,
                "quadratic": self._reg_quadratic
            }
            if poly not in poly_map:
                raise ValueError(f"Unknown poly type: '{poly}'. Available: {list(poly_map.keys())}")
            self.reg_func = poly_map[poly]
        else:
            self.reg_func = poly

        # dispatch logic for kernel function
        if isinstance(kernel, str):
            kernel_map = {
                "exponential": self._kernel_exponential,
                "exponential_general": self._kernel_exponential_general,
                "gaussian": self._kernel_gaussian,
                "linear": self._kernel_linear,
                "spherical": self._kernel_spherical,
                "cubic": self._kernel_cubic,
                "spline": self._kernel_spline
            }
            if kernel not in kernel_map:
                raise ValueError(f"Unknown kernel type: '{kernel}'. Available: {list(kernel_map.keys())}")
            self.corr_func = kernel_map[kernel]
        else:
            self.corr_func = kernel

        # parameters
        self.poly_name = poly if isinstance(poly, str) else "custom"
        self.kernel_name = kernel if isinstance(kernel, str) else "custom"

        self.theta0 = theta0
        self.theta_bounds = theta_bounds

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.x_train_scaled: Optional[np.ndarray] = None
        self.theta: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.gamma: Optional[np.ndarray] = None
        self.sigma2: Optional[np.ndarray] = None

        # decomposition matrices
        self.C: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None
        self.Ft: Optional[np.ndarray] = None

    # ======================================================================
    # Static Methods: Regression Models (Trend Functions)
    # ======================================================================

    @staticmethod
    def _reg_constant(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-order polynomial regression function (Constant trend)
        F = [1]

        Args:
            x (np.ndarray): Input feature data.
                Shape: (num_samples, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                F (np.ndarray): Regression matrix.
                    Shape: (num_samples, 1), dtype: float64.
                dF (np.ndarray): Jacobian of F at the first point.
                    Shape: (input_dim, 1), dtype: float64.
        """
        num_samples, input_dim = x.shape
        F = np.ones((num_samples, 1))
        dF = np.zeros((input_dim, 1))
        return F, dF

    @staticmethod
    def _reg_linear(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        First-order polynomial regression function (Linear trend)
        F = [1, x_1, ..., x_n]

        Args:
            x (np.ndarray): Input feature data.
                Shape: (num_samples, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                F (np.ndarray): Regression matrix.
                    Shape: (num_samples, input_dim + 1), dtype: float64.
                dF (np.ndarray): Jacobian of F at the first point.
                    Shape: (input_dim, input_dim + 1), dtype: float64.
        """

        num_samples, input_dim = x.shape
        F = np.hstack([np.ones((num_samples, 1)), x])
        dF = np.hstack([np.zeros((input_dim, 1)), np.eye(input_dim)])
        return F, dF

    @staticmethod
    def _reg_quadratic(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Second-order polynomial regression function (Quadratic trend)
        F = [1, x_i, x_i^2, x_i*x_j]

        Args:
            x (np.ndarray): Input feature data.
                Shape: (num_samples, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                F (np.ndarray): Regression matrix.
                    Shape: (num_samples, num_terms), dtype: float64.
                dF (np.ndarray): Jacobian of F at the first point.
                    Shape: (input_dim, num_terms), dtype: float64.
        """

        num_samples, input_dim = x.shape

        # Calculate quadratic terms F = [1, S, S^2, S(:, 1)*S(:, 2), ..., S(:, n)^2]

        # Constant and Linear terms
        F_parts = [np.ones((num_samples, 1)), x]

        # Quadratic terms
        for k in range(input_dim):
            # Multiply k-th column with all columns from k onwards
            # Shape: (num_samples, 1) * (num_samples, num_remaining)
            quad_part = x[:, k:k+1] * x[:, k:]
            F_parts.append(quad_part)

        F = np.hstack(F_parts)

        # Jacobian (dF) at the first point (x[0])
        x0 = x[0, :]
        num_terms = F.shape[1]
        dF = np.zeros((input_dim, num_terms))

        # Constant term derivative is 0

        # Linear terms derivative
        dF[:, 1 : 1 + input_dim] = np.eye(input_dim)

        # Quadratic terms derivative
        curr_col = 1 + input_dim
        for k in range(input_dim):
            # For terms x_k * x_j where j goes from k to input_dim - 1
            for j in range(k, input_dim):
                # Term is x_k * x_j
                # Derivative w.r.t x_p:
                # if p==k and p==j (i.e. x_k^2): 2 * x_k
                # if p==k (and k!=j): x_j
                # if p==j (and k!=j): x_k

                # d/dx_k += x_j
                dF[k, curr_col] += x0[j]
                # d/dx_j += x_k
                dF[j, curr_col] += x0[k]

                curr_col += 1

        return F, dF

    # ======================================================================
    # Static Methods: Correlation Models (Kernel Functions)
    # ======================================================================

    @staticmethod
    def _kernel_exponential(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Exponential correlation function
        r_i = prod_{j=1}^{n} exp(-theta_j * |d_{ij}|)

        Args:
            theta (np.ndarray): Correlation parameters (length-scales).
                Shape: (input_dim,) or (1,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        if theta.ndim == 1 and theta.size == 1:
            # Isotropic case: all theta_j = theta
            theta_mat = np.tile(theta, (num_diffs, input_dim))
        else:
            # Anisotropic case
            theta = theta.flatten()
            if theta.size != input_dim:
                raise ValueError(f"Theta must be of length 1 or {input_dim}")
            theta_mat = np.tile(theta, (num_diffs, 1))

        # Correlation terms: r_i = prod_{j=1}^{n} exp(-theta_j * |d_{ij}|)
        td = -theta_mat * np.abs(d)
        r = np.exp(np.sum(td, axis=1, keepdims=True))

        dr = None
        if num_diffs > 0:
            dr = (-theta_mat) * np.sign(d) * r

        return r, dr

    @staticmethod
    def _kernel_exponential_general(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        General exponential correlation function
        r_i = prod_{j=1}^{n} exp(-theta_j * |d_{ij}|^theta_{n+1})

        Note: Theta must have length n + 1 (anisotropic) or 2 (isotropic)
            The last element of theta is the exponent (power)

        Args:
            theta (np.ndarray): Correlation parameters. Length n+1 or 2.
                Shape: (input_dim + 1,) or (2,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        theta = theta.flatten()

        if input_dim > 1 and theta.size == 2:
            # Isotropic case: theta[0] for all dims, theta[1] is power
            theta_params = np.tile(theta[0], (num_diffs, input_dim))
            power = theta[1]
        elif theta.size == input_dim + 1:
            # Anisotropic: theta[0:n] for dims, theta[n] is power
            theta_params = np.tile(theta[:input_dim], (num_diffs, 1))
            power = theta[input_dim]
        else:
            raise ValueError(f"Length of theta must be 2 (isotropic) or {input_dim + 1} (anisotropic)")

        # Correlation terms: r_i = prod_{j=1}^{n} exp(-theta_j * |d_{ij}|^theta_{n+1})
        td = -theta_params * (np.abs(d) ** power)
        r = np.exp(np.sum(td, axis=1, keepdims=True))

        dr = None
        if num_diffs > 0:
            term_deriv = power * (-theta_params) * np.sign(d) * (np.abs(d) ** (power - 1))
            dr = term_deriv * r

        return r, dr

    @staticmethod
    def _kernel_gaussian(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Gaussian correlation function
        r_i = prod_{j=1}^{n} exp(-theta_j * d_{ij}^2)

        Args:
            theta (np.ndarray): Correlation parameters (length-scales).
                Shape: (input_dim,) or (1,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        if theta.ndim == 1 and theta.size == 1:
            # Isotropic case: all theta_j = theta
            theta_mat = np.tile(theta, (num_diffs, input_dim))
        else:
            # Anisotropic case
            theta = theta.flatten()
            if theta.size != input_dim:
                raise ValueError(f"Theta must be of length 1 or {input_dim}")
            theta_mat = np.tile(theta, (num_diffs, 1))

        # Correlation terms: r_i = prod_{j=1}^{n} exp(-theta_j * d_{ij}^2)
        td = -theta_mat * d**2
        r = np.exp(np.sum(td, axis=1, keepdims=True))

        dr = None
        if num_diffs > 0:
            dr = (-2 * theta_mat) * d * r

        return r, dr

    @staticmethod
    def _kernel_linear(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Linear correlation function (Local support)
        r_i = prod_{j=1}^{n} max(0, 1 - theta_j * |d_{ij}|)

        Args:
            theta (np.ndarray): Correlation parameters.
                Shape: (input_dim,) or (1,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        if theta.ndim == 1 and theta.size == 1:
            # Isotropic case: all theta_j = theta
            theta_mat = np.tile(theta, (num_diffs, input_dim))
        else:
            # Anisotropic case
            theta = theta.flatten()
            if theta.size != input_dim:
                raise ValueError(f"Theta must be of length 1 or {input_dim}")
            theta_mat = np.tile(theta, (num_diffs, 1))

        # Correlation terms: r_i = prod_{j=1}^{n} max(0, 1 - theta_j * |d_{ij}|)
        td = np.maximum(1 - theta_mat * np.abs(d), 0)
        r = np.prod(td, axis=1, keepdims=True)

        dr = None
        if num_diffs > 0:
            dr = np.zeros_like(d)
            for j in range(input_dim):
                # Derivative is -theta * sign(d) where term > 0, else 0
                dd = -theta_mat[:, j] * np.sign(d[:, j])
                # Zero out derivative where max(0, ...) clipped the function
                dd[td[:, j] == 0] = 0

                # Product rule
                mask = np.ones(input_dim, dtype=bool)
                mask[j] = False
                r_others = np.prod(td[:, mask], axis=1)

                dr[:, j] = r_others * dd

        return r, dr

    @staticmethod
    def _kernel_spherical(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Spherical correlation function (Local support)
        r_i = prod_{j=1}^{n} max(0, 1 - 1.5 * (theta_j * d_{ij}) + 0.5 * (theta_j * d_{ij})^3)

        Args:
            theta (np.ndarray): Correlation parameters.
                Shape: (input_dim,) or (1,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        if theta.ndim == 1 and theta.size == 1:
            # Isotropic case: all theta_j = theta
            theta_mat = np.tile(theta, (num_diffs, input_dim))
        else:
            # Anisotropic case
            theta = theta.flatten()
            if theta.size != input_dim:
                raise ValueError(f"Theta must be of length 1 or {input_dim}")
            theta_mat = np.tile(theta, (num_diffs, 1))

        # Correlation terms: r_i = prod_{j=1}^{n} max(0, 1 - 1.5 * (theta_j * d_{ij}) + 0.5 * (theta_j * d_{ij})^3)
        td = np.minimum(np.abs(d) * theta_mat, 1)
        r_term = 1 - td * (1.5 - 0.5 * td**2)
        r = np.prod(r_term, axis=1, keepdims=True)

        dr = None
        if num_diffs > 0:
            dr = np.zeros_like(d)
            for j in range(input_dim):
                # Derivative logic
                dd = 1.5 * theta_mat[:, j] * np.sign(d[:, j]) * (td[:, j]**2 - 1)

                # Product rule
                mask = np.ones(input_dim, dtype=bool)
                mask[j] = False
                r_others = np.prod(r_term[:, mask], axis=1)

                dr[:, j] = r_others * dd

        return r, dr

    @staticmethod
    def _kernel_cubic(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Cubic correlation function (Local support)
        r_i = prod_{j=1}^{n} max(0, 1 - 3 * (theta_j * d_{ij})^2 + 2 * (theta_j * d_{ij})^3)

        Args:
            theta (np.ndarray): Correlation parameters.
                Shape: (input_dim,) or (1,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        if theta.ndim == 1 and theta.size == 1:
            # Isotropic case: all theta_j = theta
            theta_mat = np.tile(theta, (num_diffs, input_dim))
        else:
            # Anisotropic case
            theta = theta.flatten()
            if theta.size != input_dim:
                raise ValueError(f"Theta must be of length 1 or {input_dim}")
            theta_mat = np.tile(theta, (num_diffs, 1))

        # Correlation terms: r_i = prod_{j=1}^{n} max(0, 1 - 3 * (theta_j * d_{ij})^2 + 2 * (theta_j * d_{ij})^3)
        td = np.minimum(np.abs(d) * theta_mat, 1)
        r_term = 1 - td**2 * (3 - 2 * td)
        r = np.prod(r_term, axis=1, keepdims=True)

        dr = None
        if num_diffs > 0:
            dr = np.zeros_like(d)
            for j in range(input_dim):
                # Derivative logic
                dd = 6 * theta_mat[:, j] * np.sign(d[:, j]) * td[:, j] * (td[:, j] - 1)

                # Product rule
                mask = np.ones(input_dim, dtype=bool)
                mask[j] = False
                r_others = np.prod(r_term[:, mask], axis=1)

                dr[:, j] = r_others * dd

        return r, dr

    @staticmethod
    def _kernel_spline(theta: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Cubic Spline correlation function (Local support)
        Piecewise function:
            r_i = 1 - 15 * x^2 + 30 * x^3  for 0 <= x <= 0.2
            r_i = 1.25 * (1 - x)^3         for 0.2 < x < 1
            r_i = 0                        for x >= 1
        where x = theta_j * |d_{ij}|

        Args:
            theta (np.ndarray): Correlation parameters.
                Shape: (input_dim,) or (1,), dtype: float64.
            d (np.ndarray): Differences between input sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                r (np.ndarray): Correlation vector.
                    Shape: (num_diffs, 1), dtype: float64.
                dr (np.ndarray): Jacobian matrix.
                    Shape: (num_diffs, input_dim), dtype: float64.
        """

        d = np.atleast_2d(d)
        num_diffs, input_dim = d.shape

        if theta.ndim == 1 and theta.size == 1:
            # Isotropic case: all theta_j = theta
            theta_mat = np.tile(theta, (num_diffs, input_dim))
        else:
            # Anisotropic case
            theta = theta.flatten()
            if theta.size != input_dim:
                raise ValueError(f"Theta must be of length 1 or {input_dim}")
            theta_mat = np.tile(theta, (num_diffs, 1))

        # Correlation terms
        xi = np.abs(d) * theta_mat
        r_term = np.zeros_like(xi)

        # Region 1: 0 <= xi <= 0.2
        mask1 = xi <= 0.2
        r_term[mask1] = 1 - 15 * xi[mask1]**2 + 30 * xi[mask1]**3

        # Region 2: 0.2 < xi < 1
        mask2 = (xi > 0.2) & (xi < 1)
        r_term[mask2] = 1.25 * (1 - xi[mask2])**3

        r = np.prod(r_term, axis=1, keepdims=True)

        dr = None
        if num_diffs > 0:
            dr = np.zeros_like(d)

            # Compute derivatives for individual components (ds / d_{ij})
            ds = np.zeros_like(d)
            u = np.sign(d) * theta_mat

            # Derivative for Region 1
            ds[mask1] = u[mask1] * (xi[mask1] * (90 * xi[mask1] - 30))

            # Derivative for Region 2
            ds[mask2] = -3.75 * u[mask2] * (1 - xi[mask2])**2

            # Product rule loop
            for j in range(input_dim):
                mask = np.ones(input_dim, dtype=bool)
                mask[j] = False
                r_others = np.prod(r_term[:, mask], axis=1)
                dr[:, j] = r_others * ds[:, j]

        return r, dr

    # ======================================================================
    # Core Logic
    # ======================================================================

    def _fit_gls(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, D: np.ndarray) -> Dict[str, Any]:
        """
        Performs Generalized Least Squares (GLS) estimation.

        Args:
            x (np.ndarray): Scaled inputs of shape: (num_samples, input_dim), dtype: float64.
            y (np.ndarray): Scaled targets of shape: (num_samples, target_dim), dtype: float64.
            theta (np.ndarray): Correlation parameters of shape: (input_dim,), dtype: float64.
            D (np.ndarray): Distance matrix of shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            Dict[str, Any]: A dictionary containing decomposition matrices and estimates.
        """

        num_samples = x.shape[0]

        # 1. Build correlation matrix R
        r_vec, _ = self.corr_func(theta, D)

        R = np.eye(num_samples)

        # Nested loop
        # idx = 0
        # for i in range(num_samples):
        #     for j in range(i + 1, num_samples):
        #         val = r_vec[idx, 0]
        #         R[i, j] = val
        #         R[j, i] = val
        #         idx += 1

        # Vectorized
        idx_upper = np.triu_indices(num_samples, k=1)
        R[idx_upper] = r_vec.flatten()
        R.T[idx_upper] = r_vec.flatten()

        # Nugget for numerical stability
        R += np.eye(num_samples) * (10 + num_samples) * np.finfo(float).eps

        # 2. Cholesky decomposition: R = C @ C.T
        try:
            C = cholesky(R, lower=True)
        except np.linalg.LinAlgError:
            return {"valid": False}

        # 3. GLS auxiliary matrices
        try:
            # Solve C @ Yt = y -> Yt = C^{-1} @ y
            Yt = solve_triangular(C, y, lower=True)

            # Calculate regression matrix F
            F, _ = self.reg_func(x)

            # Solve C @ Ft = F -> Ft = C^{-1} @ F
            Ft = solve_triangular(C, F, lower=True)
        except np.linalg.LinAlgError:
            return {"valid": False}

        # 4. QR factorization: Ft = Q @ G
        # Scipy QR: Q is (m, p) orthonormal, G is (p, p) Upper Triangular
        Q, G = qr(Ft, mode="economic")

        # 5. Calculate beta: G @ beta = Q.T @ Yt
        # G is upper, use back-substitution (lower=False)
        try:
            QT_Yt = Q.T @ Yt
            beta = solve_triangular(G, QT_Yt, lower=False)
        except np.linalg.LinAlgError:
            return {"valid": False}

        # 6. Calculate Residuals and Sigma2
        # Standard residuals in decorrelated space: rho = Yt - Ft @ beta
        rho = Yt - Ft @ beta

        # Process variance (Sigma2) per output dimension
        # sigma2 = ||Yt - Ft @ beta_hat||_2^2 * (1 / num_samples)
        sigma2 = np.sum(rho ** 2, axis=0) / num_samples

        # 7. Calculate Gamma for prediction
        # gamma = R^{-1} @ (y - F @ beta) = C^{-T} @ rho
        # C.T is Upper, use back-substitution (lower=False)
        gamma = solve_triangular(C.T, rho, lower=False)

        return {"C": C, "G": G, "Ft": Ft, "beta": beta, "gamma": gamma, "sigma2": sigma2, "valid": True}

    # ------------------------------------------------------------------

    def _objective_function(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray, D: np.ndarray) -> float:
        """
        Objective function for maximizing the concentrated log-likelihood of Kriging

        We minimize the negative concentrated log-likelihood:
        log L_c_neg(theta) = (m/2) * log(sigma2_hat) + log(det(R))/2

        Args:
            theta (np.ndarray): Correlation parameters to be optimized.
                Shape: (input_dim,), dtype: float64.
            x (np.ndarray): Scaled inputs (design sites).
                Shape: (num_samples, input_dim), dtype: float64.
            y (np.ndarray): Scaled targets.
                Shape: (num_samples, target_dim), dtype: float64.
            D (np.ndarray): Unique difference matrix between design sites.
                Shape: (num_diffs, input_dim), dtype: float64.

        Returns:
            float: The negative concentrated log-likelihood for minimization.
        """

        num_samples = x.shape[0]
        target_dim = y.shape[1]

        # Perform Generalized Least Squares (GLS) estimation
        results = self._fit_gls(x, y, theta, D)

        if not results["valid"]:
            return 1e20
        if np.any(results["sigma2"] <= 0):
            return 1e20

        # Calculate log determinant: log(|R|) = 2 * sum(log(diag(C)))
        log_det_R = 2.0 * np.sum(np.log(np.diag(results["C"])))

        # Compute negative concentrated log-likelihood (Summed over all outputs)
        # Obj = sum_j [m * log(sigma_j^2) + log_det_R]
        # Note: log_det_R is shared across all outputs because theta (and thus R) is shared
        neg_log_likelihood = num_samples * np.sum(np.log(results["sigma2"])) + target_dim * log_det_R

        return neg_log_likelihood

    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform model training.

        Args:
            x_train (np.ndarray): Training inputs of shape: (num_samples, input_dim), dtype: float64.
            y_train (np.ndarray): Training targets of shape: (num_samples, target_dim), dtype: float64.
        """
        num_samples, input_dim = x_train.shape

        # 1. Preprocessing: scale train inputs and targets
        x_scaled = self.scaler_x.fit(x_train, channel_dim=1).transform(x_train)
        y_scaled = self.scaler_y.fit(y_train, channel_dim=1).transform(y_train)
        self.x_train_scaled = x_scaled

        # 2. Calculate unique difference matrix D (shared)

        # Nested loop
        # num_diffs = int(num_samples * (num_samples - 1) / 2)
        # D = np.zeros((num_diffs, input_dim))
        # k = 0
        # for i in range(num_samples):
        #     for j in range(i + 1, num_samples):
        #         D[k, :] = x_scaled[i, :] - x_scaled[j, :]
        #         k += 1

        # Vectorized
        idx_i, idx_j = np.triu_indices(num_samples, k=1)
        D = x_scaled[idx_i] - x_scaled[idx_j]

        # 3. Initial theta guess and optimization setup
        if isinstance(self.theta0, (int, float)):
            theta_initial = np.ones(input_dim) * self.theta0
        else:
            theta_initial = self.theta0.flatten()

        if self.theta_bounds:
            bounds = Bounds(
                lb=np.ones(input_dim) * self.theta_bounds[0],
                ub=np.ones(input_dim) * self.theta_bounds[1])
        else:
            bounds = None

        # 4. Optimization: Minimize the joint negative concentrated log-likelihood
        res = minimize(fun=self._objective_function, x0=theta_initial, args=(x_scaled, y_scaled, D),
            method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})

        # 5. Final model construction: recompute with optimal theta
        self.theta = res.x
        final_fit = self._fit_gls(x_scaled, y_scaled, self.theta, D)

        if not final_fit["valid"]:
            raise ValueError("Fit failed (Cholesky decomposition error)")

        self.beta = final_fit["beta"]
        self.gamma = final_fit["gamma"]
        self.sigma2 = final_fit["sigma2"]
        self.C = final_fit["C"]
        self.G = final_fit["G"]
        self.Ft = final_fit["Ft"]

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        Perform model prediction.

        Args:
            x_pred (np.ndarray): Prediction inputs of shape: (num_samples, input_dim), dtype: float64.

        Returns:
            y_pred (np.ndarray): Prediction targets of shape: (num_samples, target_dim), dtype: float64.
            mse_pred (np.ndarray): Prediction mse values of shape: (num_samples, target_dim), dtype: float64.
        """
        # 1. Preprocessing: scale pred inputs
        if self.beta is None:
            raise RuntimeError("Model has not been fitted.")

        num_train = self.x_train_scaled.shape[0]
        num_pred, input_dim = x_pred.shape

        x_scaled = self.scaler_x.transform(x_pred)

        # 2. Perform predictions: loop over pred points
        # logger.info(f'predicting KRG (poly: {self.poly_name}, kernel: {self.kernel_name})...')

        # Nested loop
        # for i in range(num_pred):
        #     # Current test point (1, n)
        #     x_curr = self.x_scaled[i : i + 1, :]

        #     # Calculate distances d = x_train - x
        #     d_i = x_train_scaled - x_curr

        #     # Get correlation vector r (num_train, 1)
        #     r_i, _ = self.corr_func(self.theta, d_i)

        #     # Get regression vector f (1, p)
        #     f_i, _ = self.reg_func(x_curr)

        #     # Prediction (Mean)
        #     # Formula: y_hat = f * beta + r.T * gamma
        #     # Shapes: (1, p) @ (p, q) + (1, m) @ (m, q) -> (1, q)
        #     y_hat = f_i @ self.beta + r_i.T @ self.gamma
        #     y_pred_scaled[i, :] = y_hat

        #     # Prediction (MSE)
        #     # Formula: sigma2 * (1 + ||G^{-T} * u||^2 - ||C^{-1} * r||^2)
        #     # Note: The term inside brackets is scalar per test point, sigma2 is vector

        #     # 1). Calculate rt = C^{-1} * r
        #     rt = solve_triangular(self.C, r_i, lower=True)

        #     # 2). Calculate u = Ft.T * rt - f.T
        #     # Shapes: (p, m) @ (m, 1) - (1, p).T -> (p, 1)
        #     u = self.Ft.T @ rt - f_i.T

        #     # 3). Calculate v = G^{-T} * u
        #     v = solve_triangular(self.G.T, u, lower=True)

        #     # 4). MSE factor (scalar for this x)
        #     # mse_factor = 1 + v^T @ v - rt^T @ rt
        #     mse_factor = 1.0 + np.sum(v ** 2) - np.sum(rt ** 2)
        #     mse_factor = max(mse_factor, 0.0)  # Ensure non-negative

        #     # 5). Final MSE (vector for multiple outputs)
        #     mse_pred_scaled[i, :] = self.sigma2 * mse_factor

        # Vectorized
        # Step A: Calculate cross-correlations
        d_matrix_3d = self.x_train_scaled[:, np.newaxis, :] - x_scaled[np.newaxis, :, :]
        d_flat = d_matrix_3d.reshape(-1, input_dim)
        r_flat, _ = self.corr_func(self.theta, d_flat)
        R_cross = r_flat.reshape(num_train, num_pred)

        # Step B: Calculate regression basis
        F_test, _ = self.reg_func(x_scaled)

        # Step C: Prediction (Mean)
        # Formula: y_hat = f * beta + r.T * gamma
        # Shapes: (num_pred, p) @ (p, q) + (num_pred, num_train) @ (num_train, q) -> (num_pred, q)
        y_pred_scaled = F_test @ self.beta + R_cross.T @ self.gamma

        # Step D: Prediction (MSE Uncertainty)
        # Formula: sigma2 * (1 + ||G^{-T} * u||^2 - ||C^{-1} * r||^2)
        # Note: The term inside brackets is scalar per test point, sigma2 is vector

        # 1). Calculate RT = C^{-1} * R_cross
        RT = solve_triangular(self.C, R_cross, lower=True)

        # 2). Calculate U = Ft.T * RT - F_test.T
        # Shapes: (p, num_train) @ (num_train, num_pred) - (p, num_pred) -> (p, num_pred)
        U = self.Ft.T @ RT - F_test.T

        # 3). Calculate V = G^{-T} * U
        V = solve_triangular(self.G.T, U, lower=True)

        # 4). Calculate var_factor = 1 + v^T @ v - rt^T @ rt
        sum_v2 = np.sum(V ** 2, axis=0)
        sum_rt2 = np.sum(RT ** 2, axis=0)
        var_factor = 1.0 + sum_v2 - sum_rt2
        var_factor = np.maximum(var_factor, 0.0)  # Ensure non-negative

        # 5). Calculate uncertainty
        var_pred_scaled = np.outer(var_factor, self.sigma2)

        # 3. Unscale predictions (both mean and variance)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        var_pred = var_pred_scaled * (self.scaler_y.std ** 2)

        return y_pred, var_pred
