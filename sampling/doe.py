# Design of Experiments (DoE) by Latin Hypercube Design
# Author: Shengning Wang

import numpy as np
from scipy.spatial.distance import pdist
from typing import Optional


def lhs_design(num_samples: int, num_dimensions: int, iterations: Optional[int] = None) -> np.ndarray:
    """
    Generate a latin hypercube sampling design with maximin optimization

    Args:
    - num_samples (int): Number of samples to generate
    - num_dimensions (int): Number of dimensions for each sample
    - iterations (Optional[int]): Number of iterations for maximin optimization.
                                  If None, uses basic LHS without optimization.

    Returns:
    - np.ndarray: The design matrix (num_samples, num_dimensions) normalized to [0, 1]
    """

    def generate_basic_lhs(num_samples: int, num_dimensions: int) -> np.ndarray:
        """
        Generate basic LHS design without optimization

        Returns:
        - np.ndarray: Basic LHS design normalized to [0, 1]
        """

        # Initialize design matrix
        design = np.zeros([num_samples, num_dimensions])

        # Generate stratified samples for each dimension
        for dimension in range(num_dimensions):
            design[:, dimension] = np.random.permutation(num_samples) + np.random.uniform(0.0, 1.0, num_samples)

        # Normalize to [0, 1] range
        design = design / num_samples

        return design


    # Basic LHS without optimization
    if iterations is None:
        return generate_basic_lhs(num_samples, num_dimensions)


    # Maximin optimization version
    best_design = None
    best_min_distance = -np.inf

    for _ in range(iterations):
        current_design = generate_basic_lhs(num_samples, num_dimensions)
        current_min_distance = np.min(pdist(current_design))

        # Update best design if current has larger minimum distance
        if current_min_distance > best_min_distance:
            best_min_distance = current_min_distance
            best_design = current_design.copy()

    return best_design


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == "__main__":
    # Quick test
    num_samples = 1000
    num_dimensions= 5

    num_show = num_samples
    if num_samples > 20:
        num_show = 20

    # Basic LHS
    design = lhs_design(num_samples, num_dimensions)
    print(f'# Basic LHS ({num_samples} samples, {num_dimensions} dimensions):')
    for sample in range(num_show):
        print(design[sample, :])

    # Optimal LHS
    design = lhs_design(num_samples, num_dimensions, 100)
    print(f'\n# Optimal LHS ({num_samples} samples, {num_dimensions} dimensions):')
    for sample in range(num_show):
        print(design[sample, :])
