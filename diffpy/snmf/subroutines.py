import numpy as np


def objective_function(residual_matrix, stretching_factor_matrix, smoothness, sparsity_term, component_matrix,
                       sparsity):
    """Defines the objective function of the algorithm and returns its value

    Parameters
    ----------
    residual_matrix: 2d array like
      The matrix where each column is the difference between an experimental PDF/XRD pattern and a calculated PDF/XRD
      pattern at each grid point. Has dimensions R x M where R is the length of each pattern and M is the amount of
      patterns.

    stretching_factor_matrix: 2d array like
      The matrix containing the stretching factors of the calculated component signal. Has dimensions

    smoothness: float
      The coefficient of the


    sparsity_term
    component_matrix
    sparsity

    Returns
    -------

    """
    return .5 * np.linalg.norm(residual_matrix, 'fro') ** 2 + .5 * smoothness * np.linalg.norm(
        sparsity_term @ stretching_factor_matrix.T, 'fro') ** 2 + sparsity * np.sum(np.sqrt(component_matrix))
