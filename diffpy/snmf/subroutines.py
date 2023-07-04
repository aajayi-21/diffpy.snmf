import numpy as np


def objective_function(residual_matrix, stretching_factor_matrix, smoothness, sparsity_term, component_matrix,
                       sparsity):
    """Defines the objective function of the algorithm and returns its value

    Parameters
    ----------
    residual_matrix
    stretching_factor_matrix
    smoothness
    sparsity_term
    component_matrix
    sparsity

    Returns
    -------

    """
    return .5 * np.linalg.norm(residual_matrix, 'fro') ** 2 + .5 * smoothness * np.linalg.norm(
        sparsity_term @ stretching_factor_matrix.T, 'fro') ** 2 + sparsity * np.sum(np.sqrt(component_matrix))