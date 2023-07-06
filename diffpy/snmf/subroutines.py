import numpy as np
import scipy.interpolate


def objective_function(residual_matrix, stretching_factor_matrix, smoothness, smoothness_term, component_matrix,
                       sparsity):
    """Defines the objective function of the algorithm and returns its value

    Parameters
    ----------
    residual_matrix: 2d array like
      The matrix where each column is the difference between an experimental PDF/XRD pattern and a calculated PDF/XRD
      pattern at each grid point. Has dimensions R x M where R is the length of each pattern and M is the amount of
      patterns.

    stretching_factor_matrix: 2d array like
      The matrix containing the stretching factors of the calculated component signal. Has dimensions K x M where K is
      the amount of components and M is the number of experimental PDF/XRD patterns.

    smoothness: float
      The coefficient of the smoothness term which determines the intensity of the smoothness term and its behavior.
      It is not very sensitive and is usually adjusted by multiplying it by ten.

    smoothness_term: 2d array like
      The regularization term that ensures that smooth changes in the component stretching signals are favored.
      Has dimensions (M-2) x M where M is the amount of experimentally obtained PDF/XRD patterns, the moment amount.

    component_matrix: 2d array like
      The matrix containing the calculated component signals of the experimental PDF/XRD patterns. Has dimensions R x K
      where R is the signal length and K is the number of component signals.

    sparsity: float
      The parameter determining the intensity of the sparsity regularization term which enables the algorithm to
      exploit the sparse nature of XRD data. It is usually adjusted by doubling.

    Returns
    -------
    float
      The value of the objective function.

    """
    residual_matrix = np.asarray(residual_matrix)
    stretching_factor_matrix = np.asarray(stretching_factor_matrix)
    component_matrix = np.asarray(component_matrix)
    return .5 * np.linalg.norm(residual_matrix, 'fro') ** 2 + .5 * smoothness * np.linalg.norm(
        smoothness_term @ stretching_factor_matrix.T, 'fro') ** 2 + sparsity * np.sum(np.sqrt(component_matrix))


def get_stretched_component(stretching_factor, component, grid_vector):
    component = np.asarray(component)
    grid_vector = np.asarray(grid_vector)
    spline = scipy.interpolate.UnivariateSpline(grid_vector, component, k=2, ext=1)
    stretched_component = spline.__call__(grid_vector / stretching_factor)
    stretched_component_first_derivative = spline.__call__(grid_vector / stretching_factor, nu=1)
    stretched_component_second_derivative = spline.__call__(grid_vector / stretching_factor, nu=2)
    return stretched_component, stretched_component_first_derivative,stretched_component_second_derivative


def update_weights_matrix(component_amount, signal_length, stretching_factor_matrix, component_matrix, data_input,
                          moment_amount):
    weight = np.zeros(component_amount)
    for i in range(moment_amount):
        gram_matrix = np.zeros(signal_length, component_amount)
        for n in range(component_amount):
            gram_matrix[:, n] = get_stretched_component(stretching_factor_matrix[n, i], component_matrix[:, n])

        weight
