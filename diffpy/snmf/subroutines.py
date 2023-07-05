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
      The coefficient of the smoothness term which determines


    smoothness_term: 2d array
      The term that ensures that smooth changes in the component signals are favored.

    component_matrix: 2d array
      The matrix containing the calculated component signals of the experimental PDF/XRD patterns. Has dimesions R x K
      where R is the signal length and K is the number of component signals.

    sparsity: float
      The parameter

    Returns
    -------

    """
    return .5 * np.linalg.norm(residual_matrix, 'fro') ** 2 + .5 * smoothness * np.linalg.norm(
        smoothness_term @ stretching_factor_matrix.T, 'fro') ** 2 + sparsity * np.sum(np.sqrt(component_matrix))


def get_stretched_component(stretching_factor, component, grid_vector):
    spline = scipy.interpolate.UnivariateSpline(grid_vector, component, k=2)
    stretched_component = spline.__call__(grid_vector / stretching_factor)
    stretched_component_first_derivative = spline.__call__(grid_vector / stretching_factor, nu=1)
    stretched_component_second_derivative = spline.__call__(grid_vector / stretching_factor, nu=2)


def update_weights_matrix(component_amount, signal_length, stretching_factor_matrix, component_matrix, data_input,
                          moment_amount):
    weight = np.zeros(component_amount)
    for i in range(moment_amount):
        gram_matrix = np.zeros(signal_length, component_amount)
        for n in range(component_amount):
            gram_matrix[:, n] = get_stretched_component(stretching_factor_matrix[n, i], component_matrix[:, n])

        weight
