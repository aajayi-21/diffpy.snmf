import numpy as np


def rooth(linear_coefficient, constant_term):
    """
    Returns the largest real root of x^3+(linear_coefficient) * x + constant_term. If there are no real roots return 0.
    Parameters
    ----------
    linear_coefficient
    constant_term

    Returns
    -------

    """
    inputs = [1, 0, linear_coefficient, constant_term]
    y = np.roots(inputs)
    if ((constant_term / 2) ** 2 + (linear_coefficient / 3) ** 3) < 0:  # Discriminant of depressed cubic equation
        return max(np.real(y))
    else:
        return 0