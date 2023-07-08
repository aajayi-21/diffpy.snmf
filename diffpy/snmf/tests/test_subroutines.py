import pytest
import numpy as np
from diffpy.snmf.subroutines import objective_function, get_stretched_component

to = [
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], 1e11, [[1, 2], [3, 4]], [[1, 2], [3, 4]], 1], 2.574e14),
    # Positive square matrices which large smoothness
    ([[[11, 2], [31, 4]], [[5, 63], [7, 18]], .001, [[21, 2], [3, 4]], [[11, 22], [3, 40]], 1], 650.4576),
    # Positive square matrix with small smoothness
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], 1e11, [[1, 2], [3, 4]], [[1, 2], [3, 4]], 0], 2.574e14),
    # Positive square matrices which sparseness of 0
    # Rectangular positive matrices
    # Negative Matrices
    # Positive matrices of floats
    # Negative matrices of floats
    # All zero matrices

]


@pytest.mark.parametrize("to", to)
def test_objective_function(to):
    actual = objective_function(to[0][0], to[0][1], to[0][2], to[0][3], to[0][4], to[0][5])
    expected = to[1]
    assert actual == pytest.approx(expected)


tgsc = [
    ([.2220, [1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8]], [1, 5.504504504504505, 0, 0, 0, 0, 0, 0, 0]),
    ([1.2, [1.5, 2.2, 3.5, 4.2, 5.5, 6.2, 7.5, 8.2, 9.5], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
     [1.5, 2.083333333333333, 3.066666666666667, 3.85, 4.633333333333334, 5.616666666666667, 6.200000000000000,
      7.283333333333334, 7.966666666666667]),
]


@pytest.mark.parametrize('tgsc', tgsc)
def test_get_stretched_components(tgsc):
    actual = get_stretched_component(tgsc[0][0], tgsc[0][1], tgsc[0][2])
    expected = tgsc[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
