import pytest
import numpy as np
from diffpy.snmf.subroutines import objective_function, reconstruct_data

to = [
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], 1e11, [[1, 2], [3, 4]], [[1, 2], [3, 4]], 1], 2.574e14),
    ([[[11, 2], [31, 4]], [[5, 63], [7, 18]], .001, [[21, 2], [3, 4]], [[11, 22], [3, 40]], 1], 650.4576),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], 1e11, [[1, 2], [3, 4]], [[1, 2], [3, 4]], 0], 2.574e14),

]


@pytest.mark.parametrize("to", to)
def test_objective_function(to):
    actual = objective_function(to[0][0], to[0][1], to[0][2], to[0][3], to[0][4], to[0][5])
    expected = to[1]
    assert actual == pytest.approx(expected)


trd = [
    ([np.array([[.5],[.5]]), np.array([[1, 2], [3, 4]]), np.array([[.25],[.75]]), 2, 1, 2],
     np.array([[.25, 1.5], [0, 0]]))

]


@pytest.mark.parametrize('trd', trd)
def test_reconstruct_data(trd):
    actual = reconstruct_data(trd[0][0], trd[0][1], trd[0][2], trd[0][3], trd[0][4],trd[0][5])
    expected = trd[1]
    np.testing.assert_allclose(actual, expected)
