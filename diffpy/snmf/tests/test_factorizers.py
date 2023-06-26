import numpy as np
import scipy
import pytest
from diffpy.snmf.factorizers import lsqnonneg

tl = [
    ([[[1, 0], [1, 0], [0, 1]], [2, 1, 1]], [1.5, 1.]),
    ([[[2, 3], [1, 2], [0, 0]], [7, 7, 2]], [0, 2.6923]),
    ([[[3, 2, 4, 1]], [3.2]], [0, 0, .8, 0]),
    ([[[-.4, 0], [0, 0], [-9, -18]], [-2, -3, -4.9]], [.5532, 0])
]


@pytest.mark.parametrize('tl', tl)
def test_lsqnonneg(tl):
    actual = lsqnonneg(tl[0][0], tl[0][1])
    expected = tl[1]
    np.testing.assert_array_almost_equal(actual, expected, decimal=4)
