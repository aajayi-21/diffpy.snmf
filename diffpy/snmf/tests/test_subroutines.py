import pytest
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
    ([.2220, [1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
     ([1, 5.5045, 0, 0, 0, 0, 0, 0, 0], [0, -20.2906, 0, 0, 0, 0, 0, 0, 0], [182.7978, 0, 0, 0, 0, 0, 0, 0, 0]))
]


@pytest.mark.parametrize('tgsc', tgsc)
def test_get_stretched_components(tgsc):
    actual = get_stretched_component(tgsc[0][0], tgsc[0][1], tgsc[0][2])
    expected = tgsc[1]
    print(actual)
    assert (actual == expected).all()
