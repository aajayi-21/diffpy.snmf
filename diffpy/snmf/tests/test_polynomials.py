import pytest
from diffpy.snmf.polynomials import rooth

tr = [
    ([0, 0], 0),
    ([-4, 0], 2),
    ([10, 0], 0),
    ([-7, -7], 3.0489),
    ([100, 72], 0),
    ([1, 3], 0),
    ([0, -7], 0),
    ([-9, 0], 3),
    ([-9, 3], 2.8169)
]


@pytest.mark.parametrize("tr", tr)
def test_rooth(tr):
    actual = rooth(tr[0][0], tr[0][1])
    expected = tr[1]
    assert expected == pytest.approx(actual)