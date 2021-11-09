import pytest
import numpy as np
from rflearn.utils import argmax


@pytest.mark.parametrize(
    'data, idx, seed',
    [
        ([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 8, None),
        ([1, 0, 0, 1], 0, 0)
    ]
)
def test_argmax(data, idx, seed):
    if seed is not None:
        np.random.seed(seed)
    
    assert argmax(data) == idx

def test_argmax2():
    # set random seed so results are deterministic
    np.random.seed(0)
    test_array = [1, 0, 0, 1]

    counts = [0, 0, 0, 0]
    for _ in range(100):
        a = argmax(test_array)
        counts[a] += 1

    # make sure argmax does not always choose first entry
    assert counts[0] != 100, "Make sure your argmax implementation randomly choooses among the largest values."

    # make sure argmax does not always choose last entry
    assert counts[3] != 100, "Make sure your argmax implementation randomly choooses among the largest values."

    # make sure the random number generator is called exactly once whenver `argmax` is called
    expected = [44, 0, 0, 56] # <-- notice not perfectly uniform due to randomness
    assert counts == expected
