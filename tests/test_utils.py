import pytest
import numpy as np

from bragg_detect import core, utils

@pytest.mark.parametrize('dimension, expected',
                         [(0, [1,2]),
                          (1,[0,2]),
                          (2, [0,1])])
def test_get_other_dims(dimension, expected):

    assert(utils.get_other_dims(dimension) == expected)

    #  Also check non integers and values outside [0,1,2] raises a ValueError
    with pytest.raises(ValueError):
        _ = utils.get_other_dims(1.5)
    with pytest.raises(ValueError):
        _ = utils.get_other_dims(4)


def test_to_flattened():
    arr = utils.to_flattened(np.arange(4).reshape((2,2)), [0,1])
    assert(arr == np.arange(4))


def test_to_structured():
    flat_array = np.arange(6)
    arr = utils.to_structured(flat_array, [0,1,2])
    assert(arr == np.arange(6).reshape((2,3)))
