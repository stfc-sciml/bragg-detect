import pytest
import numpy as np
from functools import reduce

from bragg_detect import utils

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
    """
    This needs work as I don't understand the outputs yet
    """
    arr2d = utils.to_flattened(np.arange(9).reshape((3, 3)), [0, 1])
    arr3d = utils.to_flattened(np.arange(9).reshape((1, 3, 3)), [0, 1])
    assert(np.all(arr2d == [1,7,13]))
    assert (np.all(arr3d == [3, 5, 7]))


def test_to_structured():
    """
    This needs work as I don't understand the outputs yet
    """
    for dims in [(2,3), (1,2,3), (2,3,4)]:
        flat_array = np.arange(np.product(dims))
        arr = utils.to_structured(flat_array, dims)
        indicies = np.unravel_index(flat_array, dims)
        expected = reduce(lambda x,y: np.vstack((x,y)), indicies).T
        assert(np.all(arr == expected))
