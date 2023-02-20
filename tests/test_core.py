import pytest
import numpy as np

from bragg_detect import core


"""@pytest.mark.parametrize('blobs, block_shape, expected',
                         [(np.arange(9).reshape((3,3)), [2,2,2], np.arange(4)),
                         ])"""
@pytest.mark.skip("not yet working")
def test_extrude_blobs_3d():#blobs, block_shape, expected):
    blobs = np.arange(9).reshape((1,3,3))
    block_shape = [2, 2, 2]
    extrusion = core.extrude_blobs_3d(blobs=blobs, block_shape=block_shape, fixed_radii=None)
    expected = np.arange(4)
    assert(np.all(extrusion == expected))

@pytest.mark.skip("not yet working")
def test_extrude_blobs_1d():
    blobs = np.arange(9).reshape((3,3))
    block_shape = [2, 3, 4]
    extrusion = core.extrude_blobs_1d(blobs=blobs, z_dim=0,
                                      block_shape=block_shape,
                                      fixed_radii=None)
    expected = np.arange(4)
    assert (np.all(extrusion == expected))

@pytest.mark.parametrize('block_shape, z_dim, expected',
                         [([1,2,3], 0, [np.arange(2),np.arange(3)]),
                          ([1,2,3], 1, [np.arange(1),np.arange(3)]),
                          ([1,2,3], 2, [np.arange(1),np.arange(2)]),
                          ([2,3,5], 0, [np.arange(3),np.arange(5)]),
                          ([2,4,5], 1, [np.arange(2),np.arange(5)]),
                          ([5,5,5], 2, [np.arange(3),np.arange(5)]),
                          ([0,1,2], 0, [np.arange(1),np.arange(2)]),
                         ])
def test_find_blob_range(block_shape, z_dim, expected):
    blob = np.arange(9)

    foundx, foundy = core.find_blob_range(blob, z_dim=z_dim,
                                          block_shape=block_shape,
                                          fixed_radii=None)

    assert(np.all(foundx == expected[0]))
    assert (np.all(foundy == expected[1]))

