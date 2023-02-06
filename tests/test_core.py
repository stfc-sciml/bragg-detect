import pytest
import numpy as np

from bragg_detect import core

def test_extrude_blobs_3d():
    blobs = np.arange(9).reshape((3,3))
    block_shape = [2,2,2]
    extrusion = core.extrude_blobs_3d(blobs, block_shape, fixed_radii=None)
    expected = np.arange(4)
    assert(np.all(extrusion == expected))

def test_find_blob_range():
    blob = np.arange(9)
    block_shape = [1,3]
    found = core.find_blob_range(blob, z_dim=0, block_shape=block_shape,
                                 fixed_radii=None)
    expected = np.arange(3)
    assert(np.all(found == expected))