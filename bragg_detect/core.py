import itertools
import multiprocessing as mp
import time

import h5py
import numpy as np
from skimage.feature import blob_log
from sklearn.mixture import BayesianGaussianMixture

from bragg_detect.utils import get_other_dims, to_flattened, to_structured

'''
def blob_log_2d(img, min_sigma, max_sigma, num_sigma, threshold,
                overlap, log_scale):
    """
    Finds blobs in the given grayscale image.
    Blobs are found using the Laplacian of Gaussian (LoG) method. For each
    blob found, the method returns its coordinates and the standard deviation
    of the Gaussian kernel that detected the blob.

    :param img:np.ndarray, Input grayscale image, blobs are assumed to be light
    on dark background (white on black).
    :param min_sigma: Union[float, array], the minimum standard deviation for
    Gaussian kernel. Keep this low to detect smaller blobs.
    :param max_sigma: Union[float, array], the maximum standard deviation for
    Gaussian kernel. Keep this high to detect larger blobs.
    :param num_sigma: Optional[int], the number of intermediate values of
    standard deviations to consider between min_sigma and max_sigma, default 10
    :param threshold: Optional[float], The absolute lower bound for scale space
    maxima. Local maxima smaller than threshold are ignored. Reduce this to
    detect blobs with lower intensities.
    :param overlap: Optional[float], a value between 0 and 1. If the area of
    two blobs overlaps by a fraction greater than threshold, the smaller blob
    is eliminated. Defaults to 0.5
    :param log_scale: Optional[bool], If set intermediate values of standard
    deviations are interpolated using a logarithmic scale to the base 10. If
    not, linear interpolation is used.

    :return: np.ndarray, A 2d array with each row representing 2 coordinate
    values for a 2D image, or 3 coordinate values for a 3D image, plus the
    sigma(s) used. When a single sigma is passed, outputs are: (r, c, sigma) or
    (p, r, c, sigma) where (r, c) or (p, r, c) are coordinates of the blob and
    sigma is the standard deviation of the Gaussian kernel which detected the
    blob. When an anisotropic gaussian is used (sigmas per dimension), the
    detected sigma is returned for each dimension.
    """
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold,
                     overlap=overlap, log_scale=log_scale)
    blobs[:, 2:] *= np.sqrt(2)
    return np.ceil(blobs).astype(int)
'''

def blob_log_process(blobs):
    """
    Convert from blob's sigma, to a blob radius (sqrt(2)*sigma for 2D image),
    then return the ceiling, since we want pixel numbers.
    :param blobs: np.ndarray, 2d array with each row representing 2 coordinate
    values for a 2D image, or 3 coordinate values for a 3D image, plus the
    sigma(s) used.
    :return: np.ndarray
    """
    blobs[:, 2:] *= np.sqrt(2)
    return np.ceil(blobs).astype(int)


def find_2d_blobs(data, loc, width, extend,
                  min_sigma, max_sigma, num_sigma, threshold,
                  overlap, log_scale):
    """
    Slice the data into small blocks and then call the function blog_log to
    detect peaks withing a block and veryify the peaks using a larger block
    size (since real peaks should exist in both bock sizes).
    :param data:the 3D data as a numpy.ndarray or a tuple
        (filename, dsetname) to specify a HDF5 dataset storing the 3D data;
        when using multiple works, use (filename, dsetname) for both better
        performance and less memory consumption
    :param loc: int, location of the block in pixels
    :param width: int, width in pixels of the blocks
    :param extend: int, number of pixels to extend the block for the
    verification block
    :param min_sigma: float, min_sigma for blob_log() of scikit-image;
        default is None, or large_peak_size // 4
    :param max_sigma: float, for blob_log() of scikit-image;
        default is None, or large_peak_size
    :param num_sigma: float, num_sigma for blob_log() of scikit-image;
        default is 5
    :param threshold: float, threshold for blob_log() of scikit-image;
        default is 0.2
    :param overlap: float, overlap for blob_log() of scikit-image;
        default is 0.5
    :param log_scale: bool: option to use a log_scale for blob_log() default
    is False

    :return: tuple,
    """
    # slice detect block
    detect_begin = np.clip(loc, 0, data.shape)
    detect_end = np.clip(loc + width, 0, data.shape)
    detect_block = data[
                   detect_begin[0]:detect_end[0],
                   detect_begin[1]:detect_end[1],
                   detect_begin[2]:detect_end[2]]

    detect_blobs = []
    empty = np.ndarray((0, 4), dtype=int)
    empty_blobs = [empty, empty, empty]
    for dim in [0, 1, 2]:
        # slice verify block
        verify_begin = np.clip(loc - extend, 0, data.shape)
        verify_end = np.clip(loc + width + extend, 0, data.shape)
        verify_begin[dim] = detect_begin[dim]  # only extend other dims
        verify_end[dim] = detect_end[dim]  # only extend other dims
        verify_block = data[
                       verify_begin[0]:verify_end[0],
                       verify_begin[1]:verify_end[1],
                       verify_begin[2]:verify_end[2]]

        # integrate blocks into images
        detect_img = detect_block.sum(axis=dim)
        verify_img = verify_block.sum(axis=dim)

        # normalize both by max in detect
        norm = np.max(detect_img)
        if norm < 1e-10:
            # if one dim is null, 3d is null
            return empty_blobs, detect_block
        detect_img /= norm
        verify_img /= norm

        # find blobs in detect images
        other_dims = get_other_dims(dim)
        detect_blob = blob_log(detect_img,
                                  min_sigma=min_sigma[other_dims],
                                  max_sigma=max_sigma[other_dims],
                                  num_sigma=num_sigma, threshold=threshold,
                                  overlap=overlap, log_scale=log_scale)
        verify_blob = blob_log(verify_img,
                                  min_sigma=min_sigma[other_dims],
                                  max_sigma=max_sigma[other_dims],
                                  num_sigma=num_sigma, threshold=threshold,
                                  overlap=overlap, log_scale=log_scale)

        if len(detect_blob) or len(verify_blob) == 0:
            # if one dim is null, 3d is null
            return empty_blobs, detect_block

        detect_blob = blob_log_process(detect_blob)
        verify_blob = blob_log_process(verify_blob)

        # convert both to global coordinates and intersect
        detect_blob[:, 0] += detect_begin[other_dims[0]]
        detect_blob[:, 1] += detect_begin[other_dims[1]]
        verify_blob[:, 0] += verify_begin[other_dims[0]]
        verify_blob[:, 1] += verify_begin[other_dims[1]]

        dims_2d = np.array(data.shape)[other_dims]
        detect_flattened = to_flattened(detect_blob[:, :2], dims_2d)
        verify_flattened = to_flattened(verify_blob[:, :2], dims_2d)
        _, comm_indices, _ = np.intersect1d(detect_flattened, verify_flattened,
                                            return_indices=True)
        detect_blob = detect_blob[comm_indices, :]

        if len(detect_blob) == 0:
            # if one dim is null, 3d is null
            return empty_blobs, detect_block

        # convert back to local
        detect_blob[:, 0] -= detect_begin[other_dims[0]]
        detect_blob[:, 1] -= detect_begin[other_dims[1]]
        detect_blobs.append(detect_blob)
    return detect_blobs, detect_block


def find_blob_range(blob, z_dim, block_shape, fixed_radii=None):
    """
    Find the range of pixels corresponding to the blob in 2D
    :param blob: np.ndarray,
    :param z_dim: int, dimension in [0, 1, 2] not necessarily z
    :param block_shape: np.ndarray, shape of block in [x, y, z]
    :param fixed_radii:  int, size of radius in pixels
    :return: tuple[np.ndarray, np.ndarray], pixel values in 2D coordinates
    which are inside the blob radius
    """
    xy_dims = get_other_dims(z_dim)
    r_xy = blob[2:4] if fixed_radii is None else fixed_radii[xy_dims]
    x = np.arange(max(blob[0] - r_xy[0], 0),
                  min(blob[0] + r_xy[0] + 1, block_shape[xy_dims[0]]))
    y = np.arange(max(blob[1] - r_xy[1], 0),
                  min(blob[1] + r_xy[1] + 1, block_shape[xy_dims[1]]))
    return x, y


def find_blob_wise_peaks(blobs, block, fixed_radii):
    """

    :param blobs:
    :param block:
    :param fixed_radii:
    :return:
    """
    max_peaks = np.ndarray((0, 3), dtype=int)
    for blob_x in blobs[0]:
        y_x, z_x = find_blob_range(blob_x, 0, block.shape, fixed_radii)
        for blob_y in blobs[1]:
            x_y, z_y = find_blob_range(blob_y, 1, block.shape, fixed_radii)
            z = np.intersect1d(z_x, z_y)
            if len(z) == 0:
                continue
            for blob_z in blobs[2]:
                x_z, y_z = find_blob_range(blob_z, 2, block.shape, fixed_radii)
                x = np.intersect1d(x_y, x_z)
                y = np.intersect1d(y_x, y_z)
                if len(x) == 0 or len(y) == 0:
                    continue
                # peak volume in 3d
                mesh = np.array(np.meshgrid(x, y, z, indexing='ij'))
                mesh = np.moveaxis(mesh, 0, 3)
                mesh = np.reshape(mesh, (-1, 3))
                # find max value in this volume
                values = block[(mesh[:, 0], mesh[:, 1], mesh[:, 2])]
                max_loc = np.argwhere(values == np.max(values))
                max_peaks = np.concatenate((max_peaks, mesh[max_loc[0]]))
    return max_peaks


def extrude_blobs_1d(blobs, z_dim, block_shape, fixed_radii=None):
    """
    Take a blob from skimage, and extend the blob to a cylindrical volume
    across z_dim
    :param blobs: np.ndarray,
    :param z_dim: int, in [0, 1, 2] representing any dimension
    :param block_shape:
    :param fixed_radii:
    :return:
    """
    # loop over blobs
    candidates_3d = np.ndarray((0, 3), dtype=int)
    z = np.arange(block_shape[z_dim])
    for blob in blobs:
        x, y = find_blob_range(blob, z_dim, block_shape, fixed_radii)
        # structured locations
        if z_dim == 0:
            mesh = np.array(np.meshgrid(z, x, y, indexing='ij'))
        elif z_dim == 1:
            mesh = np.array(np.meshgrid(x, z, y, indexing='ij'))
        else:
            mesh = np.array(np.meshgrid(x, y, z, indexing='ij'))
        # move tuple dim (which equals to 3) to last
        mesh = np.moveaxis(mesh, 0, 3)
        # add to locations
        candidates_3d = np.concatenate(
            (candidates_3d, np.reshape(mesh, (-1, 3))))

    # flatten and unique
    candidates_3d = to_flattened(candidates_3d, block_shape)
    candidates_3d = np.unique(candidates_3d)
    return candidates_3d


def extrude_blobs_3d(blobs, block_shape, fixed_radii=None):
    """
    Apply 1D extrusion across all 3 dimensions and find the intersection, then
    return the interstected volume
    :param blobs: np.ndarray, array of candidate blob
    :param block_shape:
    :param fixed_radii:
    :return:
    """
    # extrude candidates along each axis
    candidates_x = extrude_blobs_1d(blobs[0], 0, block_shape, fixed_radii)
    candidates_y = extrude_blobs_1d(blobs[1], 1, block_shape, fixed_radii)
    candidates_z = extrude_blobs_1d(blobs[2], 2, block_shape, fixed_radii)

    # intersection
    candidates_flattened = np.intersect1d(
        np.intersect1d(candidates_x, candidates_y, assume_unique=True),
        candidates_z, assume_unique=True)
    return candidates_flattened


def intersect_value_peaks(candidates_flattened, block):
    """
    Takes the intersection of x,y, and z extruded volumes in order to determine
    peaks have been found in all three dimensions. Then look within these
    intersected volume to find the actual peak locations.
    :param candidates_flattened: np.ndarray[int], array of peak indicies
    :param block: np.ndarray, the subset of the entire dataset being
    investigated

    :return: np.ndarray[int], unflattened array of validated peak coordinates
    """
    # structured
    candidates_structured = to_structured(candidates_flattened, block.shape)

    # shift indices
    i = candidates_structured[:, 0]
    j = candidates_structured[:, 1]
    k = candidates_structured[:, 2]
    ip = np.minimum(i + 1, block.shape[0] - 1)
    im = np.maximum(i - 1, 0)
    jp = np.minimum(j + 1, block.shape[1] - 1)
    jm = np.maximum(j - 1, 0)
    kp = np.minimum(k + 1, block.shape[2] - 1)
    km = np.maximum(k - 1, 0)

    # shifted values
    v = block[(i, j, k)]
    vip = block[(ip, j, k)]
    vim = block[(im, j, k)]
    vjp = block[(i, jp, k)]
    vjm = block[(i, jm, k)]
    vkp = block[(i, j, kp)]
    vkm = block[(i, j, km)]

    # find peaks
    peak_index = np.where(
        (v > vip) * (v > vim) * (v > vjp) * (v > vjm) * (v > vkp) * (v > vkm))
    peak_structured = candidates_structured[peak_index[0]]
    return peak_structured


def bgm_clustering(peak_structured, block, n_components, n_init):
    """
    Using sklearn's BayesianGaussianMixture, estimate the parameters of
    peak_structured and predict the position

    :param peak_structured: array, List of n_components-dimensional data
    points. Each row corresponds to a single data point
    :param block:
    :param n_components: int, The number of mixture components. The model can
    decide to not use all components, so number of effective components
    is smaller than n_components
    :param n_init: int, The number of initializations to perform. The result
    with the highest lower bound value on the likelihood is kept
    :return: array, the coordinates of the largest peaks found
    """
    # cluster
    bgm = BayesianGaussianMixture(n_components=n_components, n_init=n_init)
    labels = bgm.fit_predict(peak_structured)

    # for each cluster, only keep max-valued peaks
    max_peaks = np.ndarray((0, 3), dtype=int)
    for label in np.arange(n_components):
        peaks_in_cluster = peak_structured[np.where(labels == label)[0]]
        if len(peaks_in_cluster) == 0:
            continue
        values_in_cluster = block[(peaks_in_cluster[:, 0],
                                   peaks_in_cluster[:, 1],
                                   peaks_in_cluster[:, 2])]
        max_peaks = np.concatenate((max_peaks, np.argmax(values_in_cluster)))
    return max_peaks


def peaks_local_to_global(peak_structured_local, block_loc, data_shape):
    """
    Convert between the local block coordinates and the global ones
    :param peak_structured_local: array, local peak coordinates
    :param block_loc: array, global block coordinates
    :param data_shape: array, shape of the data
    :return:
    """
    peaks_global = peak_structured_local.copy()

    for index in np.arrange(3):
        peaks_global[:, index] += block_loc[index]

    return to_flattened(peaks_global, data_shape)


# function for Pool
def detect_peaks_pool(
        # owned
        xl, xw, xe, yl, yw, ye, zl, zw, ze, i_block,
        # shared
        n_blocks, t0, data,
        min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale,
        strategy_3d, fixed_radii, n_components, n_init,
        verbose):
    """
    Wrapper for whole process on each block in order to send it to pool for
    multiprocessing
    :param xl: float, block location in x
    :param xw: float, block width in x
    :param xe: float, block extent in x
    :param yl: float, block location in y
    :param yw: float, block width in y
    :param ye: float, block extent in y
    :param zl: float, block location in z
    :param zw: float, block width in z
    :param ze: float, block extent in z
    :param i_block: int, current block number
    :param n_blocks: int, total number of blocks
    :param t0: start time in seconds
    :param data: Union[tuple, array], the dataset in which to loacte peaks
    :param min_sigma: float, smallest width of peaks to look for in pixels
    :param max_sigma: float, largest width of peaks to look for in pixels
    :param num_sigma: int, The number of intermediate values of standard
    deviations to consider between min_sigma and max_sigma.
    :param threshold: float, The absolute lower bound for scale space maxima.
    Local maxima smaller than threshold are ignored.
    :param overlap: float, A value between 0 and 1. If the area of two blobs
    overlaps by a fraction greater than threshold, the smaller blob is
    eliminated
    :param log_scale: bool, If set intermediate values of standard deviations
    are interpolated using a logarithmic scale to the base 10. If not, linear
    interpolation is used.
    :param strategy_3d: str, strategy to use, allowed values are 'individual'
    which would use Laplacian of Gaussian, or 'bgm_clustering' which uses
    a Bayesian Gaussain Mixture model.
    :param fixed_radii:
    :param n_components:int, number of components for bgm
    :param n_init:
    :param verbose: bool, option to print output
    :return:
    """
    # data is h5 (filename, dsetname)
    if isinstance(data, tuple):
        (filename, dsetname) = data
        h5 = h5py.File(filename, 'r')
        data = h5[dsetname]

    # 2d blobs in block
    block_loc = np.array([xl, yl, zl])
    block_width = np.array([xw, yw, zw])
    block_extend = np.array([xe, ye, ze])
    blobs, block = find_2d_blobs(
        data, block_loc, block_width, block_extend,
        min_sigma, max_sigma, num_sigma,
        threshold, overlap, log_scale)
    if len(blobs[0]) * len(blobs[1]) * len(blobs[2]) == 0:
        return np.ndarray((0, 3), dtype=int)

    # 3d peaks
    if strategy_3d == 'bgm_clustering':
        # extrude 3d
        candidates = extrude_blobs_3d(blobs, block.shape, fixed_radii)
        if len(candidates) == 0:
            return np.ndarray((0, 3), dtype=int)
        # value peaks
        peaks_local = intersect_value_peaks(candidates, block)
        if len(peaks_local) == 0:
            return np.ndarray((0, 3), dtype=int)
        # clustering
        if len(peaks_local) > n_components:
            peaks_local = bgm_clustering(peaks_local, block,
                                         n_components, n_init)
    elif strategy_3d == 'individual':
        peaks_local = find_blob_wise_peaks(blobs, block, fixed_radii)
    else:
        raise ValueError(f'Unsupported 3D strategy: {strategy_3d}')

    # local to global
    peaks_global = peaks_local_to_global(peaks_local, block_loc, data.shape)
    if verbose and len(peaks_global) > 0:
        print(f'Block {i_block} / {n_blocks}: '
              f'peaks in block = {len(peaks_global)}; '
              f'elapsed sec = {time.time() - t0:.1f}; '
              f'process pid = {mp.current_process().pid}',
              flush=True)
    return peaks_global


def detect_peaks(data, strategy_3d,
                 x_loc, x_width, x_extend,
                 y_loc, y_width, y_extend,
                 z_loc, z_width, z_extend,
                 min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale,
                 fixed_radii, n_components, n_init, workers, verbose):
    """
    Main function for running whole process
    :param data: the 3D data as a numpy.ndarray or a tuple
        (filename, dsetname) to specify a HDF5 dataset storing the 3D data
    :param strategy_3d:
    :param x_loc:
    :param x_width:
    :param x_extend:
    :param y_loc:
    :param y_width:
    :param y_extend:
    :param z_loc:
    :param z_width:
    :param z_extend:
    :param min_sigma:
    :param max_sigma:
    :param num_sigma:
    :param threshold:
    :param overlap:
    :param log_scale:
    :param fixed_radii:
    :param n_components:
    :param n_init:
    :param workers: Optional[int], number of processes to use, defaults to 1
    :param verbose: bool
    :return: tuple, (number of peaks detected, time taken)
    """
    # shared
    t0 = time.time()
    n_blocks = len(x_loc) * len(y_loc) * len(z_loc)

    # args for Pool
    args_pool = []
    i_block = 0
    for xl, xw, xe in zip(x_loc, x_width, x_extend):
        for yl, yw, ye in zip(y_loc, y_width, y_extend):
            for zl, zw, ze in zip(z_loc, z_width, z_extend):
                args = (
                    # owned
                    xl, xw, xe, yl, yw, ye, zl, zw, ze, i_block,
                    # shared
                    n_blocks, t0, data,
                    min_sigma, max_sigma, num_sigma, threshold, overlap,
                    log_scale,
                    strategy_3d, fixed_radii, n_components, n_init,
                    verbose)
                args_pool.append(args)
                i_block += 1

    if workers == 1:
        peaks_global_pool = itertools.starmap(detect_peaks_pool, args_pool)
    else:
        chunk = max(int(n_blocks / workers / 8), 1)
        with mp.Pool(processes=workers) as pool:
            peaks_global_pool = pool.starmap(detect_peaks_pool, args_pool,
                                             chunksize=chunk)

    # add to list
    peaks_detected = np.ndarray((0, 3), dtype=int)
    for peaks_global_block in peaks_global_pool:
        peaks_detected = np.union1d(peaks_detected, peaks_global_block)
    #  Does this not overwrite the peaks_detected?
    #  Also, how does this give coordinates??

    # data is h5 (filename, dsetname)
    if isinstance(data, tuple):
        (filename, dsetname) = data
        with h5py.File(filename, 'r') as h5:
            data_shape = h5[dsetname].shape
    else:
        data_shape = data.shape
    peaks_detected = to_structured(peaks_detected, data_shape)
    return peaks_detected, time.time() - t0
