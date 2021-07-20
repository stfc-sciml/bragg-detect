import multiprocessing as mp
import time

import h5py
import numpy as np
from skimage.feature import blob_log
from sklearn.mixture import BayesianGaussianMixture

from bragg_detect.utils import get_other_dims, to_flattened, to_structured


def blob_log_2d(img, min_sigma, max_sigma, num_sigma, threshold,
                overlap, log_scale):
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold,
                     overlap=overlap, log_scale=log_scale)
    blobs[:, 2:] *= np.sqrt(2)
    return np.ceil(blobs).astype(int)


def find_2d_blobs(data, loc, width, extend,
                  min_sigma, max_sigma, num_sigma, threshold,
                  overlap, log_scale):
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
        detect_blob = blob_log_2d(detect_img,
                                  min_sigma=min_sigma[other_dims],
                                  max_sigma=max_sigma[other_dims],
                                  num_sigma=num_sigma, threshold=threshold,
                                  overlap=overlap, log_scale=log_scale)
        if len(detect_blob) == 0:
            # if one dim is null, 3d is null
            return empty_blobs, detect_block

        verify_blob = blob_log_2d(verify_img,
                                  min_sigma=min_sigma[other_dims],
                                  max_sigma=max_sigma[other_dims],
                                  num_sigma=num_sigma, threshold=threshold,
                                  overlap=overlap, log_scale=log_scale)
        if len(verify_blob) == 0:
            # if one dim is null, 3d is null
            return empty_blobs, detect_block

        # convert both to global and intersect
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
    xy_dims = get_other_dims(z_dim)
    r_xy = blob[2:4] if fixed_radii is None else fixed_radii[xy_dims]
    x = np.arange(max(blob[0] - r_xy[0], 0),
                  min(blob[0] + r_xy[0] + 1, block_shape[xy_dims[0]]))
    y = np.arange(max(blob[1] - r_xy[1], 0),
                  min(blob[1] + r_xy[1] + 1, block_shape[xy_dims[1]]))
    return x, y


def find_blob_wise_peaks(blobs, block, fixed_radii):
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
    # candidates along each axis
    candidates_x = extrude_blobs_1d(blobs[0], 0, block_shape, fixed_radii)
    candidates_y = extrude_blobs_1d(blobs[1], 1, block_shape, fixed_radii)
    candidates_z = extrude_blobs_1d(blobs[2], 2, block_shape, fixed_radii)

    # intersection
    candidates_flattened = np.intersect1d(
        np.intersect1d(candidates_x, candidates_y, assume_unique=True),
        candidates_z, assume_unique=True)
    return candidates_flattened


def intersect_value_peaks(candidates_flattened, block):
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
        max_loc = np.argwhere(values_in_cluster == np.max(values_in_cluster))
        max_peaks = np.concatenate((max_peaks, peaks_in_cluster[max_loc[0]]))
    return max_peaks


def peaks_local_to_global(peak_structured_local, block_loc, data_shape):
    peak_structured_global = peak_structured_local.copy()
    peak_structured_global[:, 0] += block_loc[0]
    peak_structured_global[:, 1] += block_loc[1]
    peak_structured_global[:, 2] += block_loc[2]
    return to_flattened(peak_structured_global, data_shape)


# function for Pool
def detect_peaks_pool(
        # owned
        xl, xw, xe, yl, yw, ye, zl, zw, ze, i_block,
        # shared
        n_blocks, t0, data,
        min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale,
        strategy_3d, fixed_radii, n_components, n_init,
        verbose):
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
        raise RuntimeError(f'Unsupported 3D strategy: {strategy_3d}')

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

    chunk = max(int(n_blocks / workers / 8), 1)
    with mp.Pool(processes=workers) as pool:
        peaks_global_pool = pool.starmap(detect_peaks_pool, args_pool,
                                         chunksize=chunk)

    # add to list
    peaks_detected = np.ndarray((0, 3), dtype=int)
    for peaks_global_block in peaks_global_pool:
        peaks_detected = np.union1d(peaks_detected, peaks_global_block)

    # data is h5 (filename, dsetname)
    if isinstance(data, tuple):
        (filename, dsetname) = data
        with h5py.File(filename, 'r') as h5:
            data_shape = h5[dsetname].shape
    else:
        data_shape = data.shape
    peaks_detected = to_structured(peaks_detected, data_shape)
    return peaks_detected, time.time() - t0
