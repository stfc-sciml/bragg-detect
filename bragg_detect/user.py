#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# user.py
# bragg-detect: detecting Bragg peaks in 3D X-ray/neutron images
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" user interfaces """

import matplotlib.pyplot as plt
import numpy as np

from bragg_detect.core import detect_peaks
from bragg_detect.utils import plot_image, plot_peaks


def detect_bragg_peaks(data,
                       # slicing
                       large_peak_size, detect_block_size=5,
                       detect_block_overlap=2, verify_block_size=10,
                       # 2d blobs by LoG
                       min_sigma=None, max_sigma=None, num_sigma=5,
                       threshold=.2, overlap=.5, log_scale=False,
                       # 3d peak selection
                       strategy_3d='individual', n_components_bgm=5,
                       n_init_bgm=1,
                       # others
                       workers=1, verbose=True):
    """
    Detect Bragg peaks.

    :param data: 3d data
    :param large_peak_size: approximate size of the largest peaks in data,
        array like (size_x, size_y, size_z);
    :param detect_block_size: size of the detection blocks relative to
        `large_peak_size`; default is 5
    :param detect_block_overlap: overlap of the detection blocks
        relative to `large_peak_size`; default is 2
    :param verify_block_size: size of the verification blocks relative
        to `large_peak_size`; default is 10
    :param min_sigma: min_sigma for blob_log() of scikit-image;
        default is None, or large_peak_size // 4
    :param max_sigma: max_sigma for blob_log() of scikit-image;
        default is None, or large_peak_size
    :param num_sigma: num_sigma for blob_log() of scikit-image;
        default is 5
    :param threshold: threshold for blob_log() of scikit-image;
        default is 0.2
    :param overlap: overlap for blob_log() of scikit-image;
        default is 0.5
    :param log_scale: log_scale for blob_log() of scikit-image;
        default is False
    :param strategy_3d: strategy for 3d morphological analysis; can be
        `individual` or `bgm_clustering`; default is `individual`
    :param n_components_bgm: n_components for BayesianGaussianMixture() of
        scikit-learn, used only for `bgm_clustering` strategy; default is 5
    :param n_init_bgm: n_init for BayesianGaussianMixture() of
        scikit-learn, used only for `bgm_clustering` strategy; default is 1
    :param workers: number of workers; default is 1
    :param verbose: verbose info during running; default is True
    :return: detected Bragg peaks
    """
    # parameters for block slicing
    large_peak_size = np.array(large_peak_size)
    shift = np.floor((detect_block_size - detect_block_overlap) *
                     large_peak_size).astype(int)
    x_loc = np.arange(0, data.shape[0], shift[0])
    y_loc = np.arange(0, data.shape[1], shift[1])
    z_loc = np.arange(0, data.shape[2], shift[2])
    width = np.ceil(detect_block_size * large_peak_size).astype(int)
    x_width = np.full(len(x_loc), width[0])
    y_width = np.full(len(y_loc), width[1])
    z_width = np.full(len(z_loc), width[2])
    extend = np.ceil((verify_block_size - detect_block_size) *
                     large_peak_size * .5).astype(int)
    x_extend = np.full(len(x_loc), extend[0])
    y_extend = np.full(len(y_loc), extend[1])
    z_extend = np.full(len(z_loc), extend[2])

    # parameters for LoG
    if min_sigma is None:
        min_sigma = np.maximum(large_peak_size // 4, 1)
    if max_sigma is None:
        max_sigma = np.maximum(large_peak_size, 1)

    # verbose input
    if verbose:
        print(f'Bragg peak detection started:')
        print(f'* Data dimension: {np.array(data.shape)}')
        print(f'* Large peak size: {large_peak_size}')
        print(f'* Widths of detection blocks: {width}')
        print(f'* Widths of verification blocks: {width + 2 * extend}')
        print(f'* Block slicing: {[len(x_loc), len(y_loc), len(z_loc)]} '
              f'-> {len(x_loc) * len(y_loc) * len(z_loc)} blocks')
        print(f'* LoG arguments:')
        print(f'  - min_sigma: {min_sigma}')
        print(f'  - max_sigma: {max_sigma}')
        print(f'  - num_sigma: {num_sigma}')
        print(f'  - threshold: {threshold}')
        print(f'  - overlap: {overlap}')
        print(f'  - log_scale: {log_scale}')
        print(f'* Strategy for 3d morphological analysis: {strategy_3d}')
        if strategy_3d == 'bgm_clustering':
            print(f'* BGM arguments:')
            print(f'  - n_components: {n_components_bgm}')
            print(f'  - n_init: {n_init_bgm}')
        print(f'* Number of workers: {workers}')

    # detect peaks
    peaks, wall_time = detect_peaks(
        data, strategy_3d,
        x_loc, x_width, x_extend,
        y_loc, y_width, y_extend,
        z_loc, z_width, z_extend,
        min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale,
        None, n_components_bgm, n_init_bgm, workers, verbose)

    # verbose results
    if verbose:
        print(f'Bragg peak detection finished:')
        print(f'* Peaks detected: {len(peaks)}')
        print(f'* Time elapsed: {wall_time:.1f} sec')
    return peaks


def plot_peaks_over_data(data, plot_size=None, vmax=(1, 1, 1),
                         axis_on=False, peak_sets=None, save_to_file=None):
    """
    Plot peaks over 3d data projections

    :param data: 3d data
    :param plot_size: resize images for better visualization; default is None
    :param vmax: set vmax for better visualization; default is (1, 1, 1)
    :param axis_on: show axis ticks; default is False
    :param peak_sets: sets of peaks, list of (peaks, color, marker, scale, lw);
        default is None
    :param save_to_file: save figure to this file; default is None
    :return:
    """
    # projections
    sum0 = data.sum(axis=0).astype(float)
    sum0 /= np.max(sum0)
    sum1 = data.sum(axis=1).astype(float)
    sum1 /= np.max(sum1)
    sum2 = data.sum(axis=2).astype(float)
    sum2 /= np.max(sum2)
    sum2 = np.transpose(sum2)

    # figure setup
    if plot_size is None:
        plot_size = data.shape
    fig, axes = plt.subplots(2, 2, dpi=200, gridspec_kw={
        'width_ratios': [plot_size[2], plot_size[0]],
        'height_ratios': [plot_size[1], plot_size[0]]})

    # image
    plot_image(axes[0, 0], sum0, (plot_size[1], plot_size[2]),
               vmax=vmax[0], axes_labels=['Z', 'Y'] if axis_on else None)
    plot_image(axes[1, 0], sum1, (plot_size[0], plot_size[2]),
               vmax=vmax[1], axes_labels=['Z', 'X'] if axis_on else None)
    plot_image(axes[0, 1], sum2, (plot_size[1], plot_size[0]),
               vmax=vmax[2], axes_labels=['X', 'Y'] if axis_on else None)
    plot_image(axes[1, 1], sum2.T, (plot_size[0], plot_size[0]),
               vmax=vmax[2], axes_labels=['Y', 'X'] if axis_on else None)

    # peaks
    if peak_sets is None:
        peak_sets = []
    for peak, color, marker, scale, lw in peak_sets:
        plot_peaks(axes[0, 0], (peak[:, 2], peak[:, 1]), sum0,
                   plot_size=(plot_size[1], plot_size[2]),
                   color=color, marker=marker, s=scale, lw=lw)
        plot_peaks(axes[1, 0], (peak[:, 2], peak[:, 0]), sum1,
                   plot_size=(plot_size[0], plot_size[2]),
                   color=color, marker=marker, s=scale, lw=lw)
        plot_peaks(axes[0, 1], (peak[:, 0], peak[:, 1]), sum2,
                   plot_size=(plot_size[1], plot_size[0]),
                   color=color, marker=marker, s=scale, lw=lw)
        plot_peaks(axes[1, 1], (peak[:, 1], peak[:, 0]), sum2.T,
                   plot_size=(plot_size[0], plot_size[0]),
                   color=color, marker=marker, s=scale, lw=lw)

    # save
    plt.tight_layout()
    if save_to_file is not None:
        plt.savefig(save_to_file)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print('Hello world.')
