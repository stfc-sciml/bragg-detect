#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# utils.py
# bragg-detect: detecting Bragg peaks in 3D X-ray/neutron images
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" utilities """

import numpy as np
from skimage.transform import resize


def get_other_dims(dim):
    """
    Takes an integer in [0, 1, 2] and returns the two other integers
    :param dim: int, the known dimension
    :return: array[int] of the two values which aren't the input
    """
    arr = [0, 1, 2]
    if dim in arr:
        arr.remove(dim)
        return arr
    else:
        raise ValueError('please provide a dimension 0, 1, or 2')


def to_flattened(structured, dims):
    """
    Flattens a multi-dimensional array, summing the values
    TODO: find out what this does!
    :param structured: np.ndarray, 2D or 3D array
    :param dims: int, number of dimensinos of array
    :return: list[float], the flattened array
    """
    if len(dims) == 3:
        return (structured[:, 0] * dims[1] * dims[2]
                + structured[:, 1] * dims[2] + structured[:, 2])
    else:
        return structured[:, 0] * dims[1] + structured[:, 1]


def to_structured(flattened, dims):
    """
    Takes a flattened dataset and returns it to a multi-dimensional array
    TODO: find out what this does!
    :param flattened: list[float], array to be reshaped
    :param dims: int, number of dimensions to shape the data into
    :return:
    """
    if len(dims) == 3:
        x = flattened // (dims[1] * dims[2])
        y = (flattened - x * dims[1] * dims[2]) // dims[2]
        z = (flattened - x * dims[1] * dims[2] - y * dims[2])
        return np.transpose(np.array([x, y, z]))
    else:
        x = flattened // dims[1]
        y = flattened - x * dims[1]
        return np.transpose(np.array([x, y]))


def plot_image(ax, image, plot_size=None, vmax=1, axes_labels=None):
    """
    Takes 2D data and plots it, with optional resizing of the image
    :param ax: array of axes
    :param image: Matplotlib Figure
    :param plot_size: Optional[array[float]], size of the figure, optional
    defaults to None (no scaling applied)
    :param vmax: float, max range colormap covers, defaults to 1
    :param axes_labels: Optional[array[str]], names to be shown on axes,
    horizontal then vertical axes
    :return: None
    """
    if plot_size is None:
        plot_size = image.shape
    resized = resize(image, plot_size)

    if not axes_labels:
        ax.axis('off')
    else:
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xlabel(axes_labels[0], fontsize=6)
        ax.set_ylabel(axes_labels[1], fontsize=6)

    ax.imshow(resized, vmax=vmax, origin='lower',
              aspect=image.shape[1] * plot_size[0] / (
                      image.shape[0] * plot_size[1]),
              extent=(0, image.shape[1], 0, image.shape[0]))


def plot_peaks(ax, peaks, color='r', marker='o', s=1, lw=1):
    """
    Plot a scatter graph of the peak positions onto ax
    :param ax: plt.subplots.axes
    :param peaks: array, list of coorinates of peaks
    :param color: str, colour of markers
    :param marker: str, marker style
    :param s: int, marker size in points**2
    :param lw: float, linewidth
    :return: None
    """
    ax.scatter(peaks[0], peaks[1],
               facecolors=color, marker=marker, s=s, lw=lw, zorder=4)
    #  zorder>3 ensures points always drawn on top
