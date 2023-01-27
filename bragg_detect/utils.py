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
    if dim not in range(3):
        raise ValueError('please provide a dimension 0, 1, or 2')

    return [i for i in range(3) if i != dim]


def to_flattened(structured_array, dims):
    """
    Converts an array index from a 2 or 3D array to the equivalent 1D index
    :param structured_array: np.ndarray[int], [n, 2] or [n, 3]
    :param dims: np.ndarray, 2 or 3 valued array
    :return: np.ndarray[int], the flattened array
    """
    if len(dims) == 3:
        return (structured_array[:, 0] * dims[1] * dims[2]
                + structured_array[:, 1] * dims[2] + structured_array[:, 2])
    else:
        return structured_array[:, 0] * dims[1] + structured_array[:, 1]


def to_structured(flattened_array, dims):
    """
    Takes a flattened dataset and forms a multi-dimensional array of shape dims
    TODO: find out what this does
    :param flattened_array: np.ndarray[float], array to be reshaped
    :param dims: np.ndarray[int], dimensions to shape the data into
    :return:
    """
    if len(dims) == 3:
        x = flattened_array // (dims[1] * dims[2])
        y = (flattened_array - x * dims[1] * dims[2]) // dims[2]
        z = (flattened_array - x * dims[1] * dims[2] - y * dims[2])
        return np.transpose(np.array([x, y, z]))
    else:
        x = flattened_array // dims[1]
        y = flattened_array - x * dims[1]
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
