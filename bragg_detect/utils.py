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
    if dim == 0:
        return [1, 2]
    elif dim == 1:
        return [0, 2]
    elif dim == 2:
        return [0, 1]
    else:
        assert False


def to_flattened(structured, dims):
    if len(dims) == 3:
        return (structured[:, 0] * dims[1] * dims[2]
                + structured[:, 1] * dims[2] + structured[:, 2])
    else:
        return structured[:, 0] * dims[1] + structured[:, 1]


def to_structured(flattened, dims):
    if len(dims) == 3:
        x = flattened // (dims[1] * dims[2])
        y = (flattened - x * dims[1] * dims[2]) // dims[2]
        z = (flattened - x * dims[1] * dims[2] - y * dims[2])
        return np.transpose(np.array([x, y, z]))
    else:
        x = flattened // dims[1]
        y = flattened - x * dims[1]
        return np.transpose(np.array([x, y]))


def plot_image(ax, image, plot_size=None, vmax=1, axis_on=False):
    if plot_size is None:
        plot_size = image.shape
    resized = resize(image, plot_size)
    if not axis_on:
        ax.imshow(resized, vmax=vmax)
        ax.axis('off')
    else:
        ax.imshow(resized, vmax=vmax, aspect='auto',
                  extent=(0, image.shape[1], 0, image.shape[0]))


def plot_peaks(ax, peaks, image, plot_size=None,
               color='r', marker='o', s=1, lw=1):
    ax.scatter(peaks[0] / image.shape[1] * plot_size[1],
               peaks[1] / image.shape[0] * plot_size[0],
               facecolors=color, marker=marker, s=s, lw=lw)
