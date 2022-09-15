#!/usr/bin/env python3
"""
module 6-pool
contains function pool
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs a convolution on images using multiple kernels
    """
    c = images.shape[3]
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    output_h = (h - kh)//sh + 1
    output_w = (w - kw)//sw + 1

    output = np.zeros((m, output_h, output_w, c))

    for j in range(output_h):
        for i in range(output_w):
            if mode == "max":
                output[:, j, i, :] = (np.max(images[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2)))
            if mode == "avg":
                output[:, j, i, :] = (np.mean(images[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2)))
    return output
