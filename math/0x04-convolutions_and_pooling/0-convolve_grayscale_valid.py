#!/usr/bin/env python3
"""
module 0-convolve_grayscale_valid
contains function convolve_grayscale_valid
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))
    for j in range(output_h):
        for i in range(output_w):
            output[:, j, i] = ((kernel * images[:, j: j + kh, i: i + kw]).
                               sum(axis=(1, 2)))
    return output
