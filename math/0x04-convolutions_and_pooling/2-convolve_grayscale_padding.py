#!/usr/bin/env python3
"""
module 2-convolve_grayscale_padding
contains function convolve_grayscale_padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images custom padding
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph, pw = padding[0], padding[1]

    h_custom_padded = h + 2*ph - kh + 1
    w_custom_padded = w + 2*pw - kw + 1

    custom_padded_images = np.pad(images,
                                  pad_width=((0, 0), (ph, ph), (pw, pw)),
                                  mode='constant',
                                  constant_values=0)

    output = np.zeros((m, h_custom_padded, w_custom_padded))

    for j in range(h_custom_padded):
        for i in range(w_custom_padded):
            output[:, j, i] = ((kernel * custom_padded_images[:,
                               j: j + kh, i: i + kw]).sum(axis=(1, 2)))
    return output
