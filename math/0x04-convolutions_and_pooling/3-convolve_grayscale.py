#!/usr/bin/env python3
"""
module 3-convolve_grayscale
contains function convolve_grayscale
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]

    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]

    if padding == "same":
        ph = ((h - 1)*sh + kh - h)//2 + 1
        pw = ((w - 1)*sw + kw - w)//2 + 1

    if padding == "valid":
        ph = 0
        pw = 0

    custom_padded_images = np.pad(images,
                                  pad_width=((0, 0), (ph, ph), (pw, pw)),
                                  mode='constant',
                                  constant_values=0)

    h_custom_padded = (custom_padded_images.shape[1] - kh)//sh + 1
    w_custom_padded = (custom_padded_images.shape[2] - kw)//sw + 1

    output = np.zeros((m, h_custom_padded, w_custom_padded))

    for j in range(h_custom_padded):
        for i in range(w_custom_padded):
            output[:, j, i] = ((kernel * custom_padded_images[:,
                               j*sh: j*sh + kh,
                               i*sw: i*sw + kw]).sum(axis=(1, 2)))
    return output
