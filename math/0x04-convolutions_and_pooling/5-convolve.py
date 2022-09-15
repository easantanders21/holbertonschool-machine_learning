#!/usr/bin/env python3
"""
module 5-convolve
contains function convolve_channels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels
    """
    c = images.shape[3]
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw, nc = kernels.shape[0], kernels.shape[1], kernels.shape[3]
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
                                  pad_width=((0, 0),
                                             (ph, ph), (pw, pw),
                                             (0, 0)),
                                  mode='constant',
                                  constant_values=0)

    h_custom_padded = (custom_padded_images.shape[1] - kh)//sh + 1
    w_custom_padded = (custom_padded_images.shape[2] - kw)//sw + 1

    output = np.zeros((m, h_custom_padded, w_custom_padded, nc))

    for j in range(h_custom_padded):
        for i in range(w_custom_padded):
            for k in range(nc):
                output[:, j, i, k] = ((kernels[:, :, :, k] *
                                      custom_padded_images[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw,
                                      :]).sum(axis=(1, 2, 3)))
    return output