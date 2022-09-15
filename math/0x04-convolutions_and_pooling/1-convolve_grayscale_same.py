#!/usr/bin/env python3
"""
module 1-convolve_grayscale_same
contains function convolve_grayscale_same
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a valid convolution on grayscale images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    ph = ((kh - 1) // 2) if kh % 2 else (kh // 2)
    pw = ((kw - 1) // 2) if kw % 2 else (kw // 2)

    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant',
                           constant_values=0)

    output = np.zeros((m, h, w))

    for j in range(h):
        for i in range(w):
            output[:, j, i] = ((kernel * padded_images[:,
                               j: j + kh, i: i + kw]).sum(axis=(1, 2)))
    return output
