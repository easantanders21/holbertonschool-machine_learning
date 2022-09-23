#!/usr/bin/env python3
"""
1-pool_forward module
contains function pool_forward
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    output_h = (h_prev - kh)//sh + 1
    output_w = (w_prev - kw)//sw + 1

    output = np.zeros((m, output_h, output_w, c_prev))

    for j in range(output_h):
        for i in range(output_w):
            if mode == "max":
                output[:, j, i, :] = (np.max(A_prev[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2)))
            if mode == "avg":
                output[:, j, i, :] = (np.mean(A_prev[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2)))
    return output
