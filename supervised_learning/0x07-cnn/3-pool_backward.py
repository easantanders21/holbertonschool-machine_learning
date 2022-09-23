#!/usr/bin/env python3
"""
1-pool_forward module
contains function pool_forward
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    dA_prev = np.zeros(A_prev.shape)

    for l in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):
                    h_start = i * sh
                    h_end = h_start + kh
                    w_start = j * sw
                    w_end = w_start + kw
                    if mode == 'max':
                        value = np.max(A_prev[l, h_start:h_end, w_start:w_end,
                                              c])
                        mask = np.where(A_prev[l, h_start:h_end, w_start:w_end,
                                               c] == value, 1, 0)
                        mask = mask * dA[l, i, j, c]
                    if mode == 'avg':
                        mask = np.ones(kernel_shape)*(dA[l, i, j, c]/(kh*kw))
                    dA_prev[l, h_start:h_end, w_start:w_end, c] += mask
    return dA_prev
