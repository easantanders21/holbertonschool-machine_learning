#!/usr/bin/env python3
"""
1-pool_forward module
contains function pool_forward
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs forward propagation over a pooling layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride[0], stride[1]

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    if padding == "valid":
        ph = 0
        pw = 0

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph),
                             (pw, pw), (0, 0)),
                    mode='constant',
                    constant_values=(0))

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for l in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    dA[l, i * sh:i * sh + kh,
                       j * sw:j * sw + kw, :] += (W[:, :, :, k] *
                                                  dZ[l, i, j, k])
                    dW[:, :, :, k] += (A_prev[l, i * sh:i * sh + kh,
                                              j * sw:j * sw + kw, :] *
                                       dZ[l, i, j, k])

    dA = dA[:, ph:dA.shape[1] - ph, pw:dA.shape[2] - pw, :]

    return dA, dW, db
