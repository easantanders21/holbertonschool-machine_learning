#!/usr/bin/env python3
"""4-dropout_forward_prop
contains the function l2_reg_create_layer
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        Z = (np.matmul(weights["W"+str(i+1)],
             cache["A"+str(i)]) +
             weights["b"+str(i+1)])
        drop = np.random.binomial(1, keep_prob, size=Z.shape)
        if i == L-1:
            cache["A"+str(i+1)] = np.exp(Z)/np.sum(np.exp(Z),
                                                   axis=0, keepdims=True)
        else:
            cache["A"+str(i+1)] = np.tanh(Z)
            cache["D"+str(i+1)] = drop
            cache["A"+str(i+1)] = (cache["A"+str(i+1)] *
                                   cache["D"+str(i+1)])/keep_prob
    return cache
