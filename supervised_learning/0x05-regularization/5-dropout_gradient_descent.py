#!/usr/bin/env python3
"""5-dropout_gradient_descent
contains the function dropout_gradient_descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout regularization
        using gradient descent.
    """
    auxiliar_weights = weights.copy()
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache["A"+str(i)]
        if i == L:
            dZ = A-Y
        else:
            W = auxiliar_weights["W"+str(i+1)]
            dZ = np.matmul(W.T, dZ)*(1-A**2)
            dZ *= cache["D"+str(i)]
            dZ /= keep_prob
        dW = np.matmul(dZ, cache["A"+str(i-1)].T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        weights["W"+str(i)] = auxiliar_weights["W"+str(i)]-alpha*dW
        weights["b"+str(i)] = auxiliar_weights["b"+str(i)]-alpha*db
