#!/usr/bin/env python3
"""1-l2_reg_gradient_descent.py module
contains the function l2_reg_gradient_descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient
        descent with L2 regularization.
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
        dW = np.matmul(dZ, cache["A"+str(i-1)].T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dW_reg = dW+((lambtha / m) * auxiliar_weights["W" + str(i)])
        weights["W"+str(i)] = auxiliar_weights["W"+str(i)]-alpha*dW_reg
        weights["b"+str(i)] = auxiliar_weights["b"+str(i)]-alpha*db
