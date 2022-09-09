#!/usr/bin/env python3
"""0-l2_reg_cost module
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization
    """
    Frobenius_norm = 0
    for layer in range(1, L + 1):
        Frobenius_norm += np.linalg.norm(weights['W{}'.format(layer)])
    J = cost + (lambtha/(2*m)) * Frobenius_norm
    return J
