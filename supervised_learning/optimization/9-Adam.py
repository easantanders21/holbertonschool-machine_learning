#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """  Adam optimization algorithm """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.square(grad)
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    var -= alpha * v_corrected / (epsilon + np.sqrt(s_corrected))
    return var, v, s
