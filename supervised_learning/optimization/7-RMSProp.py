#!/usr/bin/env python3
""" RMSProp """


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ RMSProp """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * (1 / (s ** 0.5 + epsilon)) * grad
    return var, s
