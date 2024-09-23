#!/usr/bin/env python3
""" Batch Normalization """


def batch_norm(Z, gamma, beta, epsilon):
    """ Batch Normalization """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    standard_deviation = (variance + epsilon) ** 0.5
    Z_normalized = (Z - mean) / standard_deviation
    Z_tilde = gamma * Z_normalized + beta
    return Z_tilde
