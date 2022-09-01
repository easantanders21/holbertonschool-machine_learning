#!/usr/bin/env python3
""" normalization constants"""
import numpy as np


def normalization_constants(X):
    """ normalization constants """
    mean = np.mean(X, axis=0)
    standar_deviation = np.std(X, axis=0)
    return mean, standar_deviation
