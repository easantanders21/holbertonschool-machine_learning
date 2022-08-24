#!/usr/bin/env python3
"""
One-Hot Encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Converts a numeric label vector into a one-hot matrix """
    if type(Y) != np.ndarray:
        return None
    try:
        one_hot = np.eye(classes)[Y].T
        return one_hot
    except Exception:
        return None
