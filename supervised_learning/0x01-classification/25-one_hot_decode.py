#!/usr/bin/env python3
"""
One-Hot Decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a one-hot matrix into a vector of labels """
    if type(one_hot) != np.ndarray or one_hot.shape < (2, 2):
        return None
    try:
        decode = np.argmax(one_hot.T, axis=1)
        return decode
    except Exception:
        return None
