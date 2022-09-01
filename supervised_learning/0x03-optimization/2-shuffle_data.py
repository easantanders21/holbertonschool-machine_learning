#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """ Shuffle Data """
    shufled_rows = np.random.permutation(X.shape[0])
    return X[shufled_rows], Y[shufled_rows]
