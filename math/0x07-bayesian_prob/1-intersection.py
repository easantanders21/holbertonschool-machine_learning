#!/usr/bin/env python3
"""
module 1-intersection
contains function intersection
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data
    with the various hypothetical probabilities
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')

    if type(x) is not int or x < 0:
        raise ValueError('x must be an integer that is ' +
                         'greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if type(P) is not np.ndarray or len(P.shape) is not 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    if np.any(Pr < 0) or np.any(P > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose([np.sum(Pr)], [1.])[0]:
        raise ValueError('Pr must sum to 1')

    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)
    cs = num / den
    likelihood = cs * (P ** x) * (1 - P) ** (n - x)
    my_intersection = likelihood * Pr
    return my_intersection
