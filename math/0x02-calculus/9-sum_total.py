#!/usr/bin/env python3
""" function thar return sumation i squared """


def summation_i_squared(n):
    """ function thar return sumation i squared """
    if n is None or n <= 0:
        return None
    return n * (n + 1) * (2 * n + 1) / 6
