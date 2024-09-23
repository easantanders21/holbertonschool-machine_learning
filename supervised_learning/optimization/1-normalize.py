#!/usr/bin/env python3
""" Normalize """


def normalize(X, m, s):
    """ Normalize """
    X_normalized = (X - m)/s
    return X_normalized
