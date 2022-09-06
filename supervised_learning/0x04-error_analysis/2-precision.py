#!/usr/bin/env python3
""" Precision """
import numpy as np


def precision(confusion):
    """ Precision """
    TP = np.diag(confusion)
    PREDICTED = confusion.sum(axis=0)
    FP = PREDICTED - TP
    precision = TP / (TP + FP)
    return precision
