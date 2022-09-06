#!/usr/bin/env python3
""" Sensitivity """
import numpy as np


def sensitivity(confusion):
    """ Sensitivity """
    TP = np.diag(confusion)
    ACTUAL = confusion.sum(axis=1)
    FN = ACTUAL - TP
    sensitivity = TP / (TP + FN)
    return sensitivity
