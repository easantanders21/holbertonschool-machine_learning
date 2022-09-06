#!/usr/bin/env python3
""" Specificity """
import numpy as np


def specificity(confusion):
    """ Specificity """
    TP = np.diag(confusion)
    ACTUAL = confusion.sum(axis=1)
    FN = ACTUAL - TP
    PREDICTED = confusion.sum(axis=0)
    FP = PREDICTED - TP
    TN = confusion.sum() - (FP + FN + TP)
    specificity = TN / (TN + FP)
    return specificity
