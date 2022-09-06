#!/usr/bin/env python3
""" F1 score """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ F1 score """
    TPR = sensitivity(confusion)
    PPV = precision(confusion)
    f1_score = 2 * (PPV * TPR / (PPV + TPR))
    return f1_score
