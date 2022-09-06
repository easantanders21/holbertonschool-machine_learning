#!/usr/bin/env python3
""" Create Confusion """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Create Confusion """
    confusion_matrix = np.matmul(labels.T, logits)
    return confusion_matrix
