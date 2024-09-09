#!/usr/bin/env python3
"""Function that calculates the softmax
 cross-entropy loss of a prediction"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
