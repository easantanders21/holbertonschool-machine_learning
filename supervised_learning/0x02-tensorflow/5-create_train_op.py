#!/usr/bin/env python3
"""Function that creates the training operation for the network"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network"""
    gradient = tf.train.GradientDescentOptimizer(alpha).compute_gradients(loss)
    return tf.train.GradientDescentOptimizer(alpha).apply_gradients(gradient)
