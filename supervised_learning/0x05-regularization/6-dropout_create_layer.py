#!/usr/bin/env python3
"""6-dropout_create_layer module
contains the function dropout_create_layer
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a tensorflow layer that includes L2 regularization
    """
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name="layer")
    drop = tf.layers.Dropout(rate=keep_prob, name="drop")
    y = drop(layer(prev))
    return y
