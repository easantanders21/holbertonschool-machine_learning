#!/usr/bin/env python3
"""3-l2_reg_create_layer module
contains the function l2_reg_create_layer
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer,
                            name="layer")
    y = layer(prev)
    return y
