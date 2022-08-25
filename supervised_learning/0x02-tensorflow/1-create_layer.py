#!/usr/bin/env python3
""" create layer """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ creates the layers for the neural network """
    init = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=init)
    y = layer(prev)
    return y
