#!/usr/bin/env python3
""" create layer """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ create layer """
    new_layer = tf.layers.Dense(
        n, activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        name="layer")
    return new_layer(prev)
