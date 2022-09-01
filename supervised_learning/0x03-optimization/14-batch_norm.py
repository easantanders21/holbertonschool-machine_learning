#!/usr/bin/env python3
""" Batch Normalization Upgraded """

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """ Batch Normalization Upgraded """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.ones([n])
    beta = tf.zeros([n])

    epsilon = 1e-8
    batch_normalization_output = tf.nn.batch_normalization(
        x=layer(prev), mean=mean,
        variance=variance, offset=beta,
        scale=gamma, variance_epsilon=epsilon)

    return activation(batch_normalization_output)
