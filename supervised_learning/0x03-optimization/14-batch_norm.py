#!/usr/bin/env python3
""" Batch Normalization Upgraded """
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """ Batch Normalization Upgraded """
    initializer = tf.\
        contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n, kernel_initializer=initializer,
                                 name="base_layer")
    X = base_layer(prev)

    mean, variance = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name="beta")
    epsilon = 1e-8
    Z = tf.nn.batch_normalization(x=X, mean=mean, variance=variance,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=epsilon, name="Z")
    A = activation(Z)
    return A
