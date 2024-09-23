#!/usr/bin/env python3
""" RMSProp Upgraded """
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ RMSProp Upgraded """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                          epsilon=epsilon)
    train_op = optimizer.minimize(loss=loss)
    return train_op
