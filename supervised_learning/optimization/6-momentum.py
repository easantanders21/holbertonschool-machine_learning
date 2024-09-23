#!/usr/bin/env python3
""" Momentum Upgraded """
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """ Momentum Upgraded """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss=loss)
    return train_op
