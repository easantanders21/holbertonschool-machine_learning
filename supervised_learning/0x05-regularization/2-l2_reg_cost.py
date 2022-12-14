#!/usr/bin/env python3
"""2-l2_reg_cost module
contains the function l2_reg_cost
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization
    """
    l2_reg_loss = tf.losses.get_regularization_losses()
    J = cost + l2_reg_loss
    return J
