#!/usr/bin/env python3
"""
module 2-optimize
contains the function optimize_model
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics
    """
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)

    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
