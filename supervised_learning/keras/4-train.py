#!/usr/bin/env python3
"""
module 4-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent. Normally,
        it is a good idea to shuffle, but for reproducibility,
        we have chosen to set the default to False
    """
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
