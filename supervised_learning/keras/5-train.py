#!/usr/bin/env python3
"""
module 5-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and also
    """
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
