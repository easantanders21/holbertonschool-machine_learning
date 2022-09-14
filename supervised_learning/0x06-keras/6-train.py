#!/usr/bin/env python3
"""
module 6-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and also
        analyze validation data using early stopping.
    """
    callbacks = []
    if early_stopping and validation_data:
        EarlyStopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience)
        callbacks.append(EarlyStopping)
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=shuffle)
    return history
