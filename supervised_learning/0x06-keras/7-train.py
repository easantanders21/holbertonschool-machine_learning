#!/usr/bin/env python3
"""
module 7-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and also
        analyze validation data using early stopping with learning rate decay
    """
    def scheduler(epoch):
        """Callback to update the learning rate decay
        """
        lr = alpha/(1+decay_rate*epoch)
        return lr

    callbacks = []

    if early_stopping and validation_data:
        EarlyStopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience)
        callbacks.append(EarlyStopping)

    if learning_rate_decay and validation_data:
        LearningRateScheduler = K.callbacks.\
            LearningRateScheduler(schedule=scheduler,
                                  verbose=1)
        callbacks.append(LearningRateScheduler)

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=shuffle)
    return history
