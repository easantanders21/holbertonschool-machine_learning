#!/usr/bin/env python3
"""
module 1-input
contains the function build_model
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library with the Input
    """
    regularizer = K.regularizers.l2(lambtha)
    # you start from Input
    inputs = K.Input(shape=(nx,))
    # you chain layer calls to specify the model's forward pass
    outputs = K.layers.Dense(units=layers[0], activation=activations[0],
                             kernel_regularizer=regularizer,
                             input_shape=(nx,))(inputs)
    for (layer, activation) in zip(layers[1:], activations[1:]):
        outputs = K.layers.Dropout(rate=1-keep_prob)(outputs)
        outputs = K.layers.Dense(units=layer, activation=activation,
                                 kernel_regularizer=regularizer)(outputs)
    # finally you create your model from inputs and outputs
    model = K.Model(inputs=inputs, outputs=outputs)
    return model
