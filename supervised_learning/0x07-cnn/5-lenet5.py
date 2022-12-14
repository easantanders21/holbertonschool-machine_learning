#!/usr/bin/env python3
"""
5-lenet5 module
contains function lenet5
"""
import tensorflow.keras as K


def lenet5(x):
    """Builds a modified version of the LeNet-5 architecture using keras
    """
    initializer = K.initializers.he_normal(seed=None)

    conv_layer_1 = K.layers.Conv2D(filters=6,
                                   kernel_size=5,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=initializer)(x)

    p_layer_2 = K.layers.MaxPool2D(pool_size=[2, 2],
                                   strides=2)(conv_layer_1)

    conv_layer_3 = K.layers.Conv2D(filters=16, kernel_size=5,
                                   padding='valid', activation='relu',
                                   kernel_initializer=initializer)(p_layer_2)

    p_layer_4 = K.layers.MaxPool2D(pool_size=[2, 2],
                                   strides=2)(conv_layer_3)

    # Flattening between conv and dense layers
    flatten_layer = K.layers.Flatten()(p_layer_4)

    fc_layer_5 = K.layers.Dense(units=120, activation='relu',
                                kernel_initializer=initializer)(flatten_layer)

    fc_layer_6 = K.layers.Dense(units=84, activation='relu',
                                kernel_initializer=initializer)(fc_layer_5)

    output_layer = K.layers.Dense(units=10, activation='softmax',
                                  kernel_initializer=initializer)(fc_layer_6)

    model = K.Model(inputs=x, outputs=output_layer)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
