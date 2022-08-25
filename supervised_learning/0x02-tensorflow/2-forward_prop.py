#!/usr/bin/env python3
""" forward propagation """
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward propagation """
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return
