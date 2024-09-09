#!/usr/bin/env python3
""" forward propagation """


def forward_prop(x, layer_sizes=[], activations=[], index=0):
    """ forward propagation """
    if index >= len(layer_sizes):
        return x
    create_layer = __import__('1-create_layer').create_layer
    layer = create_layer(x, layer_sizes[index], activations[index])
    return forward_prop(layer, layer_sizes, activations, index + 1)
