#!/usr/bin/env python3
""" forward propagation """
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward propagation """
    for i in range(len(layer_sizes)):
        if i == 0:
            pre = create_layer(x, layer_sizes[i], activations[i])
        else:
            pre = create_layer(prev, layer_sizes[i], activations[i])
    return pre
