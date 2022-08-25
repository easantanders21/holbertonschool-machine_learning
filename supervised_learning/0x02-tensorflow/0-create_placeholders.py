#!/usr/bin/env python3
"""
File for placeholders
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ Returns two placeholders, x and y, for the neural network """
    x = tf.placeholder(dtype="float32", shape=(None, nx), name="x")
    y = tf.placeholder(dtype="float32", shape=(None, classes), name="y")
    return x, y
