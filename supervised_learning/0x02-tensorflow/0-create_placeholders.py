#!/usr/bin/env python3
""" create place holders """
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ return two placeholders """
    x = tf.placeholder(dtype="float32", shape=(None, nx), name="x")
    y = tf.placeholder(dtype="float32", shape=(None, classes), name="y")
    return x, y
