#!/usr/bin/env python3
""" Moving Average """


def moving_average(data, beta):
    """ Moving Average """
    moving_averages = []

    vt = 0
    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        moving_averages.append(vt / (1 - beta ** (i + 1)))

    return moving_averages
