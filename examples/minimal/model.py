"""Synthetic linear reservoir; inputs and outputs have arbitrary consistent units."""
import numpy as np


def simulate(forcing, release, gain):
    storage = 0.0
    flow = []
    for value in forcing:
        storage += float(value) * gain
        outflow = release * storage
        storage -= outflow
        flow.append(outflow)
    return np.asarray(flow)
