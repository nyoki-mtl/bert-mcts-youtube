import numpy as np


def greedy(logits):
    return np.asarray(logits).argmax()


def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return probabilities

